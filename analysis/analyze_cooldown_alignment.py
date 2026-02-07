#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

CORE_PATH = RESULTS_DIR / "core_unmatched_live_analysis.csv"
NO_BT_DECISIONS_PATH = RESULTS_DIR / "signal_no_bt_trade_with_decisions.csv"
UNMATCHED_LIVE_PATH = RESULTS_DIR / "parity_unmatched_live.csv"
MATCHED_TRADES_PATH = RESULTS_DIR / "parity_matched_trades.csv"
OUT_PATH = RESULTS_DIR / "cooldown_misalignment_diagnostic.csv"


def to_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def normalize_live_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Standardize to:
      - 'symbol'
      - 'exit_ts_live'
      - optional 'entry_ts_live' (rare; if absent we will use exit->exit deltas)
    """
    print(f"[debug] Columns in {label}: {list(df.columns)}")

    # --- SYMBOL ---
    symbol_candidates = [
        "symbol",
        "live_symbol",
        "symbol_live",
        "bt_symbol",        # last resort (shouldn't be used for LIVE, but better than crash)
        "symbol_x",
        "symbol_y",
    ]
    sym_col = next((c for c in symbol_candidates if c in df.columns), None)
    if sym_col is None:
        raise RuntimeError(
            f"Could not find a symbol column in {label}. Tried {symbol_candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    if sym_col != "symbol":
        print(f"[info] Renaming {label} column '{sym_col}' → 'symbol'")
        df = df.rename(columns={sym_col: "symbol"})

    # --- EXIT TS (LIVE) ---
    exit_candidates = [
        "exit_ts_live",
        "exit_ts",
        "live_exit_ts",
        "exit_time",
        "exit_time_live",
        "close_ts",
        "close_ts_live",
        "exit_dt",
        "exit_dt_live",
        "live_Trade time",  # sometimes exists; may or may not be exit, but keep as fallback
        "Trade time",
    ]
    exit_col = next((c for c in exit_candidates if c in df.columns), None)
    if exit_col is None:
        raise RuntimeError(
            f"Could not find a live exit timestamp column in {label}. Tried {exit_candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    if exit_col != "exit_ts_live":
        print(f"[info] Renaming {label} column '{exit_col}' → 'exit_ts_live'")
        df = df.rename(columns={exit_col: "exit_ts_live"})

    # --- ENTRY TS (LIVE) optional ---
    entry_candidates = [
        "entry_ts_live",
        "entry_ts",
        "live_entry_ts",
        "entry_time",
        "entry_time_live",
        "open_ts",
        "open_ts_live",
        "entry_dt",
        "entry_dt_live",
    ]
    entry_col = next((c for c in entry_candidates if c in df.columns), None)
    if entry_col is not None and entry_col != "entry_ts_live":
        print(f"[info] Renaming {label} column '{entry_col}' → 'entry_ts_live'")
        df = df.rename(columns={entry_col: "entry_ts_live"})

    # Parse timestamps
    df["exit_ts_live"] = to_dt_utc(df["exit_ts_live"])
    if "entry_ts_live" in df.columns:
        df["entry_ts_live"] = to_dt_utc(df["entry_ts_live"])

    return df


def main():
    print(f"[info] Loading core unmatched LIVE from: {CORE_PATH}")
    core = pd.read_csv(CORE_PATH)
    if "exit_ts_live" in core.columns:
        core["exit_ts_live"] = to_dt_utc(core["exit_ts_live"])

    print(f"[info] Loading signal_no_bt_trade_with_decisions from: {NO_BT_DECISIONS_PATH}")
    no_bt_dec = pd.read_csv(NO_BT_DECISIONS_PATH)
    if "exit_ts_live" in no_bt_dec.columns:
        no_bt_dec["exit_ts_live"] = to_dt_utc(no_bt_dec["exit_ts_live"])
    if "signal_ts" in no_bt_dec.columns:
        no_bt_dec["signal_ts"] = to_dt_utc(no_bt_dec["signal_ts"])

    core_signal_no_bt = core[core["classification"] == "signal_no_bt_trade"].copy()
    print(f"[info] core_unmatched LIVE with classification='signal_no_bt_trade': {len(core_signal_no_bt)}")

    merged_core = core_signal_no_bt.merge(
        no_bt_dec,
        on=["symbol", "exit_ts_live"],
        how="left",
        suffixes=("", "_dec"),
    )

    cooldown_subset = merged_core[merged_core["reason"] == "cooldown"].copy()
    print(f"[info] signal_no_bt_trade rows with reason='cooldown': {len(cooldown_subset)}")

    print(f"[info] Loading unmatched LIVE trades from: {UNMATCHED_LIVE_PATH}")
    unmatched_live_raw = pd.read_csv(UNMATCHED_LIVE_PATH)

    print(f"[info] Loading matched trades from: {MATCHED_TRADES_PATH}")
    matched_live_raw = pd.read_csv(MATCHED_TRADES_PATH)

    unmatched_live = normalize_live_df(unmatched_live_raw, "parity_unmatched_live")
    matched_live = normalize_live_df(matched_live_raw, "parity_matched_trades")

    live_all = pd.concat([unmatched_live, matched_live], ignore_index=True)
    live_all = live_all.dropna(subset=["symbol", "exit_ts_live"]).copy()
    print(f"[info] Total LIVE trades loaded (after concat & dropna exit): {len(live_all)}")

    # Sort and compute previous exit per symbol
    live_all = live_all.sort_values(["symbol", "exit_ts_live"])
    live_all["prev_exit_ts_live"] = live_all.groupby("symbol")["exit_ts_live"].shift(1)

    # If we have entry_ts_live, compute prev_exit -> this_entry. Otherwise compute exit->exit (lower bound).
    if "entry_ts_live" in live_all.columns and live_all["entry_ts_live"].notna().any():
        print("[info] Using entry_ts_live for cooldown measurement (prev_exit -> this_entry)")
        live_all["hours_since_prev_exit_to_this_entry"] = (
            (live_all["entry_ts_live"] - live_all["prev_exit_ts_live"]).dt.total_seconds() / 3600.0
        )
    else:
        print("[warn] entry_ts_live not available; using exit->exit deltas as a LOWER BOUND on cooldown violations")
        live_all["hours_since_prev_exit_to_this_entry"] = (
            (live_all["exit_ts_live"] - live_all["prev_exit_ts_live"]).dt.total_seconds() / 3600.0
        )

    # Attach deltas to the cooldown cases (join on the live trade’s exit timestamp)
    diag = cooldown_subset.merge(
        live_all[["symbol", "exit_ts_live", "prev_exit_ts_live", "hours_since_prev_exit_to_this_entry"]],
        on=["symbol", "exit_ts_live"],
        how="left",
    )

    # Strict: guaranteed violation only when our lower-bound delta is < 4
    diag["guaranteed_violate_4h_cooldown"] = diag["hours_since_prev_exit_to_this_entry"] < 4

    print("\n=== Cooldown alignment diagnostic (cooldown cases) ===")
    print("guaranteed_violate_4h_cooldown value counts:")
    print(diag["guaranteed_violate_4h_cooldown"].value_counts(dropna=False))

    print("\nhours_since_prev_exit_to_this_entry (summary):")
    print(diag["hours_since_prev_exit_to_this_entry"].describe())

    print(f"\n[info] Saving detailed diagnostic to: {OUT_PATH}")
    diag.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
