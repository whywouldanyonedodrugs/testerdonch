#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bt_intrabar import resolve_first_touch_1m


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_abs(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[1] / p).resolve()
    return p.resolve()


def _load_fixtures(path: Path) -> List[Dict[str, object]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        cases = obj.get("cases", [])
    elif isinstance(obj, list):
        cases = obj
    else:
        raise ValueError(f"Unsupported fixture root type: {type(obj).__name__}")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Fixture file has no cases.")
    out: List[Dict[str, object]] = []
    for c in cases:
        if isinstance(c, dict):
            out.append(c)
    if not out:
        raise ValueError("Fixture file has no valid case objects.")
    return out


def _bars_to_df(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("bars_1m cannot be empty")
    d = pd.DataFrame(rows)
    req = {"ts", "open", "high", "low", "close"}
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise KeyError(f"bars_1m missing required columns: {miss}")
    d = d.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts", kind="mergesort")
    if d.empty:
        raise ValueError("bars_1m has no valid timestamps")
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"])
    if d.empty:
        raise ValueError("bars_1m has no valid OHLC rows")
    d = d.set_index("ts")
    return d[["open", "high", "low", "close"]]


def _iso(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return ""
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def _normalize_hit(x: object) -> Optional[str]:
    s = str(x).strip().lower()
    if s in ("", "none", "nan", "null"):
        return None
    if s not in ("sl", "tp"):
        raise ValueError(f"Unsupported hit value: {x}")
    return s


def _pick_expected(case: Dict[str, object], tie_breaker: str) -> Dict[str, object]:
    exp = case.get("expected")
    if not isinstance(exp, dict):
        raise KeyError("case.expected must be an object")

    if tie_breaker in exp and isinstance(exp[tie_breaker], dict):
        return exp[tie_breaker]
    if "default" in exp and isinstance(exp["default"], dict):
        return exp["default"]
    raise KeyError(f"No expected result for tie_breaker={tie_breaker}")


def _naive_parent_bar_hit(
    side: str,
    low: float,
    high: float,
    stop_price: float,
    take_price: float,
    tie_breaker: str,
) -> Optional[str]:
    side = str(side).lower()
    tie_breaker = str(tie_breaker).lower()
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    if tie_breaker not in ("sl_wins", "tp_wins"):
        raise ValueError("tie_breaker must be sl_wins or tp_wins")

    if side == "long":
        tp_hit = bool(high >= take_price)
        sl_hit = bool(low <= stop_price)
    else:
        tp_hit = bool(low <= take_price)
        sl_hit = bool(high >= stop_price)

    if tp_hit and sl_hit:
        return "tp" if tie_breaker == "tp_wins" else "sl"
    if tp_hit:
        return "tp"
    if sl_hit:
        return "sl"
    return None


def _legacy_sl_first_parent_bar_hit(
    side: str,
    low: float,
    high: float,
    stop_price: float,
    take_price: float,
) -> Optional[str]:
    side = str(side).lower()
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    if side == "long":
        if low <= stop_price:
            return "sl"
        if high >= take_price:
            return "tp"
    else:
        if high >= stop_price:
            return "sl"
        if low <= take_price:
            return "tp"
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay deterministic intrabar fixtures against bt_intrabar.resolve_first_touch_1m."
    )
    p.add_argument(
        "--fixtures",
        default="unit_tests/fixtures/intrabar_replay_fixtures.json",
        help="Path to replay fixture JSON.",
    )
    p.add_argument(
        "--outdir",
        default="results/intrabar_replay_parity",
        help="Output folder (a run_id subdir is created inside).",
    )
    p.add_argument("--run-id", default="", help="Optional run id.")
    p.add_argument(
        "--determinism-repeats",
        type=int,
        default=3,
        help="How many extra resolver calls per check to verify deterministic behavior.",
    )
    return p.parse_args()


def main() -> int:
    a = parse_args()
    fixtures_path = _to_abs(a.fixtures)
    out_root = _to_abs(a.outdir)
    run_id = a.run_id.strip() or _utc_id()
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    cases = _load_fixtures(fixtures_path)
    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    deterministic_failures: List[str] = []

    for case in cases:
        case_id = str(case.get("case_id", "")).strip() or f"case_{len(rows)+1:03d}"
        side = str(case.get("side", "")).strip().lower()
        if side not in ("long", "short"):
            raise ValueError(f"{case_id}: side must be long|short")

        start_ts = pd.to_datetime(case.get("start_ts"), utc=True, errors="raise")
        end_ts = pd.to_datetime(case.get("end_ts"), utc=True, errors="raise")
        stop_price = float(case.get("stop_price"))
        take_price = float(case.get("take_price"))
        bars_raw = case.get("bars_1m")
        if not isinstance(bars_raw, list):
            raise ValueError(f"{case_id}: bars_1m must be a list")
        df_1m = _bars_to_df(bars_raw)  # type: ignore[arg-type]

        tie_breakers = case.get("tie_breakers", ["sl_wins", "tp_wins"])
        if not isinstance(tie_breakers, list) or not tie_breakers:
            tie_breakers = ["sl_wins", "tp_wins"]
        tie_breakers = [str(t).strip() for t in tie_breakers]

        parent_high = float(df_1m["high"].max())
        parent_low = float(df_1m["low"].min())
        ambiguous_parent = bool(
            _naive_parent_bar_hit(side, parent_low, parent_high, stop_price, take_price, "sl_wins")
            != _naive_parent_bar_hit(side, parent_low, parent_high, stop_price, take_price, "tp_wins")
        )

        for tie in tie_breakers:
            tie_l = tie.lower()
            if tie_l not in ("sl_wins", "tp_wins"):
                raise ValueError(f"{case_id}: unsupported tie_breaker={tie}")

            expected = _pick_expected(case, tie_l)
            exp_hit = _normalize_hit(expected.get("hit"))
            exp_ts = pd.to_datetime(expected.get("hit_ts"), utc=True, errors="coerce")
            exp_ts_iso = "" if pd.isna(exp_ts) else exp_ts.isoformat()

            hit, hit_ts = resolve_first_touch_1m(
                side=side,
                df_1m=df_1m,
                start_ts=start_ts,
                end_ts=end_ts,
                stop_price=stop_price,
                take_price=take_price,
                tie_breaker=tie_l,
            )
            hit = _normalize_hit(hit)
            hit_iso = _iso(hit_ts)

            pass_hit = bool(hit == exp_hit)
            pass_ts = bool((hit_iso == exp_ts_iso) if exp_ts_iso else (hit_iso == ""))
            passed = bool(pass_hit and pass_ts)

            # Determinism check: same inputs -> same outputs
            det_ok = True
            repeats = max(0, int(a.determinism_repeats))
            for _ in range(repeats):
                h2, t2 = resolve_first_touch_1m(
                    side=side,
                    df_1m=df_1m,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    stop_price=stop_price,
                    take_price=take_price,
                    tie_breaker=tie_l,
                )
                if _normalize_hit(h2) != hit or _iso(t2) != hit_iso:
                    det_ok = False
                    break

            naive_hit = _naive_parent_bar_hit(
                side=side,
                low=parent_low,
                high=parent_high,
                stop_price=stop_price,
                take_price=take_price,
                tie_breaker=tie_l,
            )
            legacy_sl_first_hit = _legacy_sl_first_parent_bar_hit(
                side=side,
                low=parent_low,
                high=parent_high,
                stop_price=stop_price,
                take_price=take_price,
            )

            row = {
                "case_id": case_id,
                "side": side,
                "tie_breaker": tie_l,
                "start_ts": start_ts.isoformat(),
                "end_ts": end_ts.isoformat(),
                "stop_price": stop_price,
                "take_price": take_price,
                "expected_hit": exp_hit,
                "expected_hit_ts": exp_ts_iso,
                "actual_hit": hit,
                "actual_hit_ts": hit_iso,
                "pass_hit": pass_hit,
                "pass_ts": pass_ts,
                "pass": passed,
                "deterministic": det_ok,
                "parent_bar_high": parent_high,
                "parent_bar_low": parent_low,
                "parent_bar_ambiguous": ambiguous_parent,
                "naive_parent_hit_by_tie": naive_hit,
                "legacy_parent_sl_first_hit": legacy_sl_first_hit,
                "resolver_vs_legacy_diff": bool(hit != legacy_sl_first_hit),
            }
            rows.append(row)

            if not passed:
                failures.append(
                    f"{case_id}/{tie_l}: expected={exp_hit}@{exp_ts_iso or 'None'} actual={hit}@{hit_iso or 'None'}"
                )
            if not det_ok:
                deterministic_failures.append(f"{case_id}/{tie_l}")

    df = pd.DataFrame(rows)
    checks_total = int(len(df))
    checks_pass = int(df["pass"].sum()) if checks_total else 0
    checks_fail = int(checks_total - checks_pass)
    det_fail = int((~df["deterministic"]).sum()) if checks_total else 0
    amb_cases = int(df["parent_bar_ambiguous"].sum()) if checks_total else 0
    resolver_vs_legacy_diff = int(df["resolver_vs_legacy_diff"].sum()) if checks_total else 0

    status = "ok" if checks_fail == 0 and det_fail == 0 else "error"
    summary = {
        "status": status,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures_path": str(fixtures_path),
        "checks_total": checks_total,
        "checks_pass": checks_pass,
        "checks_fail": checks_fail,
        "determinism_failures": det_fail,
        "ambiguous_parent_bar_checks": amb_cases,
        "resolver_vs_legacy_diff_checks": resolver_vs_legacy_diff,
        "failed_checks": failures[:200],
        "determinism_failed_checks": deterministic_failures[:200],
        "outputs": {
            "results_csv": str(outdir / "results.csv"),
            "summary_json": str(outdir / "summary.json"),
            "report_md": str(outdir / "report.md"),
        },
    }

    (outdir / "results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Intrabar Replay Parity Report",
        "",
        f"- run_id: `{run_id}`",
        f"- status: `{status}`",
        f"- fixtures: `{fixtures_path}`",
        f"- total checks: `{checks_total}`",
        f"- passed: `{checks_pass}`",
        f"- failed: `{checks_fail}`",
        f"- determinism failures: `{det_fail}`",
        f"- ambiguous parent-bar checks: `{amb_cases}`",
        f"- resolver vs legacy-sl-first differences: `{resolver_vs_legacy_diff}`",
        "",
        "## Failed Checks",
    ]
    if failures:
        lines.extend([f"- `{x}`" for x in failures[:50]])
    else:
        lines.append("- none")
    lines += ["", "## Notes", "- `resolver_vs_legacy_diff` > 0 is expected for some ambiguous parent bars.", "- Main acceptance is zero failed checks and zero determinism failures."]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[intrabar-parity] status={status}")
    print(f"[intrabar-parity] checks={checks_total} pass={checks_pass} fail={checks_fail} det_fail={det_fail}")
    print(f"[intrabar-parity] summary={outdir / 'summary.json'}")
    print(f"[intrabar-parity] report={outdir / 'report.md'}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
