# run_reporting.py
import re
from pathlib import Path
import pandas as pd
import reporting
import config as cfg

def main():
    results = Path(cfg.RESULTS_DIR)
    results.mkdir(parents=True, exist_ok=True)

    # Find the latest closed_trades_raw_*.csv
    raws = sorted(results.glob("closed_trades_raw_*.csv"), key=lambda p: p.stat().st_mtime)
    if not raws:
        print("No raw closed trade files found in results/. Run manager.py once to produce them.")
        return

    latest_raw = raws[-1]
    m = re.search(r"closed_trades_raw_(\d{8}_\d{6})\.csv", latest_raw.name)
    if not m:
        print(f"Could not parse timestamp from {latest_raw.name}")
        return
    ts = m.group(1)

    eq_path = results / f"equity_curve_{ts}.csv"
    if not eq_path.exists():
        print(f"Missing equity curve file: {eq_path}")
        return

    print(f"[using] {latest_raw.name}")
    print(f"[using] {eq_path.name}")

    closed = pd.read_csv(latest_raw)
    eq_df  = pd.read_csv(eq_path)

    # Build aggregated trades (one row per trade)
    agg = reporting.create_aggregated_report(closed.copy(), ts)

    # Save aggregated
    agg_path = results / f"trades_aggregated_{ts}.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[ok] Aggregated → {agg_path.name}")

    # Make summary text
    summary = reporting.generate_summary_report(
        closed_df=closed.copy(),
        eq_df=eq_df.copy(),
        agg_trades_df=agg,
        filter_log={},          # you can pass backtester.get_filter_log() if you saved it
        candidate_log={},       # same for candidate_log
        ts_str=ts
    )
    sum_path = results / f"summary_report_{ts}.txt"
    sum_path.write_text(summary, encoding="utf-8")
    print(f"[ok] Summary    → {sum_path.name}\n")

    # Print a short tail so you see it worked
    print("==== SUMMARY (tail) ====\n")
    print("\n".join(summary.splitlines()[-30:]))

if __name__ == "__main__":
    main()
