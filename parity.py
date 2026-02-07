from pathlib import Path
import pandas as pd
import config as cfg

live = pd.read_csv(cfg.RESULTS_DIR / "livetrading.csv")
# adjust column name if needed; in your export it's usually "Market" or "symbol"
col = "Market" if "Market" in live.columns else "symbol"
syms = sorted(set(live[col].astype(str)))

out_path = cfg.PROJECT_ROOT / "symbols_parity.txt"
out_path.write_text("\n".join(syms) + "\n", encoding="utf-8")
print(f"Wrote {len(syms)} symbols to {out_path}")
