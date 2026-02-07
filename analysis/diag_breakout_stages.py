# diag_breakout_stages.py
from __future__ import annotations
import random, pandas as pd, numpy as np
import config as cfg
from scout import detect_signals_for_symbol, build_weekly_rs
from shared_utils import get_symbols_from_file

def main(n=25):
    random.seed(0)
    syms = get_symbols_from_file()
    syms = random.sample(syms, min(n, len(syms)))
    rs = build_weekly_rs(syms) if cfg.RS_ENABLED else None

    total = 0
    nonempty = 0
    for s in syms:
        df = detect_signals_for_symbol(s, rs)
        total += 1
        if not df.empty:
            nonempty += 1
    print(f"Symbols with >=1 signal: {nonempty}/{total}")

if __name__ == "__main__":
    main(25)
