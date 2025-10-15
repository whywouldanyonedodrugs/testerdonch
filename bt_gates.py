# bt_gates.py
"""
Optional gates/prefilters. Currently minimalist—kept for future experiments.
"""
from __future__ import annotations
import pandas as pd

def allow_all(_row: pd.Series) -> bool:
    return True
