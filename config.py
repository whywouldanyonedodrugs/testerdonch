# config.py
from __future__ import annotations
from pathlib import Path
import os

# --- Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_ROOT / "parquet"        # 5m OHLCV (SYMBOL.parquet)
PARQUET_1M_DIR = PROJECT_ROOT / "parquet_1m"  # optional 1m intrabar
SIGNALS_DIR = PROJECT_ROOT / "signals"
RESULTS_DIR = PROJECT_ROOT / "results"
SYMBOLS_FILE = PROJECT_ROOT / "symbols.txt"
for p in (PARQUET_DIR, PARQUET_1M_DIR, SIGNALS_DIR, RESULTS_DIR): p.mkdir(parents=True, exist_ok=True)

# --- Execution window
START_DATE: str | None = "2022-01-01"
END_DATE:   str | None = "2025-10-15"

# --- Performance / I/O
N_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)
SCOUT_BACKEND: str = "process"     # "thread" | "process"
IO_MEMORY_MAP: bool = True


SCOUT_STREAMING: bool = True          # stream writes instead of returning giant DF
SCOUT_CLEAN_OUTPUT_DIR: bool = True   # remove old symbol=*/ files before run
SCOUT_ROW_GROUP_SIZE: int = 100_000   # parquet row group target (balance seek vs memory)
SCOUT_PARTITION_COLS: list[str] = ["symbol"]  # partition by symbol



# --- Relative strength (weekly)
RS_ENABLED: bool = True
RS_MIN_PERCENTILE: int = 70       # tighten by default (top 5%)
RS_REBALANCE_ANCHOR_WEEKDAY: int = 0
RS_LIQ_MIN_USD_24H: float = 5_00_000.0  # filter illiquid names

# --- Donchian configuration (NEW)
# Basis for the breakout window: "days" (canonical) or "bars" (intraday)
DONCH_BASIS: str = "days"         # "days" | "bars"
DON_N_DAYS: int = 20              # used when DONCH_BASIS="days"  (Turtle S1=20, S2=55)
DON_N_BARS: int = 5760            # used when DONCH_BASIS="bars" (e.g., 20 days * 288 bars/day)
DON_CONFIRM_CLOSE_ABOVE: bool = True  # require close > donch upper on the entry bar

# --- Volume spike gate (NEW)
VOL_SPIKE_ENABLED: bool = True
VOL_SPIKE_MODE: str = "multiple"  # "multiple" | "quantile"
VOL_LOOKBACK_DAYS: int = 30       # rolling window length (days), mapped to bars internally
VOL_MULTIPLE: float = 2.0         # for mode="multiple": vol >= MULTIPLE * rolling_median
VOL_QUANTILE_Q: float = 0.95      # for mode="quantile": vol >= rolling quantile(q)

# --- Pullback + entry rule
PULLBACK_MODEL: str = "retest"    # "retest" | "mean"
PULLBACK_WINDOW_BARS: int = 12
PULLBACK_WINDOW_HOURS: int = 24
RETEST_EPS_PCT: float = 0.001
MEAN_MA_LEN: int = 20
MEAN_BAND_ATR_MULT: float = 0.5
ENTRY_RULE: str = "close_above_break"  # "rebreak_high" | "close_above_break"

# === Indicator timeframes ===
ATR_TIMEFRAME: str | None = "1h"     # e.g., "1h", "4h", "1D"; None => use base (5m) bars
ATR_LEN: int = 14

# --- Regime filter (ETH 4h MACD histogram)
REGIME_FILTER_ENABLED: bool = True
REGIME_ASSET: str = "ETHUSDT"
REGIME_TIMEFRAME: str = "4h"
REGIME_MACD_FAST: int = 12
REGIME_MACD_SLOW: int = 26
REGIME_MACD_SIGNAL: int = 9
REGIME_REQUIRE_BOTH_POSITIVE: bool = False   # macd>signal AND hist>0
REGIME_BLOCK_WHEN_DOWN: bool = False        # or size-down instead
REGIME_SIZE_WHEN_DOWN: float = 0.5

# --- Risk / exits
INITIAL_CAPITAL: float = 1000000.0

RISK_MODE: str = "cash"          # "percent" | "cash"
RISK_PCT: float = 0.01
FIXED_RISK_CASH: float = 10.0       # used when RISK_MODE="cash"
NOTIONAL_CAP_PCT_OF_EQUITY: float = 0.25

MAX_LEVERAGE: float = 10.0
FEE_RATE: float = 0.00055
ATR_LEN: int = 14
SL_ATR_MULT: float = 2.0
TP_ATR_MULT: float = 8.0
TIME_EXIT_HOURS: float | None = 72

# ===== Dynamic exits (MACD regime) =====
DYN_EXITS_ENABLED: bool = True
DYN_MACD_HIST_THRESH: float = 0.0
DYN_TP_MULT_POS: float = 1.15
DYN_SL_MULT_POS: float = 0.90
DYN_TP_MULT_NEG: float = 0.85
DYN_SL_MULT_NEG: float = 1.15

# --- Partial/Trail (optional)
PARTIAL_TP_ENABLED: bool = False
PARTIAL_TP_RATIO: float = 0.5
PARTIAL_TP1_ATR_MULT: float = 5.0
MOVE_SL_TO_BE_ON_TP1: bool = False

TRAIL_AFTER_TP1: bool = False
TRAIL_ATR_MULT: float = 1.0                  # trail gap in ATR (uses ATR at entry)
TRAIL_USE_HIGH_WATERMARK: bool = False        # trail off HH since TP1, else bar-close

# (AVWAP-anchored exits; used when EXIT_BASIS="avwap_atr")
EXIT_BASIS: str = "price_atr"
AVWAP_MODE: str = "static"          # "static" (levels fixed at entry) | "dynamic" (recompute each bar)
AVWAP_ANCHOR: str = "breakout"      # "breakout" (default) | "entry"
AVWAP_SL_MULT: float = 2          # SL = AVWAP - k*ATR_ref
AVWAP_TP_MULT: float = 8.0         # TP = AVWAP + k*ATR_ref
AVWAP_USE_ENTRY_ATR: bool = True    # True: ATR_ref = ATR at entry; False: atr_pre of each bar (dynamic only) # if False, uses rolling ATR_pre each bar for dynamic mode


# --- Throughput guards (scout + exec)
DEDUP_BUSY_WINDOW_MIN: int = 480
SYMBOL_COOLDOWN_MINUTES: int = 480
MAX_TRADES_PER_DAY: int | None = 100

# --- Intrabar resolution
USE_INTRABAR_1M: bool = False          # keep OFF for sweeps; turn ON for short-list
TIE_BREAKER: str = "sl_wins"

# --- Variant guardrails (NEW)
MAX_TRADES_PER_VARIANT: int = 10000000
MIN_EQUITY_FRACTION_BEFORE_ABORT: float = 0.05  # 20% of initial
MIN_ATR_PCT_OF_PRICE: float = 0.0001             # skip micro-stops (ATR < 0.1% of price)

# --- Reporting
SAVE_TRADES_CSV: bool = True
SAVE_EQUITY_CSV: bool = True
AGGREGATE_BY: list[str] = ["symbol","pullback_type","entry_rule","don_break_len","regime_up"]

# === Extra features for meta model (feature TFs/lengths) ===
RSI_LEN: int = 14
ADX_LEN: int = 14
RSI_TIMEFRAME: str = "1h"
ADX_TIMEFRAME: str = "1h"
DON_DIST_IN_ATR: bool = True  # include (close - don_upper)/ATR as a feature

# === Markov ===
MARKOV_TIMEFRAME = "4h"
MARKOV_P_STAY = 0.95
MARKOV_RANDOM_STATE = 0
MARKOV_PROB_EWMA_ALPHA = 0.2

MARKOV4H_PROB_EWMA_ALPHA = 0.2

# ---------------- Labeling mode (research-only) ----------------
LABELING_MODE: bool = False

# --- Throughput guards (scout + exec)
DEDUP_BUSY_WINDOW_MIN: int = 480
SYMBOL_COOLDOWN_MINUTES: int = 480
MAX_TRADES_PER_DAY: int | None = 100

# --- Intrabar (for cleaner labels)
USE_INTRABAR_1M: bool = False          # normal default

# ---------------- Apply overrides when labeling ----------------
if LABELING_MODE:
    # Keep *entry logic* identical (RS/liquidity/volume/regime), but remove execution throttles:
    DEDUP_BUSY_WINDOW_MIN = 0          # keep all clustered signals
    SYMBOL_COOLDOWN_MINUTES = 0        # allow overlapping signals on same symbol
    MAX_TRADES_PER_DAY = None          # no day cap
    MAX_TRADES_PER_VARIANT = 10_000_000
    MIN_EQUITY_FRACTION_BEFORE_ABORT = 0.0
    USE_INTRABAR_1M = False             # better tie-resolution for labels

    # Optional (usually harmless for labeling, makes sizes never block):
    NOTIONAL_CAP_PCT_OF_EQUITY = 1.0
    MAX_LEVERAGE = 1000.0


# --- Meta gating / sizing ---
META_PROB_THRESHOLD: float | None = 0.60
META_SIZING_ENABLED: bool = True
META_SIZING_P0: float = 0.60
META_SIZING_P1: float = 0.95
META_SIZING_MIN: float = 0.50
META_SIZING_MAX: float = 2.00
SIZE_MIN_CAP: float = 0.25
SIZE_MAX_CAP: float = 3.00
REGIME_DOWNSIZE_MULT: float = 0.65

# ===== Week pattern OOS stress =====
WEEK_PATTERN_ENABLED: bool = False
WEEK_PATTERN: str = "10"

# ===== Live de-risking of open positions =====
LIVE_DERISK_ENABLED: bool = True
DERISK_TARGET_MULT: float = 0.65
DERISK_DOWNSHIFT_ONLY: bool = True
DERISK_HYST: float = 0.02
DERISK_MIN_QTY_FRAC: float = 0.10
DERISK_COOLDOWN_BARS: int = 12

META_PRED_PATH = RESULTS_DIR / 'meta_export' / 'oos_predictions_calibrated.parquet'
META_MERGE_ROUND = "5min"
META_MERGE_TOL = "10min"




PORTFOLIO_RISK_CAP_PCT: float = 0.1       # e.g., total open risk ≤ 5% of equity
GROSS_EXPOSURE_CAP_MULT: float = 3.0       # sum(|notional|) ≤ 3 × equity

ON_CAP_BREACH: str = "scale"               # "scale" (downsize new trade) or "skip"

