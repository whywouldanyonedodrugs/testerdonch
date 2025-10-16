# calibration_helpers.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

EPS = 1e-15

def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1 - EPS)

def logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(p)
    return np.log(p / (1 - p))

def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    p = _clip01(p)
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((p - y) ** 2))

def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width bins in [0,1]."""
    p = _clip01(p)
    y = np.asarray(y_true, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        I = (idx == b)
        if not np.any(I):
            continue
        conf = float(np.mean(p[I]))
        acc  = float(np.mean(y[I]))
        w    = float(np.mean(I))
        ece += w * abs(acc - conf)
    return float(ece)

@dataclass
class CalibResult:
    kind: str
    p_cal: np.ndarray
    ok: bool
    metrics: dict
    error: Optional[str] = None

def temperature_scale(y_true: np.ndarray, p_raw: np.ndarray) -> CalibResult:
    """
    Temperature scaling: p_cal = sigmoid(logit(p_raw)/T).
    Fit T by minimizing NLL on validation via 1D Newton steps on w=1/T.
    """
    try:
        y = np.asarray(y_true, dtype=float)
        z = logit(p_raw)  # logits
        # Optimize w = 1/T (unconstrained); use Newton updates on NLL
        w = 1.0
        for _ in range(50):
            s = inv_logit(w * z)
            # gradient and hessian wrt w
            g = np.sum((s - y) * z)
            h = np.sum(s * (1 - s) * z * z) + 1e-12
            step = g / h
            w_new = w - step
            if np.isfinite(w_new) and abs(step) < 1e-6:
                w = w_new
                break
            if np.isfinite(w_new):
                w = w_new
        T = 1.0 / max(w, 1e-12)
        p_cal = inv_logit(z / T)

        # metrics
        ece_raw  = ece_score(y, _clip01(p_raw))
        ece_cal  = ece_score(y, _clip01(p_cal))
        brier_raw = brier_score(y, _clip01(p_raw))
        brier_cal = brier_score(y, _clip01(p_cal))
        std_ratio = float(np.std(p_cal) / (np.std(p_raw) + 1e-12))
        unique_ratio = float(len(np.unique(np.round(p_cal, 10))) / (len(np.unique(np.round(p_raw, 10))) + 1e-12))
        return CalibResult(
            kind="temperature",
            p_cal=p_cal, ok=True,
            metrics=dict(
                T=T, ece_raw=ece_raw, ece_cal=ece_cal, ece_gain=(ece_raw - ece_cal),
                brier_raw=brier_raw, brier_cal=brier_cal, brier_gain=(brier_raw - brier_cal),
                std_ratio=std_ratio, unique_ratio=unique_ratio
            )
        )
    except Exception as e:
        return CalibResult(kind="temperature", p_cal=p_raw, ok=False, metrics={}, error=str(e))

def platt_scale(y_true: np.ndarray, p_raw: np.ndarray) -> CalibResult:
    """Platt scaling via 1D logistic on logits (sigmoid(a*z+b))."""
    try:
        import numpy as np
        from numpy.linalg import lstsq
        y = np.asarray(y_true, dtype=float)
        z = logit(p_raw)
        X = np.c_[z, np.ones_like(z)]
        # Solve least squares on logits space for simplicity
        # (For strict NLL optimum, use sklearn LogisticRegression; this is robust and dependency-light.)
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        # Use a few IRLS iterations to approach NLL optimum
        a, b = 1.0, 0.0
        for _ in range(20):
            s = sigmoid(a*z + b)
            W = s*(1-s) + 1e-12
            Z = (a*z + b) + (y - s)/W
            Xw = X * np.sqrt(W[:,None])
            Zw = Z * np.sqrt(W)
            coef, *_ = lstsq(Xw, Zw, rcond=None)
            a, b = float(coef[0]), float(coef[1])
        p_cal = sigmoid(a*z + b)
        # metrics
        ece_raw  = ece_score(y, _clip01(p_raw))
        ece_cal  = ece_score(y, _clip01(p_cal))
        brier_raw = brier_score(y, _clip01(p_raw))
        brier_cal = brier_score(y, _clip01(p_cal))
        std_ratio = float(np.std(p_cal) / (np.std(p_raw) + 1e-12))
        unique_ratio = float(len(np.unique(np.round(p_cal, 10))) / (len(np.unique(np.round(p_raw, 10))) + 1e-12))
        return CalibResult(kind="platt", p_cal=p_cal, ok=True, metrics=dict(
            ece_raw=ece_raw, ece_cal=ece_cal, ece_gain=(ece_raw - ece_cal),
            brier_raw=brier_raw, brier_cal=brier_cal, brier_gain=(brier_raw - brier_cal),
            std_ratio=std_ratio, unique_ratio=unique_ratio
        ))
    except Exception as e:
        return CalibResult(kind="platt", p_cal=p_raw, ok=False, metrics={}, error=str(e))

def isotonic_scale(y_true: np.ndarray, p_raw: np.ndarray) -> CalibResult:
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1-EPS)
        p_cal = ir.fit_transform(p_raw, y_true.astype(float))
        # metrics
        y = np.asarray(y_true, dtype=float)
        ece_raw  = ece_score(y, _clip01(p_raw))
        ece_cal  = ece_score(y, _clip01(p_cal))
        brier_raw = brier_score(y, _clip01(p_raw))
        brier_cal = brier_score(y, _clip01(p_cal))
        std_ratio = float(np.std(p_cal) / (np.std(p_raw) + 1e-12))
        unique_ratio = float(len(np.unique(np.round(p_cal, 10))) / (len(np.unique(np.round(p_raw, 10))) + 1e-12))
        return CalibResult(kind="isotonic", p_cal=p_cal, ok=True, metrics=dict(
            ece_raw=ece_raw, ece_cal=ece_cal, ece_gain=(ece_raw - ece_cal),
            brier_raw=brier_raw, brier_cal=brier_cal, brier_gain=(brier_raw - brier_cal),
            std_ratio=std_ratio, unique_ratio=unique_ratio
        ))
    except Exception as e:
        return CalibResult(kind="isotonic", p_cal=p_raw, ok=False, metrics={}, error=str(e))
