# winprob_loader.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("winprob")


def _read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip01(x: float, eps: float) -> float:
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0 or eps >= 0.5:
        eps = 1e-12
    return float(np.clip(float(x), eps, 1.0 - eps))


def _norm_key(k: str) -> str:
    s = str(k).strip().lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[^\w]+", "_", s).strip("_")
    return s


def _make_keyset(k: str) -> List[str]:
    nk = _norm_key(k)
    out = [k, k.lower(), nk, nk.replace("_", "")]
    out.append(str(k).replace("_", ""))
    out.append(str(k).replace("_", "").lower())
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


_ALIAS = {
    "atr1h": "atr_1h",
    "atrpct": "atr_pct",
    "donbreaklen": "don_break_len",
    "donbreaklevel": "don_break_level",
    "dondistatr": "don_dist_atr",
    "rspct": "rs_pct",
    "rsi1h": "rsi_1h",
    "adx1h": "adx_1h",
    "volmult": "vol_mult",
    "volproblow1d": "vol_prob_low_1d",
    "regimecode1d": "regime_code_1d",
    "markovstate4h": "markov_state_4h",
    "markovprobup4h": "markov_prob_up_4h",
    "markovstateup4h": "markov_state_up_4h",
    "oilevel": "oi_level",
    "oinotionalest": "oi_notional_est",
    "oipct1h": "oi_pct_1h",
    "oipct4h": "oi_pct_4h",
    "oipct1d": "oi_pct_1d",
    "oiz7d": "oi_z_7d",
    "oichgnormvol1h": "oi_chg_norm_vol_1h",
    "oipricediv1h": "oi_price_div_1h",
    "fundingrate": "funding_rate",
    "fundingabs": "funding_abs",
    "fundingz7d": "funding_z_7d",
    "fundingrollsum3d": "funding_rollsum_3d",
    "fundingoidiv": "funding_oi_div",
    "fundingoid": "funding_oi_div",
    "hoursin": "hour_sin",
    "hourcos": "hour_cos",
    "dayofweek": "dow",
    "weekday": "dow",
}


def _alias_variants(col: str) -> List[str]:
    c = _norm_key(col)
    out = [col]
    c2 = c.replace("_", "")
    if c in _ALIAS:
        out.append(_ALIAS[c])
    if c2 in _ALIAS:
        out.append(_ALIAS[c2])
    for k, v in _ALIAS.items():
        if _norm_key(v) == c:
            out.append(k)
            out.append(k.replace("_", ""))
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


class WinProbScorer:
    """
    Deployment scorer for the meta-model.

    Expects in artifacts_dir:
      - model.joblib (sklearn Pipeline with predict_proba)
      - feature_manifest.json (features.numeric_cols / features.cat_cols)
      - calibration.json (chosen_method + params; isotonic artifact optional)
    """

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.model_dir = Path(artifacts_dir).resolve()
        self.model = None

        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.raw_cols: List[str] = []

        self._cal_method: str = "none"
        self._cal_eps: float = 1e-12
        self._iso = None
        self._sig_a: Optional[float] = None
        self._sig_b: Optional[float] = None

        self._load()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and bool(self.raw_cols)

    def _load(self) -> None:
        # model pipeline
        model_path = self.model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model.joblib in {self.model_dir}")
        self.model = joblib.load(model_path)

        # feature manifest (authoritative raw schema)
        man_path = self.model_dir / "feature_manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(f"Missing feature_manifest.json in {self.model_dir}")
        man = _read_json(man_path)
        feats = man.get("features") or {}
        self.numeric_cols = list(feats.get("numeric_cols") or [])
        self.cat_cols = list(feats.get("cat_cols") or [])
        self.raw_cols = self.numeric_cols + self.cat_cols
        if not self.raw_cols:
            raise RuntimeError("feature_manifest.json has empty features.numeric_cols/cat_cols")

        # calibration
        cal_path = self.model_dir / "calibration.json"
        if cal_path.exists():
            cal = _read_json(cal_path)
            self._cal_method = str(cal.get("chosen_method") or "none").strip().lower()
            params = cal.get("params") or {}
            self._cal_eps = float(params.get("eps", 1e-12))

            if self._cal_method == "isotonic":
                art = cal.get("artifact") or "isotonic.joblib"
                iso_path = self.model_dir / str(art)
                if not iso_path.exists():
                    raise FileNotFoundError(f"chosen_method=isotonic but missing {iso_path}")
                self._iso = joblib.load(iso_path)

            elif self._cal_method == "sigmoid":
                a = params.get("a", None)
                b = params.get("b", None)
                if a is None or b is None:
                    raise ValueError("chosen_method=sigmoid but params a/b missing")
                self._sig_a = float(a)
                self._sig_b = float(b)
            else:
                self._cal_method = "none"

        LOG.info(
            "[WINPROB] loaded model_dir=%s raw_cols=%d (num=%d, cat=%d) cal=%s",
            str(self.model_dir),
            len(self.raw_cols),
            len(self.numeric_cols),
            len(self.cat_cols),
            self._cal_method,
        )

    def _get(self, row: Dict[str, Any], col: str) -> Any:
        if "__norm" not in row:
            norm: Dict[str, Any] = {}
            for k, v in row.items():
                nk = _norm_key(k)
                norm[nk] = v
                norm[nk.replace("_", "")] = v
            row["__norm"] = norm
        norm = row["__norm"]

        for c in _alias_variants(col):
            for key in _make_keyset(c):
                if key in row:
                    return row[key]
                nk = _norm_key(key)
                if nk in norm:
                    return norm[nk]
                nk2 = nk.replace("_", "")
                if nk2 in norm:
                    return norm[nk2]
        return np.nan

    def _derive(self, out: Dict[str, Any]) -> None:
        cols_norm = {_norm_key(c): c for c in self.raw_cols}

        if ("atr_pct" in cols_norm) or ("atrpct" in cols_norm):
            name = cols_norm.get("atr_pct") or cols_norm.get("atrpct")
            v = out.get(name, np.nan)
            if not np.isfinite(pd.to_numeric(v, errors="coerce")):
                entry = pd.to_numeric(out.get("entry", np.nan), errors="coerce")
                atr1h = pd.to_numeric(out.get("atr_1h", out.get("atr1h", np.nan)), errors="coerce")
                atr = pd.to_numeric(out.get("atr", np.nan), errors="coerce")
                if np.isfinite(entry) and entry > 0:
                    a = atr1h if np.isfinite(atr1h) else atr
                    if np.isfinite(a):
                        out[name] = float(a / entry)

        if ("don_dist_atr" in cols_norm) or ("dondistatr" in cols_norm):
            name = cols_norm.get("don_dist_atr") or cols_norm.get("dondistatr")
            v = out.get(name, np.nan)
            if not np.isfinite(pd.to_numeric(v, errors="coerce")):
                entry = pd.to_numeric(out.get("entry", np.nan), errors="coerce")
                don = pd.to_numeric(out.get("don_break_level", out.get("donbreaklevel", np.nan)), errors="coerce")
                atr1h = pd.to_numeric(out.get("atr_1h", out.get("atr1h", np.nan)), errors="coerce")
                atr = pd.to_numeric(out.get("atr", np.nan), errors="coerce")
                a = atr1h if np.isfinite(atr1h) else atr
                if np.isfinite(entry) and np.isfinite(don) and np.isfinite(a) and a > 0:
                    out[name] = float((entry - don) / a)

    def _build_X_raw(self, row_in: Dict[str, Any]) -> pd.DataFrame:
        row = dict(row_in)
        out: Dict[str, Any] = {}

        for c in self.numeric_cols:
            v = self._get(row, c)
            out[c] = pd.to_numeric(v, errors="coerce")

        for c in self.cat_cols:
            v = self._get(row, c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[c] = np.nan
            else:
                out[c] = str(v)

        self._derive(out)
        return pd.DataFrame([out], columns=self.raw_cols)

    def _calibrate(self, p_raw: float) -> float:
        eps = self._cal_eps
        p = _clip01(p_raw, eps)

        if self._cal_method == "isotonic":
            if self._iso is None:
                raise RuntimeError("calibration is isotonic but isotonic model not loaded")
            p = float(self._iso.predict(np.array([p], dtype=float))[0])
            return _clip01(p, eps)

        if self._cal_method == "sigmoid":
            if self._sig_a is None or self._sig_b is None:
                raise RuntimeError("calibration is sigmoid but a/b not loaded")
            z = np.log(p / (1.0 - p))
            p2 = 1.0 / (1.0 + np.exp(-(self._sig_a * z + self._sig_b)))
            return _clip01(float(p2), eps)

        return _clip01(p, eps)

    def score(self, row: Dict[str, Any]) -> float:
        if not self.is_loaded:
            return float("nan")
        X = self._build_X_raw(row)
        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        return float(np.clip(self._calibrate(p_raw), 0.0, 1.0))
