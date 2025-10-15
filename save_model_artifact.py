# save_model_artifact.py — run on YOUR LAPTOP (research env)
import json, joblib, numpy as np
from pathlib import Path

# your trained object; works if it’s an sklearn Pipeline, a calibrated Pipeline, or LightGBM’s sklearn API
# e.g., model = pipeline   (contains Imputer/Scaler/etc internally)
# IMPORTANT: set this variable to your trained object:
model = trained_pipeline_or_classifier

# the exact ordered feature names your model expects at predict time
# if your Pipeline exposes feature_names_in_, use that; else define the list you used to train
FEATURES = list(getattr(model, "feature_names_in_", [])) or [
    # << put the exact names you trained on >>
    "rsi_at_entry","adx_at_entry","atr1h_pct_at_entry",
    "donch_dist_pct","vol_mult","rs_pct",
    "eth_macdhist_at_entry","hour_of_day_at_entry","day_of_week_at_entry"
]

ARTIFACT = Path("donch_meta")
ARTIFACT.mkdir(exist_ok=True)

joblib.dump(model, ARTIFACT / "model.joblib")

meta = {
    "name": "Donch Meta LGBM",
    "version": "2025-08-16",
    "features": FEATURES,                 # ordered, required at inference
    "impute_default": 0.0,                # fill for missing features if not handled in Pipeline
    "threshold": 0.58,                    # p* you want live; change later in config if you prefer
    "notes": "Calibrated LightGBM; features must match names/scales used in training."
}
(ARTIFACT / "model_meta.json").write_text(json.dumps(meta, indent=2))
print("✅ Wrote", ARTIFACT.resolve())
