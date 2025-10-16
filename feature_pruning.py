# feature_pruning.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", default="results/meta_export/perm_importance_oos.csv")
    ap.add_argument("--shap", default="results/meta_export/shap_importance_oos.csv")
    ap.add_argument("--out", default="results/meta_export/drop_list.txt")
    ap.add_argument("--bottom-k", type=int, default=30)
    args = ap.parse_args()

    perm = pd.read_csv(args.perm)   # columns: feature, mean_drop, ...
    shap = pd.read_csv(args.shap)   # columns: feature, mean_abs_shap, ...
    perm = perm.rename(columns=str.lower); shap = shap.rename(columns=str.lower)
    perm_r = perm.assign(rank_perm = perm["mean_drop"].rank(ascending=True))
    shap_r = shap.assign(rank_shap = shap["mean_abs_shap"].rank(ascending=True))
    merged = perm_r.merge(shap_r, on="feature", how="outer").fillna(1e9)
    merged["score_low"] = merged[["rank_perm","rank_shap"]].min(axis=1)
    drops = merged.sort_values(["score_low","rank_perm","rank_shap"]).head(args.bottom_k)["feature"].tolist()
    Path(args.out).write_text("\n".join(drops) + "\n")
    print(f"[prune] proposed {len(drops)} drops → {args.out}")

if __name__ == "__main__":
    main()
