from __future__ import annotations

import unittest

import pandas as pd

from tools.build_kraken_c01_event_contract import (
    FEATURE_HASH, PRIMARY_MODEL, ROBUSTNESS_MODEL, apply_cohort, extract_onsets, model_agreement,
)


def row(candidate_id: str, ts: str, model: str = PRIMARY_MODEL, sign: str = "positive", path: str = "smooth") -> dict:
    return {
        "candidate_id": candidate_id, "symbol": "PF_AAVEUSD", "venue": "Kraken",
        "decision_ts": ts, "shock_window_start": pd.Timestamp(ts) - pd.Timedelta(hours=6),
        "shock_window_end": ts, "residual_model_version": model, "sign": sign, "path_state": path,
        "feature_version": "c01_residual_path_features_v1_20260717", "reference_panel_hash": "ref",
        "canonical_episode_id": "episode_a", "canonical_episode_input_start": pd.Timestamp(ts) - pd.Timedelta(hours=6),
        "canonical_episode_input_end": pd.Timestamp(ts) + pd.Timedelta(hours=24),
        "residual_shock_6h": 1.0, "residual_scale_6h": 0.2, "residual_shock_z_6h": 5.0,
        "largest_bar_share": 0.2, "path_efficiency": 0.6,
    }


class C01EventContractTests(unittest.TestCase):
    def test_onset_requires_full_72_inactive_bars_and_does_not_choose_peak(self) -> None:
        tape = pd.DataFrame([
            row("first", "2023-01-01T06:00:00Z"),
            row("later_peak", "2023-01-01T06:05:00Z"),
            row("after_reset", "2023-01-01T12:10:00Z"),
        ])
        out = extract_onsets(tape)
        self.assertEqual(out["candidate_id"].tolist(), ["first", "after_reset"])
        self.assertGreaterEqual(out.iloc[1]["reset_inactive_bar_count"], 72)

    def test_cohort_identity_is_deterministic_and_contains_no_outcome(self) -> None:
        onset = extract_onsets(pd.DataFrame([row("first", "2023-02-01T06:00:00Z")]))
        cohort = pd.DataFrame([{
            "utc_day": pd.Timestamp("2023-02-01", tz="UTC"), "symbol": "PF_AAVEUSD",
            "top_100_eligible": True, "rank": 1, "valid_prior_days": 30,
        }])
        left, _ = apply_cohort(onset, cohort, "cohort_hash")
        right, _ = apply_cohort(onset.sample(frac=1, random_state=7), cohort, "cohort_hash")
        self.assertEqual(left["event_id"].tolist(), right["event_id"].tolist())
        self.assertFalse(any(token in name.lower() for name in left.columns for token in ("forward_return", "pnl", "mae", "mfe")))

    def test_model_agreement_does_not_select_model(self) -> None:
        events = pd.DataFrame([
            {**row("p", "2023-01-01T06:00:00Z"), "event_id": "p"},
            {**row("r", "2023-01-01T06:05:00Z", ROBUSTNESS_MODEL), "event_id": "r"},
        ])
        events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True)
        report = model_agreement(events).set_index("metric")["count"]
        self.assertEqual(report["primary_events"], 1)
        self.assertEqual(report["robustness_events"], 1)
        self.assertEqual(report["same_sign_onset_within_30m"], 1)
        self.assertEqual(report["exact_timestamp_and_sign"], 0)

    def test_feature_hash_is_frozen(self) -> None:
        self.assertEqual(FEATURE_HASH, "c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb")


if __name__ == "__main__":
    unittest.main()
