import json
from pathlib import Path
import tempfile
import unittest

from tools.sealed_slice_guard import (
    assert_sealed_slice_access_allowed,
    register_default_sealed_slice,
)


class SealedSliceGuardTests(unittest.TestCase):
    def test_candidate_selection_overlap_requires_contract(self):
        with tempfile.TemporaryDirectory() as td:
            reg = Path(td) / "registry.json"
            register_default_sealed_slice(reg)
            with self.assertRaises(RuntimeError):
                assert_sealed_slice_access_allowed(
                    start="2026-03-06",
                    end="2026-03-10",
                    purpose="candidate_selection",
                    registry_path=reg,
                )

    def test_non_overlapping_window_passes_without_contract(self):
        with tempfile.TemporaryDirectory() as td:
            reg = Path(td) / "registry.json"
            register_default_sealed_slice(reg)
            assert_sealed_slice_access_allowed(
                start="2026-02-01",
                end="2026-03-05",
                purpose="candidate_selection",
                registry_path=reg,
            )

    def test_frozen_contract_must_declare_slice(self):
        with tempfile.TemporaryDirectory() as td:
            reg = Path(td) / "registry.json"
            register_default_sealed_slice(reg)
            contract = Path(td) / "contract.json"
            contract.write_text(json.dumps({"contract_frozen": True, "sealed_slice_access": []}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                assert_sealed_slice_access_allowed(
                    start="2026-03-06",
                    end="2026-03-10",
                    purpose="candidate_selection",
                    contract_path=contract,
                    registry_path=reg,
                )

    def test_frozen_contract_with_declared_slice_passes(self):
        with tempfile.TemporaryDirectory() as td:
            reg = Path(td) / "registry.json"
            register_default_sealed_slice(reg)
            contract = Path(td) / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "contract_frozen": True,
                        "sealed_slice_access": [
                            {
                                "slice_id": "sealed_2026_03_06_to_2026_06_18",
                                "allowed_purposes": ["candidate_selection"],
                                "frozen_before_access": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            assert_sealed_slice_access_allowed(
                start="2026-03-06",
                end="2026-03-10",
                purpose="candidate_selection",
                contract_path=contract,
                registry_path=reg,
            )


if __name__ == "__main__":
    unittest.main()
