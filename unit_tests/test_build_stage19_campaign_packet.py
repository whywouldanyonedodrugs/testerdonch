import json
import unittest
from pathlib import Path

from tools.build_stage19_campaign_packet import (
    ARCHIVE, STAGE16, funding_canary, stage19_funding_contract, translation_registry,
)
from tools.qlmg_stage16_campaign import build_translation_registry, canonical_sha256


class Stage19CampaignPacketTests(unittest.TestCase):
    def test_all_186_economic_addresses_change_only_with_funding_dependency(self):
        old = build_translation_registry()
        new = translation_registry(stage19_funding_contract())
        self.assertEqual(len(new["cells"]), 186)
        self.assertEqual(
            [cell["cell_id"] for cell in old["cells"]],
            [cell["cell_id"] for cell in new["cells"]],
        )
        self.assertTrue(all(
            left["canonical_economic_address_template"] != right["canonical_economic_address_template"]
            for left, right in zip(old["cells"], new["cells"])
        ))
        for left, right in zip(old["cells"], new["cells"]):
            for key in left:
                if key not in {"cost_funding", "canonical_economic_address_template", "canonical_translation_id"}:
                    self.assertEqual(left[key], right[key])

    def test_packet_is_non_authorizing_and_hash_bound(self):
        packet = json.loads((ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json").read_text())
        payload = {key: value for key, value in packet.items() if key != "packet_payload_canonical_sha256"}
        self.assertFalse(packet["economic_run_authorized"])
        self.assertTrue(packet["external_human_approval_required"])
        self.assertEqual(canonical_sha256(payload), packet["packet_payload_canonical_sha256"])

    def test_stage19_canary_asserts_all_funding_checks(self):
        canary = funding_canary(translation_registry(stage19_funding_contract()))
        self.assertEqual(canary["status"], "pass")
        self.assertFalse(canary["runtime_semantic_discretion"])
        self.assertTrue(canary["favourable_funding_ignored_for_selection"])


if __name__ == "__main__":
    unittest.main()
