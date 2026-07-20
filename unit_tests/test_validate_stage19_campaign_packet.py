import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.qlmg_stage16_campaign import canonical_sha256
from tools.validate_stage19_campaign_packet import sha256_file, validate


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2"


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def rebind(root, dependency):
    manifest_path = root / "CAMPAIGN_MANIFEST.json"
    packet_path = root / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"
    manifest = json.loads(manifest_path.read_text())
    packet = json.loads(packet_path.read_text())
    digest = sha256_file(root / dependency)
    manifest["dependency_file_sha256"][dependency] = digest
    packet["dependency_file_sha256"][dependency] = digest
    write_json(manifest_path, manifest)
    packet["campaign_manifest_file_sha256"] = sha256_file(manifest_path)
    payload = {key: value for key, value in packet.items() if key != "packet_payload_canonical_sha256"}
    packet["packet_payload_canonical_sha256"] = canonical_sha256(payload)
    write_json(packet_path, packet)


class Stage19ValidatorTests(unittest.TestCase):
    def mutated_root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "packet"
        shutil.copytree(ARCHIVE, root)
        return temporary, root

    def test_rejects_rebound_dual_alignment_mutation(self):
        temporary, root = self.mutated_root()
        with temporary:
            path = root / "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json"
            value = json.loads(path.read_text())
            value["selection_funding_bps"] = "signed_alignment_start_bps"
            write_json(path, value)
            rebind(root, path.name)
            funding_path = root / "FUNDING_COST_AND_COVERAGE_CONTRACT.json"
            funding = json.loads(funding_path.read_text())
            funding["dual_alignment_contract_sha256"] = sha256_file(path)
            write_json(funding_path, funding)
            rebind(root, funding_path.name)
            with self.assertRaisesRegex(RuntimeError, "dual-alignment semantic"):
                validate(root)

    def test_rejects_rebound_gap_contract_mutation(self):
        temporary, root = self.mutated_root()
        with temporary:
            path = root / "FUNDING_GAP_ALLOWANCE_CONTRACT.json"
            value = json.loads(path.read_text())
            value["minimum_symbol_observations"] = 1
            write_json(path, value)
            rebind(root, path.name)
            with self.assertRaisesRegex(RuntimeError, "gap allowance semantic"):
                validate(root)

    def test_rejects_rebound_funding_contract_mutation(self):
        temporary, root = self.mutated_root()
        with temporary:
            path = root / "FUNDING_COST_AND_COVERAGE_CONTRACT.json"
            value = json.loads(path.read_text())
            value["missing_hour_cost"] = "0"
            write_json(path, value)
            rebind(root, path.name)
            with self.assertRaisesRegex(RuntimeError, "semantically inconsistent"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
