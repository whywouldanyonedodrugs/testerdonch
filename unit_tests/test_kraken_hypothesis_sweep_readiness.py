import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from tools import run_kraken_hypothesis_sweep_readiness as mod


def make_minimal_xlsx(path: Path) -> None:
    files = {
        "[Content_Types].xml": """<?xml version='1.0' encoding='UTF-8'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
<Default Extension='xml' ContentType='application/xml'/>
<Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/>
<Override PartName='/xl/worksheets/sheet1.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>
<Override PartName='/xl/worksheets/sheet2.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>
<Override PartName='/xl/sharedStrings.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml'/>
<Override PartName='/xl/styles.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml'/>
</Types>""",
        "_rels/.rels": """<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/></Relationships>""",
        "xl/_rels/workbook.xml.rels": """<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet1.xml'/><Relationship Id='rId2' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet2.xml'/></Relationships>""",
        "xl/workbook.xml": """<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets><sheet name='Hypothesis Library' sheetId='1' r:id='rId1'/><sheet name='Priority View' sheetId='2' r:id='rId2'/></sheets></workbook>""",
        "xl/sharedStrings.xml": """<sst xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><si><t>Hypothesis ID</t></si><si><t>Canonical Family</t></si><si><t>Short Name</t></si><si><t>Mechanism / Principle</t></si><si><t>Entry Sketch</t></si><si><t>Data Tier</t></si><si><t>H01</t></si><si><t>Liquid continuation</t></si><si><t>A1 special &amp; chars</t></si><si><t>trend continuation</t></si><si><t>close confirmed entry</t></si><si><t>Tier 1.</t></si></sst>""",
        "xl/styles.xml": """<styleSheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><cellXfs count='2'><xf numFmtId='0'/><xf numFmtId='14'/></cellXfs></styleSheet>""",
        "xl/worksheets/sheet1.xml": """<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData>
<row r='1'><c r='A1' t='s'><v>0</v></c><c r='B1' t='s'><v>1</v></c><c r='C1' t='s'><v>2</v></c><c r='D1' t='s'><v>3</v></c><c r='E1' t='s'><v>4</v></c><c r='F1' t='s'><v>5</v></c><c r='G1'><v>44562</v></c></row>
<row r='2'><c r='A2' t='s'><v>6</v></c><c r='B2' t='s'><v>7</v></c><c r='C2' t='s'><v>8</v></c><c r='D2' t='s'><v>9</v></c><c r='E2' t='s'><v>10</v></c><c r='F2' t='s'><v>11</v></c><c r='G2' s='1'><v>44562</v></c><c r='H2'><f>1+1</f><v>2</v></c><c r='I2' t='inlineStr'><is><t>inline text</t></is></c></row>
</sheetData></worksheet>""",
        "xl/worksheets/sheet2.xml": """<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData><row r='1'><c r='A1' t='s'><v>0</v></c><c r='B1' t='s'><v>5</v></c></row><row r='2'><c r='A2' t='s'><v>6</v></c><c r='B2' t='s'><v>11</v></c></row></sheetData></worksheet>""",
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)


class KrakenHypothesisReadinessTest(unittest.TestCase):
    def test_stdlib_xlsx_reader_preserves_sheets_types_formula_and_special_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.xlsx"
            make_minimal_xlsx(path)
            parsed = mod.read_xlsx_stdlib(path)
            self.assertIn("Hypothesis Library", parsed["sheets"])
            self.assertIn("Priority View", parsed["sheets"])
            rows = parsed["sheets"]["Hypothesis Library"]
            self.assertEqual(rows[1][0], "H01")
            self.assertIn("&", rows[1][2])
            self.assertEqual(rows[1][7], 2)
            self.assertEqual(rows[1][8], "inline text")
            formula_cells = [r for r in parsed["cell_meta"] if r["formula"]]
            self.assertEqual(len(formula_cells), 1)
            self.assertRegex(str(rows[1][6]), r"2022")

    def test_duplicate_and_missing_hypothesis_detection_data_shape(self):
        df = pd.DataFrame({"hypothesis_id": ["H01", "H01", ""], "alpha_mechanism": ["a", "b", "c"], "entry_sketch": ["x", "y", "z"], "data_tier": ["Tier 1", "Tier 1", "Tier 2"]})
        checks = []
        for hid, group in df.groupby("hypothesis_id", dropna=False):
            checks.append({"hypothesis_id": hid, "rows": len(group), "duplicate": len(group) > 1, "missing_id": str(hid).strip() == ""})
        self.assertTrue(any(r["duplicate"] for r in checks))
        self.assertTrue(any(r["missing_id"] for r in checks))

    def test_semantic_sanity_blocks_touch_and_microstructure(self):
        row = {"data_tier": "Tier 1", "entry_sketch": "touch breakout", "alpha_mechanism": "orderbook depth liquidation"}
        issues = mod.semantic_checks_for(row)
        self.assertIn("touch_fill_entry_not_allowed_tier1", issues)
        self.assertIn("microstructure_hypothesis_not_tier1_rankable", issues)

    def test_classify_missing_does_not_block_recent_1m_for_readiness(self):
        self.assertEqual(mod.classify_missing("candles_recent", {"path_exists": False, "parquet_files": 0}), "not_needed_for_readiness")
        self.assertEqual(mod.classify_missing("historical_trade_candles_5m", {"path_exists": False, "parquet_files": 0}), "needed_for_full_sweep")

    def test_pilot_forbidden_labels_are_blocked(self):
        df = pd.DataFrame({"pipeline_label": ["pipeline_passed", "validated lead"]})
        bad = mod.validate_no_forbidden_pilot_labels(df)
        self.assertTrue(bad)

    def test_tmux_wrapper_launch_gate(self):
        text = Path("tools/run_kraken_hypothesis_sweep_readiness_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("refusing to launch", text)
        self.assertIn("run_kraken_hypothesis_sweep_readiness.py", text)

    def test_event_fixture_math_long_short(self):
        gross, net = mod.event_r("long", 100, 98, 104, 0.004, 0.002, 0)
        self.assertAlmostEqual(gross, 2.0)
        self.assertLess(net, gross)
        gross_s, _ = mod.event_r("short", 100, 102, 96, 0, 0, 0)
        self.assertAlmostEqual(gross_s, 2.0)


if __name__ == "__main__":
    unittest.main()
