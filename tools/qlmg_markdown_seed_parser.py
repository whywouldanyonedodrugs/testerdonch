#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")
DECLARED_COUNT_RE = re.compile(
    r"(?i)\b(\d{1,4})\s+(?:high/medium-confidence\s+)?(?:row|rows|event|events|asset|assets|catalyst|catalysts)\b"
)


@dataclass(frozen=True)
class ParsedTable:
    section: str
    section_slug: str
    header: list[str]
    rows: list[dict[str, Any]]
    start_line: int
    end_line: int
    raw_text: str


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "section"


def _split_pipe_row(line: str) -> list[str]:
    raw = line.rstrip("\n")
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    cells: list[str] = []
    cur: list[str] = []
    escaped = False
    for ch in raw:
        if escaped:
            cur.append(ch)
            escaped = False
            continue
        if ch == "\\":
            cur.append(ch)
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def _is_separator(cells: list[str]) -> bool:
    if not cells:
        return False
    ok = 0
    for c in cells:
        t = c.strip().replace(":", "").replace("-", "")
        if t == "":
            ok += 1
    return ok == len(cells)


def _normalize_cells(cells: list[str], n: int) -> tuple[list[str], str]:
    warning = ""
    if len(cells) == n:
        return cells, warning
    if len(cells) > n:
        warning = f"extra_cells_joined_{len(cells)}_to_{n}"
        return cells[: n - 1] + [" | ".join(cells[n - 1 :])], warning
    warning = f"missing_cells_padded_{len(cells)}_to_{n}"
    return cells + [""] * (n - len(cells)), warning


def parse_markdown_tables(path: str | Path) -> tuple[list[ParsedTable], list[dict[str, Any]]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    current_section = "document"
    tables: list[ParsedTable] = []
    unparsed: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = HEADING_RE.match(lines[i])
        if m:
            current_section = m.group(2).strip()
            i += 1
            continue
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        start = i
        raw_lines = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            raw_lines.append(lines[i])
            i += 1
        end = i
        if len(raw_lines) < 2:
            unparsed.append({"source_section": current_section, "start_line": start + 1, "end_line": end, "reason": "too_few_table_lines", "raw_text": "\n".join(raw_lines)})
            continue
        header = _split_pipe_row(raw_lines[0])
        sep = _split_pipe_row(raw_lines[1])
        data_start = 2 if _is_separator(sep) else 1
        if not header or data_start >= len(raw_lines):
            unparsed.append({"source_section": current_section, "start_line": start + 1, "end_line": end, "reason": "missing_header_or_rows", "raw_text": "\n".join(raw_lines)})
            continue
        rows: list[dict[str, Any]] = []
        for off, raw in enumerate(raw_lines[data_start:], start=data_start):
            line_no = start + off + 1
            raw_cells = _split_pipe_row(raw)
            cells, warning = _normalize_cells(raw_cells, len(header))
            row = dict(zip(header, cells))
            row["source_section"] = current_section
            row["source_row_number"] = line_no
            row["raw_row_text"] = raw
            row["raw_row_hash"] = stable_hash(raw)
            row["source_cells_raw"] = json.dumps(raw_cells, ensure_ascii=False)
            row["parse_status"] = "parsed_with_warning" if warning else "parsed"
            row["parse_warning"] = warning
            rows.append(row)
        tables.append(ParsedTable(section=current_section, section_slug=slugify(current_section), header=header, rows=rows, start_line=start + 1, end_line=end, raw_text="\n".join(raw_lines)))
    return tables, unparsed


def find_table(tables: Iterable[ParsedTable], *needles: str) -> ParsedTable | None:
    lows = [n.lower() for n in needles]
    for t in tables:
        sec = t.section.lower()
        if any(n in sec for n in lows):
            return t
    return None


def table_to_df(table: ParsedTable | None, source_path: str | Path) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame()
    df = pd.DataFrame(table.rows)
    df.insert(0, "source_md_path", str(source_path))
    if "source_section" not in df.columns:
        df["source_section"] = table.section
    return df


def declared_counts(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    current_section = "document"
    for i, line in enumerate(text.splitlines(), start=1):
        m = HEADING_RE.match(line)
        if m:
            current_section = m.group(2).strip()
        for match in DECLARED_COUNT_RE.finditer(line):
            num = match.group(1) or match.group(2)
            if not num:
                continue
            try:
                n = int(num)
            except ValueError:
                continue
            if n < 3:
                continue
            rows.append({"source_md_path": str(p), "source_section": current_section, "line_number": i, "declared_count": n, "line_text": line.strip()})
    return rows


def detect_date_precision(value: Any) -> str:
    s = str(value if value is not None else "").strip()
    low = s.lower()
    if low in {"", "unknown", "null", "none", "nan", "undated current doc"}:
        return "unknown"
    if low.startswith("<="):
        return "lte_date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", low):
        return "exact_datetime"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", low):
        return "date_only"
    if re.fullmatch(r"\d{4}-\d{2}", low):
        return "month_only"
    if re.fullmatch(r"\d{4}", low):
        return "year_only"
    if re.search(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}", low):
        return "exact_datetime"
    if re.search(r"\d{4}-\d{2}-\d{2}", low):
        return "date_only"
    if re.search(r"\d{4}-\d{2}", low):
        return "month_only"
    if re.search(r"\b\d{4}\b", low):
        return "year_only"
    return "unknown"


def write_df(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, quoting=csv.QUOTE_MINIMAL)
