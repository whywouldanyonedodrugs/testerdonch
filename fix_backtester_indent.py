#!/usr/bin/env python3
from pathlib import Path
import re
import sys

def die(msg: str) -> None:
    print(f"[fix] ERROR: {msg}")
    sys.exit(1)

def find_line(lines, pattern, start=0):
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i
    return None

def add_indent(lines, i, n=4):
    if lines[i].strip() == "":
        return
    lines[i] = (" " * n) + lines[i]

def add_indent_range(lines, start, end, n=4):
    for i in range(start, end):
        if lines[i].strip() == "":
            continue
        lines[i] = (" " * n) + lines[i]

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("backtester.py")
    if not path.exists():
        die(f"File not found: {path}")

    raw = path.read_text().splitlines(True)

    cls_i = find_line(raw, r"^class Backtester:\s*$")
    if cls_i is None:
        die("Could not find 'class Backtester:'")

    daily_i = find_line(raw, r"^def _daily_regime_at\(", cls_i)
    if daily_i is None:
        die("Could not find outdented 'def _daily_regime_at(' after class Backtester")

    init_i = find_line(raw, r"^def __init__\(", cls_i)
    if init_i is None:
        die("Could not find outdented 'def __init__(' after class Backtester")

    markov4h_i = find_line(raw, r"^\s{4}def _markov4h_at\(", init_i)
    if markov4h_i is None:
        die("Could not find '    def _markov4h_at(' after __init__ (needed to detect __init__ body end)")

    if not (cls_i < daily_i < init_i < markov4h_i):
        die("Unexpected ordering of Backtester / _daily_regime_at / __init__ / _markov4h_at blocks")

    # 1) Indent the entire _daily_regime_at function block into the class
    add_indent_range(raw, daily_i, init_i, 4)

    # 2) Indent the __init__ signature into the class
    if not raw[init_i].startswith(" " * 4):
        raw[init_i] = (" " * 4) + raw[init_i]

    # 3) Indent only the __init__ BODY by +4 spaces (so it becomes 8 total)
    #    Stop right before the next class method '    def _markov4h_at('
    add_indent_range(raw, init_i + 1, markov4h_i, 4)

    # Write backup + patched file
    bak = path.with_suffix(path.suffix + ".bak_autofix")
    bak.write_text("".join(path.read_text().splitlines(True)))
    path.write_text("".join(raw))

    print(f"[fix] OK. Wrote patched file: {path}")
    print(f"[fix] Backup saved as: {bak}")

if __name__ == "__main__":
    main()
