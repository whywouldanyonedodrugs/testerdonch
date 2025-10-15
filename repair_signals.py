# repair_signals_dir.py
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SRC = Path("signals")              # <-- your current signals folder
DST = Path("signals_fixed")        # <-- will write repaired files here

def cast_to_string(tbl, name):
    i = tbl.schema.get_field_index(name)
    if i != -1:
        col = tbl.column(i)
        # cast dictionary or any non-string to string
        if not pa.types.is_string(col.type):
            col = pc.cast(col, pa.string())
            tbl = tbl.set_column(i, name, col)
    return tbl

def main():
    files = [p for p in SRC.rglob("*.parquet") if p.is_file()]
    for p in files:
        t = pq.ParquetFile(p).read()              # read per-file (no schema merge)
        t = cast_to_string(t, "symbol")
        # (optional) if you also have 'entry_rule' or other categoricals, normalize them too:
        # t = cast_to_string(t, "entry_rule")

        rel = p.relative_to(SRC)
        outp = DST / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        # write without dictionary encoding to avoid future mismatches
        pq.write_table(t, outp, compression="zstd", use_dictionary=False)
    print(f"Rewrote {len(files)} files to {DST}")

if __name__ == "__main__":
    main()
