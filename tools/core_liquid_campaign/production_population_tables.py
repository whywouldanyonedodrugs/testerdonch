from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .engine_types import ExactPopulationTableView, ThresholdPopulation
from .family_engines import a1_compression
from .production_cache import ALLOWED_CANDLE_COLUMNS, EMPTY_CANDLE_COLUMNS, SourcePart


UTC = timezone.utc
FIVE_MINUTES_MS = 300_000
DAY_MS = 86_400_000
IMPULSE_WINDOWS = {"6h": 72, "12h": 144, "1d": 288, "3d": 864, "7d": 2016}
BASE_WINDOWS = {"2h": 24, "6h": 72, "12h": 144, "1d": 288, "3d": 864}


class PopulationTableError(RuntimeError):
    pass


def _npy_sha256(path: Path) -> str:
    return sha256_file(path)


def _slice_sha256(array) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _feature_arrays(times, closes) -> dict[str, Any]:
    """Vectorized exact-boundary A1 raw features; unavailable rows are NaN."""
    import numpy as np

    times = np.asarray(times, dtype="<i8"); closes = np.asarray(closes, dtype="<f8")
    if times.ndim != 1 or closes.ndim != 1 or len(times) != len(closes) or len(times) < 2:
        raise PopulationTableError("A1 source arrays are empty or mismatched")
    if np.any(np.diff(times) <= 0) or np.any(~np.isfinite(closes)) or np.any(closes <= 0):
        raise PopulationTableError("A1 source arrays are unsorted or invalid")
    result: dict[str, Any] = {}
    size = len(times)
    for name, count in IMPULSE_WINDOWS.items():
        values = np.full(size, np.nan, dtype="<f8")
        indices = np.arange(count, size)
        contiguous = times[indices] - times[indices - count] == count * FIVE_MINUTES_MS
        selected = indices[contiguous]
        values[selected] = np.log(closes[selected] / closes[selected - count])
        result[f"A1_impulse:window={name}"] = values

    returns = np.log(closes[1:] / closes[:-1])
    prefix = np.concatenate((np.zeros(1, dtype="<f8"), np.cumsum(returns, dtype="<f8")))
    squares = np.concatenate((np.zeros(1, dtype="<f8"), np.cumsum(returns * returns, dtype="<f8")))
    absolute = np.concatenate((np.zeros(1, dtype="<f8"), np.cumsum(np.abs(returns), dtype="<f8")))

    def rolling_std(start, end):
        count = end - start
        total = prefix[end] - prefix[start]
        square = squares[end] - squares[start]
        return np.sqrt(np.maximum(0.0, (square - total * total / count) / (count - 1)))

    for name, count in BASE_WINDOWS.items():
        smooth = np.full(size, np.nan, dtype="<f8")
        indices = np.arange(count - 1, size)
        starts = indices - count + 1
        contiguous = times[indices] - times[starts] == (count - 1) * FIVE_MINUTES_MS
        selected = indices[contiguous]; selected_starts = starts[contiguous]
        path = absolute[selected] - absolute[selected_starts]
        valid = path > 0
        smooth_selected = selected[valid]; smooth_starts = selected_starts[valid]
        smooth[smooth_selected] = np.abs(np.log(closes[smooth_selected] / closes[smooth_starts])) / path[valid]
        result[f"A1_smoothness:base={name}"] = smooth

        for baseline_name, multiple in (("adjacent_equal_duration", 1), ("trailing_5x_base_duration", 5)):
            baseline_count = count * multiple
            minimum = count + baseline_count - 1
            contraction = np.full(size, np.nan, dtype="<f8")
            indices = np.arange(minimum, size)
            contiguous = times[indices] - times[indices - minimum] == minimum * FIVE_MINUTES_MS
            selected = indices[contiguous]
            base_start = selected - count + 1; base_end = selected
            baseline_start = selected - count - baseline_count + 1; baseline_end = selected - count
            base_vol = rolling_std(base_start, base_end)
            baseline_vol = rolling_std(baseline_start, baseline_end)
            valid = np.isfinite(base_vol) & np.isfinite(baseline_vol) & (baseline_vol > 0)
            contraction[selected[valid]] = base_vol[valid] / baseline_vol[valid]
            result[f"A1_contraction:base={name}:baseline={baseline_name}"] = contraction
    return result


def _load_trade_arrays(parts: Sequence[SourcePart]):
    import numpy as np
    import pyarrow.parquet as pq

    rows: list[Any] = []
    for part in parts:
        if part.dataset != "historical_trade_candles_5m":
            continue
        path = Path(part.path)
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if columns == EMPTY_CANDLE_COLUMNS:
            continue
        if columns != ALLOWED_CANDLE_COLUMNS:
            raise PopulationTableError("trade source schema differs while building population table")
        table = parquet.read(columns=["time", "close"])
        times = table["time"].combine_chunks().to_numpy(zero_copy_only=False).astype("<i8", copy=False)
        closes = table["close"].combine_chunks().to_numpy(zero_copy_only=False).astype("<f8", copy=False)
        if len(times):
            rows.append((times, closes))
    if not rows:
        raise PopulationTableError("symbol has no physical trade rows")
    output_times: list[Any] = []; output_closes: list[Any] = []
    previous_time: int | None = None; previous_close: float | None = None
    for times, closes in rows:
        start = 0
        if previous_time is not None and int(times[0]) == previous_time:
            if float(closes[0]) != previous_close:
                raise PopulationTableError("trade source boundary duplicate differs")
            start = 1
        elif previous_time is not None and int(times[0]) < previous_time:
            raise PopulationTableError("trade source parts overlap")
        output_times.append(times[start:]); output_closes.append(closes[start:])
        previous_time = int(times[-1]); previous_close = float(closes[-1])
    times = np.concatenate(output_times); closes = np.concatenate(output_closes)
    if len(times) != len(np.unique(times)):
        raise PopulationTableError("trade source contains duplicate timestamps")
    return times, closes


@dataclass
class A1PopulationTableCompiler:
    cache_root: Path
    parts_by_symbol: Mapping[str, Sequence[SourcePart]]
    pit_rows: Sequence[Mapping[str, Any]]

    def build(self) -> dict[str, Any]:
        import numpy as np
        from numpy.lib.format import open_memmap

        symbols = tuple(sorted(self.parts_by_symbol))
        symbol_codes = {symbol: index + 1 for index, symbol in enumerate(symbols)}
        pit_by_symbol: dict[str, dict[int, Mapping[str, Any]]] = {symbol: {} for symbol in symbols}
        for row in self.pit_rows:
            symbol = str(row["symbol"])
            if symbol not in pit_by_symbol:
                raise PopulationTableError("PIT table contains an unregistered symbol")
            day = int(row["day_open_ms"])
            if day in pit_by_symbol[symbol]:
                raise PopulationTableError("PIT table contains a duplicate symbol-day")
            pit_by_symbol[symbol][day] = row
        expected_counts = {symbol: sum(int(row["decision_count_5m"]) for row in pit_by_symbol[symbol].values()) for symbol in symbols}
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for symbol in symbols:
            offsets[symbol] = (cursor, cursor + expected_counts[symbol]); cursor += expected_counts[symbol]
        total = cursor
        if total <= 0:
            raise PopulationTableError("PIT population table would be empty")

        root = self.cache_root / "population_tables/a1"
        root.mkdir(parents=True, exist_ok=True)
        common_specs = {
            "timestamps": ("<i8", root / "timestamps.npy"),
            "symbols": ("<u2", root / "symbols.npy"),
            "deciles": ("u1", root / "deciles.npy"),
        }
        feature_names = tuple(
            [f"A1_impulse:window={name}" for name in IMPULSE_WINDOWS]
            + [item for name in BASE_WINDOWS for item in (
                f"A1_contraction:base={name}:baseline=adjacent_equal_duration",
                f"A1_contraction:base={name}:baseline=trailing_5x_base_duration",
                f"A1_smoothness:base={name}",
            )]
        )
        feature_paths = {name: root / f"feature-{canonical_hash(name)}.npy" for name in feature_names}

        def mmap(path: Path, dtype: str, *, fill: float | int | None = None):
            if path.exists():
                array = open_memmap(path, mode="r+")
                if array.shape != (total,) or array.dtype != np.dtype(dtype):
                    raise PopulationTableError("resumable population table shape or dtype differs")
                return array
            array = open_memmap(path, mode="w+", dtype=dtype, shape=(total,))
            if fill is not None:
                array[:] = fill; array.flush()
            return array

        common = {name: mmap(path, dtype) for name, (dtype, path) in common_specs.items()}
        features = {name: mmap(path, "<f8") for name, path in feature_paths.items()}
        progress_path = root / "BUILD_PROGRESS.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {
            "schema": "stage24_a1_population_table_progress_v1", "total_rows": total,
            "symbol_order_sha256": canonical_hash(symbols), "completed": {},
        }
        if progress.get("total_rows") != total or progress.get("symbol_order_sha256") != canonical_hash(symbols):
            raise PopulationTableError("resumable population-table identity differs")

        for symbol in symbols:
            start, end = offsets[symbol]
            source_identity = canonical_hash([part.payload() for part in self.parts_by_symbol[symbol] if part.dataset == "historical_trade_candles_5m"])
            completed = progress["completed"].get(symbol)
            if isinstance(completed, Mapping) and completed.get("source_identity_sha256") == source_identity and completed.get("row_count") == end - start:
                continue
            times, closes = _load_trade_arrays(self.parts_by_symbol[symbol])
            source_features = _feature_arrays(times, closes)
            selected_indices: list[int] = []; selected_deciles: list[int] = []
            for index, timestamp in enumerate(times):
                row = pit_by_symbol[symbol].get(int(timestamp) // DAY_MS * DAY_MS)
                if row is None:
                    continue
                selected_indices.append(index)
                rank = float(row["average_liquidity_rank"]); population = int(row["eligible_population"])
                selected_deciles.append(1 + min(9, int((rank - 1) * 10 / population)))
            if len(selected_indices) != end - start:
                raise PopulationTableError(f"PIT decision count differs from physical rows: {symbol}")
            indices = np.asarray(selected_indices, dtype=np.int64)
            common["timestamps"][start:end] = times[indices] + FIVE_MINUTES_MS
            common["symbols"][start:end] = symbol_codes[symbol]
            common["deciles"][start:end] = np.asarray(selected_deciles, dtype="u1")
            feature_slice_hashes = {}
            for name in feature_names:
                features[name][start:end] = source_features[name][indices]
                features[name].flush()
                feature_slice_hashes[name] = _slice_sha256(features[name][start:end])
            for array in common.values():
                array.flush()
            progress["completed"][symbol] = {
                "source_identity_sha256": source_identity, "row_count": end - start,
                "offset": [start, end], "feature_slice_inventory_sha256": canonical_hash(feature_slice_hashes),
            }
            atomic_write_json(progress_path, progress)

        if set(progress["completed"]) != set(symbols):
            raise PopulationTableError("population-table build did not complete every symbol")
        for array in (*common.values(), *features.values()):
            array.flush()
        rankable_start_day = 1_672_531_200_000 // DAY_MS
        day_indices = common["timestamps"] // DAY_MS - rankable_start_day
        day_count = int(day_indices.max()) + 1
        count_paths: dict[str, Path] = {}
        for name in feature_names:
            count_path = root / f"counts-{canonical_hash(name)}.npy"
            count_paths[name] = count_path
            if count_path.exists():
                existing = np.load(count_path, mmap_mode="r", allow_pickle=False)
                if existing.shape != (day_count, 11 + len(symbols)) or existing.dtype != np.dtype("<i4"):
                    raise PopulationTableError("population count table shape or dtype differs")
                continue
            values = features[name]
            finite = np.isfinite(values)
            days = np.asarray(day_indices[finite], dtype=np.int64)
            deciles = np.asarray(common["deciles"][finite], dtype=np.int64)
            codes = np.asarray(common["symbols"][finite], dtype=np.int64)
            counts = np.zeros((day_count, 11 + len(symbols)), dtype="<i4")
            counts[:, 0] = np.bincount(days, minlength=day_count)
            decile_counts = np.bincount(days * 11 + deciles, minlength=day_count * 11).reshape(day_count, 11)
            counts[:, 1:11] = decile_counts[:, 1:11]
            symbol_width = len(symbols) + 1
            symbol_counts = np.bincount(days * symbol_width + codes, minlength=day_count * symbol_width).reshape(day_count, symbol_width)
            counts[:, 11:] = symbol_counts[:, 1:]
            with count_path.open("wb") as handle:
                np.save(handle, counts, allow_pickle=False)
        common_records = {
            name: {"path": path.relative_to(self.cache_root).as_posix(), "bytes": path.stat().st_size, "sha256": _npy_sha256(path), "rows": total}
            for name, (_, path) in common_specs.items()
        }
        feature_records = {
            name: {
                "path": path.relative_to(self.cache_root).as_posix(), "bytes": path.stat().st_size,
                "sha256": _npy_sha256(path), "rows": total,
                "daily_counts_path": count_paths[name].relative_to(self.cache_root).as_posix(),
                "daily_counts_bytes": count_paths[name].stat().st_size,
                "daily_counts_sha256": _npy_sha256(count_paths[name]),
            }
            for name, path in feature_paths.items()
        }
        manifest = {
            "schema": "stage24_a1_exact_pit_population_table_v1",
            "rows": total, "symbols": len(symbols), "symbol_codes": symbol_codes,
            "common": common_records, "features": feature_records,
            "feature_signature_inventory_sha256": canonical_hash(feature_records),
            "daily_count_columns": {"global": 0, "liquidity_deciles": {str(value): value for value in range(1, 11)}, "symbols": {symbol: 10 + code for symbol, code in symbol_codes.items()}},
            "daily_count_rankable_start_day_ms": rankable_start_day * DAY_MS,
            "daily_count_days": day_count,
            "symbol_offsets_sha256": canonical_hash(offsets),
            "pit_row_count": len(self.pit_rows), "pit_content_sha256": canonical_hash(self.pit_rows),
            "protected_rows": int(np.count_nonzero(common["timestamps"] >= 1_767_225_600_000)),
            "formula": {
                "observation": "every PIT-eligible completed five-minute boundary",
                "gap": "feature unavailable unless the complete exact five-minute window is contiguous",
                "impulse": "side*log(close_t/close_t_minus_N)",
                "contraction": "sample_sd(base adjacent log returns)/sample_sd(immediately preceding registered baseline adjacent log returns)",
                "smoothness": "abs(log_return(base_start,base_end))/sum(abs(adjacent log returns))",
            },
        }
        if manifest["protected_rows"]:
            raise PopulationTableError("population table contains protected rows")
        manifest_path = root / "A1_POPULATION_TABLE_MANIFEST.json"
        atomic_write_json(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}


class A1PopulationTableAuthority:
    """Create exact fold/scope views from the physically verified shared table."""

    def __init__(self, cache_root: Path, manifest_path: Path) -> None:
        import numpy as np

        self.cache_root = cache_root
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "stage24_a1_exact_pit_population_table_v1" or self.manifest.get("protected_rows") != 0:
            raise PopulationTableError("A1 population table manifest is invalid")
        self.manifest_sha256 = sha256_file(manifest_path)
        self.symbol_codes = {str(key): int(value) for key, value in self.manifest["symbol_codes"].items()}
        self.symbol_by_code = {value: key for key, value in self.symbol_codes.items()}
        self._common = self.manifest["common"]
        self._verified_minimums: dict[tuple[Any, ...], tuple[str, ...]] = {}
        for record in [*self._common.values(), *self.manifest["features"].values()]:
            path = self.cache_root / record["path"]
            if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
                raise PopulationTableError("A1 population table component differs")
            if "daily_counts_path" in record:
                count_path = self.cache_root / record["daily_counts_path"]
                if not count_path.is_file() or count_path.stat().st_size != int(record["daily_counts_bytes"]) or sha256_file(count_path) != record["daily_counts_sha256"]:
                    raise PopulationTableError("A1 population daily-count component differs")
        self._timestamps = np.load(self.cache_root / self._common["timestamps"]["path"], mmap_mode="r", allow_pickle=False)
        self._symbols = np.load(self.cache_root / self._common["symbols"]["path"], mmap_mode="r", allow_pickle=False)
        self._deciles = np.load(self.cache_root / self._common["deciles"]["path"], mmap_mode="r", allow_pickle=False)

    @staticmethod
    def _base_feature(name: str) -> tuple[str, int]:
        if name.startswith("A1_impulse:"):
            fields = dict(item.split("=", 1) for item in name.split(":")[1:])
            return f"A1_impulse:window={fields['window']}", int(fields["side"])
        if name.startswith("A1_contraction:"):
            fields = dict(item.split("=", 1) for item in name.split(":")[1:])
            return f"A1_contraction:base={fields['base']}:baseline={fields['baseline']}", 1
        if name.startswith("A1_smoothness:"):
            fields = dict(item.split("=", 1) for item in name.split(":")[1:])
            return f"A1_smoothness:base={fields['base']}", 1
        raise PopulationTableError(f"unsupported A1 population signature: {name}")

    @staticmethod
    def _scope(name: str) -> str:
        fields = dict(item.split("=", 1) for item in name.split(":")[1:])
        return str(fields["scope"])

    def _selector_minimum(
        self,
        *,
        feature_name: str,
        training_start_ms: int,
        training_end_ms: int,
        symbol_code: int | None,
        decile: int | None,
        multiplier: int,
    ) -> tuple[str, ...]:
        import numpy as np

        key = (feature_name, training_start_ms, symbol_code, decile, multiplier)
        cached = self._verified_minimums.get(key)
        if cached is not None:
            return cached
        feature = self.manifest["features"][feature_name]
        values = np.load(self.cache_root / feature["path"], mmap_mode="r", allow_pickle=False)
        mask = (self._timestamps >= training_start_ms) & (self._timestamps < training_end_ms) & np.isfinite(values)
        if symbol_code is not None:
            mask &= self._symbols == symbol_code
        if decile is not None:
            mask &= self._deciles == decile
        selected = np.asarray(values[mask], dtype="<f8") * multiplier
        if len(selected) < 30 or int(np.unique(selected).size) < 20:
            raise PopulationTableError("A1 population selector fails the frozen 30-row/20-unique minimum")
        contributing = tuple(self.symbol_by_code[int(code)] for code in np.unique(self._symbols[mask]))
        if (symbol_code is None or decile is not None) and len(contributing) < 5:
            raise PopulationTableError("A1 pooled population selector has fewer than five symbols")
        self._verified_minimums[key] = contributing
        return contributing

    def population(
        self,
        name: str,
        *,
        target_symbol: str,
        target_decile: int,
        training_start: datetime,
        training_end: datetime,
    ) -> ThresholdPopulation:
        import numpy as np

        feature_name, multiplier = self._base_feature(name)
        feature = self.manifest["features"].get(feature_name)
        if feature is None:
            raise PopulationTableError("A1 population feature is absent")
        scope = self._scope(name)
        symbol_code = self.symbol_codes[target_symbol] if scope in {"symbol", "symbol_side"} else None
        decile = target_decile if "liquidity_decile" in scope else None
        start_ms = int(training_start.timestamp() * 1000); end_ms = int(training_end.timestamp() * 1000)
        day_zero = int(self.manifest["daily_count_rankable_start_day_ms"])
        start_day = max(0, (start_ms - day_zero) // DAY_MS)
        end_day = max(0, (end_ms - day_zero) // DAY_MS)
        counts = np.load(self.cache_root / feature["daily_counts_path"], mmap_mode="r", allow_pickle=False)
        if scope in {"symbol", "symbol_side"}:
            column = int(self.manifest["daily_count_columns"]["symbols"][target_symbol])
        elif decile is not None:
            column = int(self.manifest["daily_count_columns"]["liquidity_deciles"][str(decile)])
        else:
            column = 0
        selected_count = int(counts[start_day:min(end_day, len(counts)), column].sum())
        contributing = self._selector_minimum(
            feature_name=feature_name, training_start_ms=start_ms, training_end_ms=end_ms,
            symbol_code=symbol_code, decile=decile, multiplier=multiplier,
        )
        view = ExactPopulationTableView(
            values_path=str(feature["path"]), values_sha256=str(feature["sha256"]),
            timestamps_path=str(self._common["timestamps"]["path"]), timestamps_sha256=str(self._common["timestamps"]["sha256"]),
            symbols_path=str(self._common["symbols"]["path"]), symbols_sha256=str(self._common["symbols"]["sha256"]),
            deciles_path=str(self._common["deciles"]["path"]), deciles_sha256=str(self._common["deciles"]["sha256"]),
            physical_count=int(self.manifest["rows"]), training_start_ms=start_ms, training_end_ms=end_ms,
            selected_count=selected_count, unique_count=None, minimum_unique_count_verified=20,
            value_multiplier=multiplier, symbol_code=symbol_code, liquidity_decile=decile,
            root=str(self.cache_root), physical_verified=True,
        )
        source_sha256 = canonical_hash({"table_manifest_sha256": self.manifest_sha256, "population": view.identity_payload(), "multiplier": multiplier})
        return ThresholdPopulation(
            view, contributing, scope, training_start, training_end,
            training_end - timedelta(minutes=5),
            training_end, source_sha256,
        )


__all__ = ["A1PopulationTableAuthority", "A1PopulationTableCompiler", "PopulationTableError", "_feature_arrays"]
