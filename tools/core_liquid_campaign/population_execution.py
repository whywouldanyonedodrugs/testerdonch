from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import sha256_file
from .launch_population_authority import validate_launch_population_authority
from .lazy_production_inputs import FamilyDecisionLocator


UTC = timezone.utc
DAY_MS = 86_400_000


class PopulationExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PopulationPartition:
    phase: str
    outer_fold_id: str
    inner_fold_id: str | None
    evaluation_start_ms: int
    evaluation_end_exclusive_ms: int

    @property
    def key(self) -> tuple[str, str, str | None]:
        return self.phase, self.outer_fold_id, self.inner_fold_id


class LaunchPopulationSchedule:
    """Deterministic lazy decision schedule over the complete PIT authority.

    The object exposes exact counts without materializing millions of job
    objects.  Iteration is ordered by UTC day, PIT rank, symbol and timestamp,
    so a supervisor may submit bounded work while preserving deterministic
    restart identities.
    """

    def __init__(self, authority_path: Path, expected_sha256: str) -> None:
        self.authority_path = authority_path.resolve()
        if not self.authority_path.is_file() or sha256_file(self.authority_path) != expected_sha256:
            raise PopulationExecutionError("launch population authority bytes differ")
        self.authority = json.loads(self.authority_path.read_text(encoding="utf-8"))
        validate_launch_population_authority(self.authority)
        partition_rows = self.authority["population_census"]["partitions"]
        self.partitions = {
            (str(row["phase"]), str(row["outer_fold_id"]), None if row.get("inner_fold_id") is None else str(row["inner_fold_id"])):
            PopulationPartition(
                str(row["phase"]), str(row["outer_fold_id"]),
                None if row.get("inner_fold_id") is None else str(row["inner_fold_id"]),
                int(row["evaluation_start_ms"]), int(row["evaluation_end_exclusive_ms"]),
            )
            for row in partition_rows
        }
        if len(self.partitions) != 132:
            raise PopulationExecutionError("launch population partition inventory differs")
        pit_record = self.authority["pit_membership"]
        pit_path = Path(str(pit_record["path"]))
        if not pit_path.is_file() or pit_path.stat().st_size != int(pit_record["bytes"]) or sha256_file(pit_path) != pit_record["sha256"]:
            raise PopulationExecutionError("launch PIT membership bytes differ")
        rows = [json.loads(line) for line in pit_path.read_text(encoding="utf-8").splitlines() if line]
        self._rows = tuple(sorted(rows, key=lambda row: (int(row["day_open_ms"]), float(row["average_liquidity_rank"]), str(row["symbol"]))))

    @staticmethod
    def _top_n(row: Mapping[str, Any]) -> int:
        try:
            value = int(row["config"]["PIT_liquidity_top_n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PopulationExecutionError("registered attempt lacks PIT_liquidity_top_n") from exc
        if value not in {10, 20, 40}:
            raise PopulationExecutionError("registered PIT_liquidity_top_n is outside the frozen values")
        return value

    def partition(self, *, phase: str, outer_fold_id: str, inner_fold_id: str | None) -> PopulationPartition:
        row = self.partitions.get((phase, outer_fold_id, inner_fold_id))
        if row is None:
            raise PopulationExecutionError("requested launch partition is absent")
        return row

    def _eligible_rows(self, partition: PopulationPartition, top_n: int) -> Iterable[Mapping[str, Any]]:
        field = f"top_{top_n}"
        return (
            row for row in self._rows
            if partition.evaluation_start_ms <= int(row["day_open_ms"]) < partition.evaluation_end_exclusive_ms
            and bool(row[field])
        )

    @staticmethod
    def _decision_offsets(row: Mapping[str, Any], family: str, config: Mapping[str, Any]) -> range | tuple[int, ...]:
        count = int(row["decision_count_5m"])
        if count not in {1, 288}:
            raise PopulationExecutionError("PIT row has an unresolved partial schedule")
        if family == "A4_TSMOM_V7":
            rebalance = str(config.get("rebalance"))
            if rebalance == "1d":
                return (0,)
            if rebalance == "8h":
                return (0,) if count == 1 else (0, 96, 192)
            raise PopulationExecutionError("A4 registered rebalance is invalid")
        if family in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"}:
            return range(count)
        raise PopulationExecutionError("only parent A1/A3/A4 schedules can be enumerated directly")

    def count(self, attempt: Mapping[str, Any], *, phase: str, outer_fold_id: str, inner_fold_id: str | None) -> int:
        family = str(attempt["family_id"]); config = attempt["config"]
        if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            raise PopulationExecutionError("A2 count requires its exact resolved parent attempt")
        partition = self.partition(phase=phase, outer_fold_id=outer_fold_id, inner_fold_id=inner_fold_id)
        top_n = self._top_n(attempt)
        return sum(len(self._decision_offsets(row, family, config)) for row in self._eligible_rows(partition, top_n))

    def iter_locators(
        self,
        attempt: Mapping[str, Any],
        *,
        phase: str,
        outer_fold_id: str,
        inner_fold_id: str | None,
        parent_attempt: Mapping[str, Any] | None = None,
    ) -> Iterable[FamilyDecisionLocator]:
        family = str(attempt["family_id"])
        source = attempt
        if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            if parent_attempt is None or parent_attempt.get("family_id") != attempt.get("config", {}).get("parent_family"):
                raise PopulationExecutionError("A2 schedule lacks its exact registered parent")
            source = parent_attempt
        source_family = str(source["family_id"]); config = source["config"]
        partition = self.partition(phase=phase, outer_fold_id=outer_fold_id, inner_fold_id=inner_fold_id)
        top_n = self._top_n(source)
        output_family = family
        for row in self._eligible_rows(partition, top_n):
            day = datetime.fromtimestamp(int(row["day_open_ms"]) / 1000, tz=UTC)
            for offset in self._decision_offsets(row, source_family, config):
                yield FamilyDecisionLocator(
                    output_family, phase, outer_fold_id, inner_fold_id,
                    str(row["symbol"]), day + timedelta(minutes=5 * offset),
                    str(attempt["executable_attempt_id"]),
                    str(attempt["canonical_economic_address_sha256"]),
                )

    def representative_locators(
        self,
        attempt: Mapping[str, Any],
        *,
        phase: str,
        outer_fold_id: str,
        inner_fold_id: str | None,
        symbol_size_bytes: Mapping[str, int],
        parent_attempt: Mapping[str, Any] | None = None,
    ) -> tuple[FamilyDecisionLocator, ...]:
        """Return small/median/large source-size locators for one partition.

        The sample is selected entirely from immutable input size and PIT
        membership.  It never reads feature values, payoffs, ranks or outcomes.
        All returned locators remain ordinary launch-authority locators and are
        consumed by the same production ``FamilyInput`` adapter.
        """
        family = str(attempt["family_id"])
        source = attempt
        if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            if parent_attempt is None or parent_attempt.get("family_id") != attempt.get("config", {}).get("parent_family"):
                raise PopulationExecutionError("A2 representative schedule lacks its exact parent")
            source = parent_attempt
        source_family = str(source["family_id"]); config = source["config"]
        partition = self.partition(phase=phase, outer_fold_id=outer_fold_id, inner_fold_id=inner_fold_id)
        top_n = self._top_n(source)
        rows = list(self._eligible_rows(partition, top_n))
        if not rows:
            raise PopulationExecutionError("representative partition has no PIT-eligible rows")
        symbols = sorted(
            {str(row["symbol"]) for row in rows},
            key=lambda symbol: (int(symbol_size_bytes.get(symbol, -1)), symbol),
        )
        if any(symbol not in symbol_size_bytes or int(symbol_size_bytes[symbol]) <= 0 for symbol in symbols):
            raise PopulationExecutionError("representative source-size authority is incomplete")
        selected_symbols = tuple(dict.fromkeys((symbols[0], symbols[len(symbols) // 2], symbols[-1])))
        output = []
        for symbol in selected_symbols:
            row = next(item for item in rows if str(item["symbol"]) == symbol)
            offsets = self._decision_offsets(row, source_family, config)
            offset = next(iter(offsets), None)
            if offset is None:
                raise PopulationExecutionError("representative route contains no decision offset")
            day = datetime.fromtimestamp(int(row["day_open_ms"]) / 1000, tz=UTC)
            output.append(FamilyDecisionLocator(
                family, phase, outer_fold_id, inner_fold_id,
                symbol, day + timedelta(minutes=5 * int(offset)),
                str(attempt["executable_attempt_id"]),
                str(attempt["canonical_economic_address_sha256"]),
            ))
        if len(output) != 3:
            raise PopulationExecutionError("representative partition lacks three source-size strata")
        return tuple(output)


__all__ = ["LaunchPopulationSchedule", "PopulationExecutionError", "PopulationPartition"]
