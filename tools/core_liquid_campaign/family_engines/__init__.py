from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from types import ModuleType


_MODULES = {
    "A4_TSMOM_V7": "a4_tsmom",
    "A1_COMPRESSION_V2": "a1_compression",
    "A2_PRIOR_HIGH_RS_CONTEXT_V1": "a2_context",
    "A3_STARTER_RETEST_V3": "a3_starter_retest",
    "KDA02B_SURVIVOR_ADJUDICATION_V1": "kda02b_adjudication",
}


def __getattr__(name: str) -> ModuleType:
    if name in _MODULES.values():
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)


class _EngineMapping(Mapping[str, ModuleType]):
    def __getitem__(self, key: str) -> ModuleType:
        return importlib.import_module(f"{__name__}.{_MODULES[key]}")

    def __iter__(self) -> Iterator[str]:
        return iter(_MODULES)

    def __len__(self) -> int:
        return len(_MODULES)


ENGINES: Mapping[str, ModuleType] = _EngineMapping()

__all__ = ["ENGINES"]
