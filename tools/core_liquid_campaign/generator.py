from __future__ import annotations

import hashlib
from typing import Iterator, Sequence

from .schema import CAMPAIGN_ID


GENERATOR_ID = "cranley_patterson_halton_v1"
GENERATOR_SEED = 20260722
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131)


def radical_inverse(index: int, base: int) -> float:
    if index <= 0 or base < 2:
        raise ValueError("radical inverse requires index>0 and base>=2")
    inverse = 1.0 / base
    factor = inverse
    value = 0.0
    remaining = index
    while remaining:
        remaining, digit = divmod(remaining, base)
        value += digit * factor
        factor *= inverse
    return value


def _shift(family_id: str, dimension: int) -> float:
    payload = f"{CAMPAIGN_ID}|{GENERATOR_ID}|{GENERATOR_SEED}|{family_id}|{dimension}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def point(family_id: str, stream_index: int, dimension: int) -> tuple[float, ...]:
    if stream_index < 0 or dimension <= 0 or dimension > len(PRIMES):
        raise ValueError("invalid Halton point request")
    return tuple((radical_inverse(stream_index + 1, PRIMES[axis]) + _shift(family_id, axis)) % 1.0 for axis in range(dimension))


def stream(family_id: str, dimension: int, start: int = 0) -> Iterator[tuple[int, tuple[float, ...]]]:
    index = start
    while True:
        yield index, point(family_id, index, dimension)
        index += 1


def one_dimensional_discrepancy(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    count = len(ordered)
    return max(max((index + 1) / count - value, value - index / count) for index, value in enumerate(ordered))


def approximate_discrepancy(points: Sequence[Sequence[float]]) -> float:
    if not points:
        return 1.0
    return max(one_dimensional_discrepancy([point_[dimension] for point_ in points]) for dimension in range(len(points[0])))


def approximate_nearest_neighbor(points: Sequence[Sequence[float]], limit: int = 512) -> dict[str, float]:
    selected = [tuple(float(value) for value in item) for item in points[:limit]]
    if len(selected) < 2:
        return {"minimum": 0.0, "median": 0.0, "sample_size": float(len(selected))}
    distances = []
    for index, left in enumerate(selected):
        nearest = min(sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5 for other, right in enumerate(selected) if other != index)
        distances.append(nearest)
    ordered = sorted(distances)
    return {"minimum": ordered[0], "median": ordered[len(ordered) // 2], "sample_size": float(len(selected))}
