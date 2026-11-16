"""Timing attack harness: deterministic synthetic models + real measurement.

Two *synthetic* timing models let tests assert attack behaviour without any
wall-clock dependence:

* :class:`SyntheticNaiveModel` — duration grows with the length of the
  common prefix between secret and guess (exactly what an early-exit
  comparison leaks), plus seeded noise so distributions are realistic.
* :class:`SyntheticConstantTimeModel` — duration is independent of the
  guess (what ``ct_compare`` provides): only seeded noise remains.

Real models (:class:`RealNaiveModel`, :class:`RealConstantTimeModel`) wrap
the actual comparison functions with ``time.perf_counter``. They are used
for *report output only* — no test asserts on their recovery, because
CPython + OS scheduling noise makes real timing inherently machine-
dependent.

Guess construction uses **known context**: the attacker is assumed to know
everything except the byte under attack (typical when brute-forcing a short
unknown field inside a known-format token), so a correct candidate scans the
whole string while a wrong one stops at the unknown position.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Sequence

from ct import ct_compare, naive_compare
from stats import welch_t_test

__all__ = [
    "DEFAULT_LEAK",
    "DEFAULT_BASE",
    "DEFAULT_NOISE",
    "T_THRESHOLD",
    "measure",
    "common_prefix_len",
    "build_guess",
    "SyntheticNaiveModel",
    "SyntheticConstantTimeModel",
    "RealNaiveModel",
    "RealConstantTimeModel",
    "attack_position",
    "run_attack",
]

DEFAULT_LEAK = 1e-5    # seconds of leak per matching prefix byte
DEFAULT_BASE = 1e-4    # fixed overhead per comparison
DEFAULT_NOISE = 2e-6   # uniform noise amplitude per sample
T_THRESHOLD = 6.0      # a candidate must beat the runner-up by t > 6


def measure(
    fn: Callable[[], object], samples: int, clock: Callable[[], float] = time.perf_counter
) -> list[float]:
    """Run *fn* *samples* times, returning one duration per run.

    The clock is injectable so tests can drive this with a fake, fully
    deterministic time source (no sleeps, no wall-clock dependence).
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    durations = []
    for _ in range(samples):
        t0 = clock()
        fn()
        t1 = clock()
        durations.append(t1 - t0)
    return durations


def common_prefix_len(a: bytes, b: bytes) -> int:
    """Length of the longest common prefix of *a* and *b*."""
    n = len(a) if len(a) < len(b) else len(b)
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_guess(secret: bytes, position: int, candidate: int) -> bytes:
    """Guess with *candidate* at *position* and the known context around it.

    ``secret[:position] + bytes([candidate]) + secret[position+1:]`` — a
    correct candidate reproduces the secret exactly.
    """
    if not 0 <= position < len(secret):
        raise ValueError(f"position {position} out of range")
    if not 0 <= candidate <= 255:
        raise ValueError("candidate must be a byte value 0..255")
    return secret[:position] + bytes([candidate]) + secret[position + 1:]


class SyntheticNaiveModel:
    """Deterministic early-exit timing model: ``base + prefix*leak + noise``."""

    def __init__(
        self,
        secret: bytes,
        leak: float = DEFAULT_LEAK,
        base: float = DEFAULT_BASE,
        noise: float = DEFAULT_NOISE,
        seed: int = 0xA11CE,
    ) -> None:
        self.secret = bytes(secret)
        self.leak = leak
        self.base = base
        self.noise = noise
        self._rng = random.Random(seed)

    def time(self, guess: bytes) -> float:
        prefix = common_prefix_len(self.secret, guess)
        jitter = self.noise * self._rng.random()
        return self.base + prefix * self.leak + jitter


class SyntheticConstantTimeModel:
    """Deterministic constant-time timing model: ``base + noise`` only.

    The duration carries no information about *guess*: this is the model
    that :func:`ct_compare` exhibits.
    """

    def __init__(
        self,
        secret: bytes,
        base: float = DEFAULT_BASE,
        noise: float = DEFAULT_NOISE,
        seed: int = 0xB0B,
    ) -> None:
        self.secret = bytes(secret)
        self.base = base
        self.noise = noise
        self._rng = random.Random(seed)

    def time(self, guess: bytes) -> float:
        jitter = self.noise * self._rng.random()
        return self.base + jitter


class RealNaiveModel:
    """Wall-clock durations of the real early-exit comparison.

    Report output only — never asserted upon in tests.
    """

    def __init__(self, secret: bytes, clock=time.perf_counter) -> None:
        self.secret = bytes(secret)
        self._clock = clock

    def time(self, guess: bytes) -> float:
        t0 = self._clock()
        naive_compare(self.secret, guess)
        return self._clock() - t0


class RealConstantTimeModel:
    """Wall-clock durations of the real constant-time comparison.

    Report output only — never asserted upon in tests.
    """

    def __init__(self, secret: bytes, clock=time.perf_counter) -> None:
        self.secret = bytes(secret)
        self._clock = clock

    def time(self, guess: bytes) -> float:
        t0 = self._clock()
        ct_compare(self.secret, guess)
        return self._clock() - t0


def attack_position(
    model,
    secret: bytes,
    position: int,
    candidates: Sequence[int],
    samples: int,
    min_gap: float = 0.5 * DEFAULT_LEAK,
    t_threshold: float = T_THRESHOLD,
) -> tuple[int | None, dict]:
    """Attack one byte position.

    Collects *samples* durations per candidate, then compares the best
    candidate against the runner-up. A byte is only reported when the mean
    gap exceeds *min_gap* **and** Welch's t exceeds *t_threshold* — a
    candidate that is merely "fastest by noise" never counts as a signal.
    Returns ``(recovered_byte_or_None, info_dict)``.
    """
    if len(candidates) < 2:
        raise ValueError("need at least two candidates to compare")
    groups: dict[int, list[float]] = {}
    for cand in candidates:
        guess = build_guess(secret, position, cand)
        groups[cand] = [
            model.time(guess) for _ in range(samples)
        ]
    means = {c: sum(g) / len(g) for c, g in groups.items()}
    order = sorted(means, key=lambda c: means[c], reverse=True)
    best, second = order[0], order[1]
    gap = means[best] - means[second]
    result = welch_t_test(groups[best], groups[second])
    significant = bool(gap > min_gap and result.t > t_threshold)
    info = {
        "position": position,
        "best": best,
        "second": second,
        "means": means,
        "groups": groups,
        "gap": gap,
        "t": result.t,
        "df": result.df,
        "significant": significant,
    }
    return (best if significant else None), info


def run_attack(
    model,
    secret: bytes,
    candidates: Sequence[int] | None = None,
    samples: int = 12,
    min_gap: float = 0.5 * DEFAULT_LEAK,
    t_threshold: float = T_THRESHOLD,
    positions: Sequence[int] | None = None,
) -> dict:
    """Recover *secret* byte-by-byte; returns a full report dict.

    ``report["recovered"]`` is the list of per-position results (int or
    ``None``), ``report["match"]`` is True only when every attacked position
    was recovered exactly, and ``report["positions"]`` holds the per-position
    detail (means, t statistics, significance flags). *positions* defaults to
    every byte of *secret*.
    """
    if candidates is None:
        candidates = range(256)
    candidates = tuple(candidates)
    if positions is None:
        positions = range(len(secret))
    positions = tuple(positions)
    recovered: list[int | None] = []
    details: list[dict] = []
    for pos in positions:
        chosen, info = attack_position(
            model, secret, pos, candidates, samples, min_gap, t_threshold
        )
        recovered.append(chosen)
        details.append(info)
    match = bool(positions) and all(
        got == secret[pos] for pos, got in zip(positions, recovered)
    )
    return {
        "secret": bytes(secret),
        "attacked": positions,
        "recovered": recovered,
        "match": match,
        "positions": details,
        "samples": samples,
        "candidates": len(candidates),
    }
