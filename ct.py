"""Timing-safe comparison primitives and the naive baseline.

``ct_compare`` never branches on secret *content*: it normalizes both inputs
to bytes, then XOR-accumulates over ``max(len(a), len(b))`` positions — the
work performed depends only on public lengths, not on where (or whether) the
inputs differ. ``naive_compare`` is the classic early-exit implementation
kept here purely as the leaky baseline for the attack demo.

``ct_select`` is a branchless conditional: a bitmask derived from the
condition selects between two integer or equal-length byte operands without
testing the operands themselves.
"""

from __future__ import annotations

__all__ = [
    "ct_compare",
    "ct_verify_mac",
    "ct_verify_mac_hex",
    "ct_select",
    "naive_compare",
]


def _as_bytes(value, name: str) -> bytes:
    """Normalize *value* to ``bytes`` (public-length work only)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return bytes(value)  # sequence of ints, e.g. test instrumentation
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be bytes-like, str or a sequence of ints, "
            f"got {type(value).__name__}"
        ) from exc


def ct_compare(a, b) -> bool:
    """Constant-time equality: fixed-length XOR accumulation.

    Both inputs are normalized to bytes first, then every position up to
    ``max(len(a), len(b))`` is XORed into a single accumulator. Length
    mismatch is folded into the accumulator (lengths are public information).
    The number of memory reads depends only on the lengths — never on the
    content — so the timing does not reveal *where* two secrets differ.
    """
    left = _as_bytes(a, "a")
    right = _as_bytes(b, "b")
    la, lb = len(left), len(right)
    n = la if la >= lb else lb
    acc = la ^ lb
    for i in range(n):  # trip count fixed by public lengths alone
        av = left[i] if i < la else 0
        bv = right[i] if i < lb else 0
        acc |= av ^ bv
    return acc == 0


def ct_verify_mac(tag, expected) -> bool:
    """Constant-time MAC verification (bytes-like inputs)."""
    return ct_compare(tag, expected)


def ct_verify_mac_hex(tag_hex: str, expected_hex: str) -> bool:
    """Constant-time MAC verification over hex strings.

    Malformed hex never verifies (returns ``False``).
    """
    try:
        tag = bytes.fromhex(tag_hex)
        expected = bytes.fromhex(expected_hex)
    except (ValueError, TypeError):
        return False
    return ct_compare(tag, expected)


def ct_select(cond, a, b):
    """Constant-time select: return *a* if *cond* else *b*.

    The condition is turned into an all-ones/all-zeros mask with arithmetic
    (no ``if`` on the operand values):

    * integers: ``(a & mask) | (b & ~mask)`` — works for any Python ints,
      negative or not;
    * bytes-like: the same trick per byte; operands must have equal length
      (lengths are public, mismatch is a programming error).

    Type dispatch branches only on the *type* of the operands, which is
    public. As with everything CPython, this is "constant-time at the
    bytecode level", not a hardware guarantee — see README.
    """
    mask = -(int(cond != 0))  # 0 or -1, no branch on a/b
    a_int, b_int = isinstance(a, int), isinstance(b, int)
    if a_int and b_int:
        return (a & mask) | (b & ~mask)
    if a_int != b_int:
        raise TypeError(
            "ct_select operands must both be ints or both bytes-like"
        )
    left = _as_bytes(a, "a")
    right = _as_bytes(b, "b")
    if len(left) != len(right):
        raise ValueError(
            "ct_select byte operands must have equal length "
            f"({len(left)} != {len(right)})"
        )
    m = mask & 0xFF  # 0x00 or 0xFF
    not_m = (~mask) & 0xFF
    return bytes((x & m) | (y & not_m) for x, y in zip(left, right))


def naive_compare(a, b) -> bool:
    """Early-exit equality — LEAKS timing. Baseline for the demo only.

    Stops at the first differing element, so its runtime reveals how long a
    matching prefix was: exactly the signal ``ct_compare`` removes.
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False
    return True
