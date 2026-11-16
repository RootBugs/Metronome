"""Demo CLI: watch the naive comparison leak a byte — and ct_compare not.

Usage::

    python cli.py demo [--text STR | --bytes N] [--samples N]
                       [--positions N] [--candidates N]

Runs four attacks on the same secret (synthetic-naive, synthetic-ct,
real-naive, real-ct) and prints a verdict table. The synthetic rows are
fully deterministic; the real rows are report-only measurements.
"""

from __future__ import annotations

import argparse
import random
import sys

from harness import (
    RealConstantTimeModel,
    RealNaiveModel,
    SyntheticConstantTimeModel,
    SyntheticNaiveModel,
    run_attack,
)

__all__ = ["main"]


def _fmt_byte(value: int | None) -> str:
    if value is None:
        return "--"
    glyph = chr(value) if 32 <= value < 127 else "."
    return f"0x{value:02x} '{glyph}'"


def _verdict(report: dict, suffix: str = "") -> str:
    if report["match"]:
        return "MATCH" + suffix
    if all(b is None for b in report["recovered"]):
        return "NO SIGNAL" + suffix
    return "PARTIAL" + suffix


def _print_row(label: str, report: dict, truth: int | None, suffix: str) -> None:
    got = report["recovered"][0] if report["recovered"] else None
    print(
        f"{label:<18} {_fmt_byte(got):<16} "
        f"{_verdict(report, suffix):<28} truth={_fmt_byte(truth)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="constant-time",
        description="Timing-safe primitives and the attack demo.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="run the naive-vs-ct attack demo")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--text", default=None, metavar="STR",
                       help="secret as text (default: random bytes)")
    group.add_argument("--bytes", dest="nbytes", type=int, default=256,
                       metavar="N", help="random secret length (default 256)")
    p.add_argument("--samples", type=int, default=8, help="samples per candidate")
    p.add_argument("--positions", type=int, default=4,
                   help="how many leading bytes to attack")
    p.add_argument("--candidates", type=int, default=256,
                   help="candidate bytes to try per position")
    p.add_argument("--seed", type=int, default=0xDEAD,
                   help="seed for the random secret")
    args = parser.parse_args(argv)

    if args.samples < 2 or args.positions < 1 or args.candidates < 2:
        print("error: need --samples >= 2, --positions >= 1, "
              "--candidates >= 2", file=sys.stderr)
        return 2

    if args.text is not None:
        secret = args.text.encode("utf-8")
    else:
        rng = random.Random(args.seed)
        secret = bytes(rng.randrange(256) for _ in range(args.nbytes))
    if len(secret) < 2:
        print("error: secret must be at least 2 bytes", file=sys.stderr)
        return 2

    positions = tuple(range(min(args.positions, len(secret))))
    candidates = tuple(range(min(args.candidates, 256)))

    print(f"secret ({len(secret)} bytes): {_fmt_byte(secret[0])} "
          f"(attacking positions {list(positions)})")
    print(f"{'model':<18} {'recovered':<16} {'verdict':<28} truth")
    print("-" * 78)

    synth_naive = SyntheticNaiveModel(secret)
    synth_ct = SyntheticConstantTimeModel(secret)
    real_naive = RealNaiveModel(secret)
    real_ct = RealConstantTimeModel(secret)

    rows = [
        ("synthetic-naive", synth_naive, ""),
        ("synthetic-ct", synth_ct, ""),
        ("real-naive", real_naive, " (report only)"),
        ("real-ct", real_ct, " (report only)"),
    ]
    for label, model, suffix in rows:
        report = run_attack(
            model, secret, candidates=candidates,
            samples=args.samples, positions=positions,
        )
        _print_row(label, report, secret[positions[0]], suffix)

    print()
    print("synthetic rows are deterministic (seeded model); real rows are")
    print("wall-clock measurements and reported without assertions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
