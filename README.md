# constant-time — Timing Attacks & the Code That Stops Them

A working timing-attack lab: `ct_compare` and friends implement
**constant-time comparison** in pure Python, and a full attack harness
*proves* why you need them — the naive early-exit comparison leaks a
secret byte-by-byte, the constant-time one doesn't.

## Why it's special

- **Instrumented proof, not hand-waving**: a counting sequence shows
  `naive_compare` stops after 1 read when the first byte differs, while
  `ct_compare` reads all 32 positions regardless of *where* the
  difference is — the leak is measured directly, not inferred.
- **Deterministic attack demo**: seeded synthetic timing models recover
  the entire secret byte-by-byte (Welch t-test gate, t > 6), while the
  constant-time model yields a permanent `NO SIGNAL` — the same suite run
  5× never flakes.
- **Real wall-clock rows are report-only**: the demo also measures the
  actual functions with `perf_counter`, but no test ever asserts on
  recovery from real timing (machine-dependent by nature).
- **Statistics from scratch**: `welch_t_test` implements the
  unequal-variance t-test with exact degenerate-case definitions
  (zero variance → signed infinity).

## Architecture

| Module | Responsibility |
|---|---|
| `ct.py` | `ct_compare`, `ct_verify_mac` (+hex), branchless `ct_select`, leaky `naive_compare` |
| `stats.py` | `mean`, `stddev`, Welch's t-test (`WelchResult`) |
| `harness.py` | `measure` (injectable clock), synthetic/real timing models, `attack_position`, `run_attack` |
| `cli.py` | `demo` command: four-model verdict table |

## Runnable example

```console
$ python cli.py demo --bytes 512 --samples 64 --positions 3 --seed 42
secret (512 bytes): 0x39 '9' (attacking positions [0, 1, 2])
model              recovered        verdict                      truth
------------------------------------------------------------------------------
synthetic-naive    0x39 '9'         MATCH                        truth=0x39 '9'
synthetic-ct       --               NO SIGNAL                    truth=0x39 '9'
real-naive         0x39 '9'         MATCH (report only)          truth=0x39 '9'
real-ct            --               NO SIGNAL (report only)      truth=0x39 '9'

synthetic rows are deterministic (seeded model); real rows are
wall-clock measurements and reported without assertions.
```

Smaller, fully deterministic run (synthetic rows are identical on every
machine; real rows vary):

```console
$ python cli.py demo --text "CorrectHorseBattery9" --samples 8 --positions 3
secret (20 bytes): 0x43 'C' (attacking positions [0, 1, 2])
model              recovered        verdict                      truth
------------------------------------------------------------------------------
synthetic-naive    0x43 'C'         MATCH                        truth=0x43 'C'
synthetic-ct       --               NO SIGNAL                    truth=0x43 'C'
real-naive         --               NO SIGNAL (report only)      truth=0x43 'C'
real-ct            --               NO SIGNAL (report only)      truth=0x43 'C'

synthetic rows are deterministic (seeded model); real rows are
wall-clock measurements and reported without assertions.
```

Programmatic use:

```console
$ python -c "
from harness import SyntheticNaiveModel, SyntheticConstantTimeModel, run_attack
s = b'CorrectHorseBattery9'
print('naive match:', run_attack(SyntheticNaiveModel(s), s, samples=12)['match'])
print('ct match:   ', run_attack(SyntheticConstantTimeModel(s), s, samples=12)['match'])
"
naive match: True
ct match:    False
```

## Honest limitations (documented, not hidden)

- CPython guarantees constant-time behavior only at the *bytecode* level;
  the interpreter and hardware can still introduce timing variance. The
  point is removing the *algorithmic* early-exit leak.
- The attack uses **known context**: guesses are
  `secret[:pos] + candidate + secret[pos+1:]`, i.e. brute-forcing one
  unknown byte inside an otherwise known-format token — the realistic
  scenario for MAC/token comparison.
- Real-measurement rows are printed for education; only synthetic rows
  carry assertions.

## Tests

```console
python -m unittest discover -s tests -v
```

70 tests: read-count instrumentation (naive stops early, ct reads fixed
length), exact Welch values (t = -1.8973665961010275, df =
5.882352941176471), zero-variance infinities, full-secret recovery
determinism across repeated seeded runs, `NO SIGNAL` stability for the
constant-time model, gap/t significance gates, injected-clock `measure`,
and structure-only assertions for real timing models.
