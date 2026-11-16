"""Attack harness: deterministic synthetic models, injected clocks, reports.

Real wall-clock models are exercised only for *structure* — no assertion
ever depends on recovered bytes from a real measurement, so this suite is
non-flaky by construction.
"""

from __future__ import annotations

import random
import unittest

from harness import (
    DEFAULT_LEAK,
    T_THRESHOLD,
    RealConstantTimeModel,
    RealNaiveModel,
    SyntheticConstantTimeModel,
    SyntheticNaiveModel,
    attack_position,
    build_guess,
    common_prefix_len,
    measure,
    run_attack,
)

SECRET = b"CorrectHorseBattery9"


class MeasureTests(unittest.TestCase):

    def test_injected_clock_is_deterministic(self):
        ticks = iter([0.0, 0.001, 0.002, 0.005])
        durations = measure(lambda: None, 2, clock=lambda: next(ticks))
        self.assertEqual(durations, [0.001, 0.003])

    def test_samples_must_be_positive(self):
        with self.assertRaises(ValueError):
            measure(lambda: None, 0)

    def test_fn_is_called_once_per_sample(self):
        calls = []
        measure(lambda: calls.append(1), 5, clock=lambda: 0.0)
        self.assertEqual(len(calls), 5)


class GuessTests(unittest.TestCase):

    def test_build_guess_replaces_one_byte(self):
        guess = build_guess(b"abcdef", 2, ord("X"))
        self.assertEqual(guess, b"abXdef")

    def test_correct_candidate_reproduces_secret(self):
        for pos in range(len(SECRET)):
            self.assertEqual(
                build_guess(SECRET, pos, SECRET[pos]), SECRET
            )

    def test_position_out_of_range(self):
        with self.assertRaises(ValueError):
            build_guess(b"abc", 3, 0)
        with self.assertRaises(ValueError):
            build_guess(b"abc", -1, 0)

    def test_candidate_range_checked(self):
        with self.assertRaises(ValueError):
            build_guess(b"abc", 0, 256)
        with self.assertRaises(ValueError):
            build_guess(b"abc", 0, -1)

    def test_common_prefix_len(self):
        self.assertEqual(common_prefix_len(b"abcdef", b"abcxyz"), 3)
        self.assertEqual(common_prefix_len(b"", b"abc"), 0)
        self.assertEqual(common_prefix_len(b"same", b"same"), 4)


class SyntheticNaiveTests(unittest.TestCase):

    def test_full_secret_recovered_deterministically(self):
        report = run_attack(SyntheticNaiveModel(SECRET), SECRET, samples=12)
        self.assertTrue(report["match"])
        self.assertEqual(bytes(report["recovered"]), SECRET)

    def test_same_seed_gives_identical_result(self):
        a = run_attack(SyntheticNaiveModel(SECRET), SECRET, samples=12)
        b = run_attack(SyntheticNaiveModel(SECRET), SECRET, samples=12)
        self.assertEqual(a["recovered"], b["recovered"])

    def test_recovery_survives_repeated_runs(self):
        # Run the seeded attack 5 times: never a flake.
        for _ in range(5):
            report = run_attack(SyntheticNaiveModel(SECRET), SECRET,
                                samples=12)
            self.assertTrue(report["match"])

    def test_report_structure(self):
        report = run_attack(SyntheticNaiveModel(SECRET), SECRET,
                            samples=8, positions=(0, 1))
        self.assertEqual(report["attacked"], (0, 1))
        self.assertEqual(len(report["recovered"]), 2)
        self.assertEqual(len(report["positions"]), 2)
        info = report["positions"][0]
        for key in ("best", "second", "means", "gap", "t", "df",
                    "significant"):
            self.assertIn(key, info)
        self.assertTrue(info["significant"])
        self.assertGreater(info["gap"], 0)
        self.assertEqual(report["samples"], 8)

    def test_larger_noise_still_recovers_with_more_samples(self):
        # 10x the default noise, 40 samples: still deterministic signal.
        model = SyntheticNaiveModel(SECRET, noise=2e-5, seed=7)
        report = run_attack(model, SECRET, samples=40, positions=(0, 1))
        self.assertTrue(report["match"])

    def test_min_gap_gate_blocks_marginal_gap(self):
        # A tiny leak with a large min_gap must be rejected even if t is big.
        model = SyntheticNaiveModel(SECRET, leak=1e-9, noise=1e-12, seed=3)
        chosen, info = attack_position(
            model, SECRET, 0, range(256), samples=10,
            min_gap=DEFAULT_LEAK,
        )
        self.assertIsNone(chosen)
        self.assertFalse(info["significant"])


class SyntheticCtTests(unittest.TestCase):

    def test_no_signal_ever_reported(self):
        report = run_attack(SyntheticConstantTimeModel(SECRET), SECRET,
                            samples=16)
        self.assertFalse(report["match"])
        self.assertTrue(all(b is None for b in report["recovered"]))

    def test_no_signal_is_deterministic_across_runs(self):
        for _ in range(5):
            report = run_attack(SyntheticConstantTimeModel(SECRET), SECRET,
                                samples=16)
            self.assertTrue(
                all(b is None for b in report["recovered"])
            )

    def test_gap_below_threshold_even_for_correct_byte(self):
        model = SyntheticConstantTimeModel(SECRET, seed=99)
        chosen, info = attack_position(
            model, SECRET, 0, range(256), samples=16
        )
        self.assertIsNone(chosen)
        self.assertFalse(info["significant"])
        self.assertLessEqual(info["gap"], 0.5 * 1e-5 + 1e-12)

    def test_all_candidates_share_the_same_distribution_shape(self):
        # Means differ only by noise: spread is much smaller than the
        # synthetic leak unit.
        model = SyntheticConstantTimeModel(SECRET, seed=5)
        _, info = attack_position(model, SECRET, 3, range(256), samples=8)
        spread = max(info["means"].values()) - min(info["means"].values())
        self.assertLess(spread, DEFAULT_LEAK)


class ThresholdAndValidationTests(unittest.TestCase):

    def test_default_threshold_is_six(self):
        self.assertEqual(T_THRESHOLD, 6.0)

    def test_attack_needs_two_candidates(self):
        with self.assertRaises(ValueError):
            attack_position(SyntheticNaiveModel(SECRET), SECRET, 0, [65],
                            samples=4)

    def test_empty_secret_does_not_match(self):
        report = run_attack(SyntheticNaiveModel(b""), b"", samples=4)
        self.assertFalse(report["match"])
        self.assertEqual(report["recovered"], [])


class RealModelStructureTests(unittest.TestCase):
    """Real measurements: structure only, never recovery assertions."""

    def test_real_naive_returns_positive_durations(self):
        model = RealNaiveModel(SECRET)
        for _ in range(3):
            self.assertGreaterEqual(model.time(SECRET), 0.0)

    def test_real_ct_returns_positive_durations(self):
        model = RealConstantTimeModel(SECRET)
        for _ in range(3):
            self.assertGreaterEqual(model.time(SECRET), 0.0)

    def test_real_attack_report_structure(self):
        report = run_attack(RealNaiveModel(SECRET), SECRET,
                            samples=4, positions=(0, 1))
        self.assertEqual(len(report["recovered"]), 2)
        self.assertEqual(len(report["positions"]), 2)
        info = report["positions"][0]
        self.assertIsInstance(info["best"], int)
        self.assertIsInstance(info["t"], float)
        self.assertIsInstance(info["significant"], bool)
        self.assertTrue(info["means"])
        # match may be True or False depending on the machine — only the
        # shape of the report is asserted here.

    def test_real_models_share_interface_with_synthetic(self):
        rng = random.Random(42)
        secret = bytes(rng.randrange(256) for _ in range(16))
        for cls in (RealNaiveModel, RealConstantTimeModel,
                    SyntheticNaiveModel, SyntheticConstantTimeModel):
            model = cls(secret)
            duration = model.time(secret)
            self.assertIsInstance(duration, float)
            self.assertGreaterEqual(duration, 0.0)


if __name__ == "__main__":
    unittest.main()
