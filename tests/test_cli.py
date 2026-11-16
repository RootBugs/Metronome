"""CLI tests: demo command end-to-end (synthetic rows deterministic)."""

from __future__ import annotations

import contextlib
import io
import unittest

import cli


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class DemoCliTests(unittest.TestCase):

    def test_text_secret_demo_succeeds(self):
        code, out, _ = _run(
            ["demo", "--text", "CorrectHorseBattery9",
             "--samples", "8", "--positions", "3"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("synthetic-naive", out)
        self.assertIn("synthetic-ct", out)
        self.assertIn("real-naive", out)
        self.assertIn("real-ct", out)

    def test_synthetic_naive_row_reports_match(self):
        code, out, _ = _run(
            ["demo", "--text", "CorrectHorseBattery9",
             "--samples", "8", "--positions", "3"]
        )
        self.assertEqual(code, 0)
        naive_line = next(
            line for line in out.splitlines()
            if line.startswith("synthetic-naive")
        )
        self.assertIn("MATCH", naive_line)

    def test_synthetic_ct_row_reports_no_signal(self):
        code, out, _ = _run(
            ["demo", "--text", "CorrectHorseBattery9",
             "--samples", "8", "--positions", "3"]
        )
        self.assertEqual(code, 0)
        ct_line = next(
            line for line in out.splitlines()
            if line.startswith("synthetic-ct")
        )
        self.assertIn("NO SIGNAL", ct_line)

    def test_report_only_rows_are_labelled(self):
        code, out, _ = _run(
            ["demo", "--text", "CorrectHorseBattery9",
             "--samples", "4", "--positions", "2"]
        )
        self.assertEqual(code, 0)
        self.assertIn("(report only)", out)
        self.assertIn("deterministic", out)

    def test_random_secret_with_seed_runs(self):
        code, out, _ = _run(
            ["demo", "--bytes", "32", "--seed", "1234",
             "--samples", "4", "--positions", "2", "--candidates", "16"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("32 bytes", out)

    def test_too_few_samples_rejected(self):
        code, _, err = _run(["demo", "--samples", "1"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_zero_candidates_rejected(self):
        code, _, err = _run(["demo", "--candidates", "1"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_one_byte_secret_rejected(self):
        code, _, err = _run(["demo", "--text", "x"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_missing_command_exits(self):
        with self.assertRaises(SystemExit):
            cli.main([])


if __name__ == "__main__":
    unittest.main()
