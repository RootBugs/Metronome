"""Timing-safe primitives: correctness, read-count instrumentation, MAC checks."""

from __future__ import annotations

import unittest

from ct import (
    ct_compare,
    ct_select,
    ct_verify_mac,
    ct_verify_mac_hex,
    naive_compare,
)


class CountingSeq:
    """Sequence of ints that counts how many elements are pulled from it.

    Both ``ct_compare`` (via ``bytes()`` normalization) and
    ``naive_compare`` (via ``zip``) consume this sequence, so the counter
    reveals exactly how many positions each implementation touches before
    it stops.
    """

    def __init__(self, values) -> None:
        self.values = list(values)
        self.reads = 0

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for v in self.values:
            self.reads += 1
            yield v


class CtCompareTests(unittest.TestCase):

    def test_equal_bytes_true(self):
        self.assertTrue(ct_compare(b"secret", b"secret"))

    def test_different_bytes_false(self):
        self.assertFalse(ct_compare(b"secret", b"secreu"))

    def test_length_mismatch_false(self):
        self.assertFalse(ct_compare(b"abc", b"abcd"))

    def test_both_empty_true(self):
        self.assertTrue(ct_compare(b"", b""))

    def test_str_inputs_compared_as_utf8(self):
        self.assertTrue(ct_compare("héllo", "héllo"))
        self.assertFalse(ct_compare("héllo", "hello"))

    def test_bytearray_and_memoryview_accepted(self):
        self.assertTrue(ct_compare(bytearray(b"abc"), memoryview(b"abc")))

    def test_rejects_unconvertible_type(self):
        with self.assertRaises(TypeError):
            ct_compare(object(), b"x")

    def test_read_count_is_fixed_regardless_of_mismatch_position(self):
        """The core property: ct_compare reads every position always."""
        base = list(range(32))
        early = CountingSeq(base)
        late = CountingSeq(base)
        late.values[31] = 255  # differs only at the last position
        ct_compare(early, CountingSeq(base))
        ct_compare(late, CountingSeq(base))
        # Both comparisons normalized two 32-element inputs -> 32 reads
        # each side, no matter where the difference sits.
        self.assertEqual(early.reads, 32)
        self.assertEqual(late.reads, 32)

    def test_naive_stops_early_while_ct_reads_everything(self):
        a_vals = list(range(32))
        b_vals = list(range(32))
        b_vals[0] = 99  # mismatch at position 0

        naive_a = CountingSeq(a_vals)
        naive_b = CountingSeq(b_vals)
        self.assertFalse(naive_compare(naive_a, naive_b))
        self.assertEqual(naive_a.reads, 1)  # stopped at the first byte

        ct_a = CountingSeq(a_vals)
        ct_b = CountingSeq(b_vals)
        self.assertFalse(ct_compare(ct_a, ct_b))
        self.assertEqual(ct_a.reads, 32)  # still read everything

    def test_naive_mismatch_later_reads_further(self):
        a_vals = list(range(32))
        b_vals = list(range(32))
        b_vals[7] = 99
        seq = CountingSeq(a_vals)
        self.assertFalse(naive_compare(seq, CountingSeq(b_vals)))
        self.assertEqual(seq.reads, 8)  # positions 0..7


class NaiveCompareTests(unittest.TestCase):

    def test_equal_and_unequal(self):
        self.assertTrue(naive_compare(b"abc", b"abc"))
        self.assertFalse(naive_compare(b"abc", b"abd"))

    def test_length_mismatch_false(self):
        self.assertFalse(naive_compare(b"abc", b"ab"))

    def test_rejects_non_sized_input(self):
        with self.assertRaises(TypeError):
            naive_compare(None, None)


class MacVerifyTests(unittest.TestCase):

    def test_verify_mac_bytes_true(self):
        tag = bytes(range(32))
        self.assertTrue(ct_verify_mac(tag, bytes(range(32))))

    def test_verify_mac_bytes_false(self):
        tag = bytes(range(32))
        wrong = bytearray(range(32))
        wrong[31] ^= 0x01
        self.assertFalse(ct_verify_mac(tag, bytes(wrong)))

    def test_verify_mac_hex_true(self):
        self.assertTrue(ct_verify_mac_hex("deadbeef", "deadbeef"))

    def test_verify_mac_hex_false_on_last_byte(self):
        self.assertFalse(ct_verify_mac_hex("deadbeef", "deadbeed"))

    def test_verify_mac_hex_rejects_malformed_hex(self):
        self.assertFalse(ct_verify_mac_hex("nothex!!", "deadbeef"))
        self.assertFalse(ct_verify_mac_hex("dead", None))


class CtSelectTests(unittest.TestCase):

    def test_select_ints(self):
        self.assertEqual(ct_select(True, 1, 2), 1)
        self.assertEqual(ct_select(False, 1, 2), 2)

    def test_select_ints_with_negative_values(self):
        self.assertEqual(ct_select(True, -5, 7), -5)
        self.assertEqual(ct_select(False, -5, 7), 7)

    def test_select_bytes(self):
        self.assertEqual(ct_select(True, b"aaaa", b"bbbb"), b"aaaa")
        self.assertEqual(ct_select(False, b"aaaa", b"bbbb"), b"bbbb")

    def test_select_mixed_types_rejected(self):
        with self.assertRaises(TypeError):
            ct_select(True, 1, b"b")

    def test_select_unequal_byte_lengths_rejected(self):
        with self.assertRaises(ValueError):
            ct_select(True, b"aa", b"bbb")

    def test_select_on_zero_condition(self):
        self.assertEqual(ct_select(0, b"aaaa", b"bbbb"), b"bbbb")
        self.assertEqual(ct_select(0, 10, 20), 20)


if __name__ == "__main__":
    unittest.main()
