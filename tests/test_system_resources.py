from __future__ import annotations

import subprocess
import unittest
from unittest.mock import Mock

from video_editor.system_resources import (
    MEMORY_PRESSURE_COMMAND,
    MemorySnapshot,
    assess_memory_pressure,
    collect_memory_snapshot,
    parse_memory_pressure,
)


NORMAL_OUTPUT = """The system has 17179869184 (1048576 pages with a page size of 16384).

Stats:
Pages free: 6309
Swap I/O:
Swapins: 1,234
Swapouts: 56
System-wide memory free percentage: 72%
"""


LOW_OUTPUT = """Swap I/O:
Swapins: 20
Swapouts: 56
System-wide memory free percentage: 8%
"""


class SystemResourcesTests(unittest.TestCase):
    def test_parser_reads_free_percentage_and_swap_counters(self):
        snapshot = parse_memory_pressure(NORMAL_OUTPUT, observed_at="fixed")

        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.free_percent, 72.0)
        self.assertEqual(snapshot.swapins, 1234)
        self.assertEqual(snapshot.swapouts, 56)
        self.assertEqual(snapshot.observed_at, "fixed")
        self.assertIsNone(snapshot.error)

    def test_low_memory_and_rising_swapouts_block_start(self):
        earlier = parse_memory_pressure(NORMAL_OUTPUT)
        current = parse_memory_pressure(LOW_OUTPUT)

        assessment = assess_memory_pressure((earlier, current))

        self.assertEqual(assessment.status, "pressure")
        self.assertFalse(assessment.allow_start)
        self.assertEqual(assessment.swapout_delta, 0)
        self.assertTrue(any("below" in reason for reason in assessment.reasons))

    def test_swapout_increase_blocks_even_when_free_memory_is_normal(self):
        earlier = MemorySnapshot(50.0, 10, 100, "a")
        current = MemorySnapshot(48.0, 12, 101, "b")

        assessment = assess_memory_pressure((earlier, current))

        self.assertEqual(assessment.status, "pressure")
        self.assertFalse(assessment.allow_start)
        self.assertEqual(assessment.swapout_delta, 1)

    def test_two_healthy_samples_allow_start(self):
        samples = (
            MemorySnapshot(20.0, 10, 100, "a"),
            MemorySnapshot(18.0, 11, 100, "b"),
        )

        assessment = assess_memory_pressure(samples, low_free_percent=10)

        self.assertEqual(assessment.status, "ready")
        self.assertTrue(assessment.allow_start)
        self.assertEqual(assessment.swapout_delta, 0)

    def test_missing_critical_reading_is_unknown_and_fails_closed(self):
        complete = parse_memory_pressure(NORMAL_OUTPUT)
        unavailable = parse_memory_pressure("Swapouts: 60\n")

        assessment = assess_memory_pressure((complete, unavailable))

        self.assertEqual(assessment.status, "unknown")
        self.assertFalse(assessment.allow_start)
        self.assertIn("free-memory percentage", unavailable.error or "")

    def test_fewer_than_two_samples_is_unknown(self):
        assessment = assess_memory_pressure((parse_memory_pressure(NORMAL_OUTPUT),))

        self.assertEqual(assessment.status, "unknown")
        self.assertFalse(assessment.allow_start)

    def test_recovering_from_one_low_sample_requires_another_pair(self):
        earlier = parse_memory_pressure(LOW_OUTPUT)
        current = parse_memory_pressure(NORMAL_OUTPUT)

        assessment = assess_memory_pressure((earlier, current))

        self.assertEqual(assessment.status, "unknown")
        self.assertFalse(assessment.allow_start)
        self.assertIn("another pair", assessment.reasons[0])

    def test_probe_calls_read_only_command_and_parses_stdout(self):
        run = Mock(return_value=subprocess.CompletedProcess([], 0, NORMAL_OUTPUT, ""))

        snapshot = collect_memory_snapshot(run=run, system_name="Darwin", observed_at="fixed")

        run.assert_called_once_with(
            [MEMORY_PRESSURE_COMMAND],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.observed_at, "fixed")

    def test_probe_returns_unavailable_on_unsupported_platform(self):
        run = Mock()

        snapshot = collect_memory_snapshot(run=run, system_name="Linux", observed_at="fixed")

        run.assert_not_called()
        self.assertFalse(snapshot.complete)
        self.assertIn("only on macOS", snapshot.error or "")

    def test_counter_reset_is_unknown(self):
        earlier = MemorySnapshot(40.0, 5, 100, "a")
        current = MemorySnapshot(40.0, 0, 0, "b")

        assessment = assess_memory_pressure((earlier, current))

        self.assertEqual(assessment.status, "unknown")
        self.assertFalse(assessment.allow_start)
        self.assertIn("decreased", assessment.reasons[0])


if __name__ == "__main__":
    unittest.main()
