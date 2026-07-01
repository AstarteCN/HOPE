from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs import VALID_STEER, WHEEL_BASE  # noqa: E402
from env import reeds_shepp  # noqa: E402


class ReedsSheppSamplingTests(unittest.TestCase):
    def test_generate_local_course_handles_fractional_sampling_boundary(self) -> None:
        lengths = [
            4.957681383312066,
            -2.0016830858790184,
            3.465908332602947,
            -3.5361659649759782,
        ]
        modes = ["L", "S", "R", "R"]
        max_curvature = math.tan(VALID_STEER[-1]) / WHEEL_BASE
        step_size = 0.1 * max_curvature
        total_length = sum(abs(length) for length in lengths)

        try:
            x, y, yaw, directions = reeds_shepp.generate_local_course(
                total_length,
                lengths,
                modes,
                max_curvature,
                step_size,
            )
        except IndexError as exc:
            self.fail(f"generate_local_course underallocated sampling buffers: {exc}")

        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(yaw))
        self.assertEqual(len(x), len(directions))
        self.assertGreater(len(x), 0)


if __name__ == "__main__":
    unittest.main()
