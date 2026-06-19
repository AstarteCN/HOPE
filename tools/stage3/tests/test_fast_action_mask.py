from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.action_mask import ActionMask  # noqa: E402


class FastActionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = ActionMask()
        cls.fast = ActionMask(fast_get_steps=True)

    def assert_fast_matches_original(self, lidar: np.ndarray) -> None:
        expected = self.original.get_steps(lidar.copy())
        actual = self.fast.get_steps(lidar.copy())
        np.testing.assert_array_equal(actual, expected)

    def test_default_fast_get_steps_is_disabled(self) -> None:
        self.assertFalse(self.original.fast_get_steps)

    def test_fast_get_steps_matches_original_for_edge_lidar_samples(self) -> None:
        for value in (-5.0, 0.0, 0.01, 1.0, 9.99, 10.0, 25.0):
            with self.subTest(value=value):
                lidar = np.full((self.original.lidar_num,), value, dtype=float)
                self.assert_fast_matches_original(lidar)

    def test_fast_get_steps_matches_original_for_seeded_random_lidar_samples(self) -> None:
        rng = np.random.default_rng(20260619)
        for index in range(32):
            with self.subTest(index=index):
                lidar = rng.uniform(-2.0, 14.0, size=(self.original.lidar_num,))
                self.assert_fast_matches_original(lidar)


if __name__ == "__main__":
    unittest.main()
