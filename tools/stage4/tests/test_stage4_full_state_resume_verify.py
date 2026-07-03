from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.stage4.verify_stage4_full_state_resume import (
    compare_resume_summaries,
    run_bounded_resume_simulation,
)


class Stage4FullStateResumeVerifyTests(unittest.TestCase):
    def test_bounded_resume_simulation_matches_direct_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 resume verify ") as temp_dir:
            report = run_bounded_resume_simulation(
                work_dir=Path(temp_dir),
                episode_count=8,
                interrupt_after_episode=3,
                seed=2468,
            )

        comparison = report["comparison"]
        direct = report["direct"]
        resumed = report["interrupted_resumed"]
        restored = report["restored_state"]

        self.assertTrue(comparison["match"], comparison["differences"])
        self.assertEqual(comparison["differences"], [])
        self.assertEqual(direct["total_step_num"], resumed["total_step_num"])
        self.assertEqual(direct["replay_length"], resumed["replay_length"])
        self.assertEqual(direct["scene_record"], resumed["scene_record"])
        self.assertEqual(direct["dlp_case_record"], resumed["dlp_case_record"])
        self.assertEqual(direct["checkpoint_names"], resumed["checkpoint_names"])
        self.assertEqual(direct["tensorboard"]["total_reward_steps"], list(range(8)))
        self.assertEqual(resumed["tensorboard"]["total_reward_steps"], list(range(8)))
        self.assertEqual(restored["global_episode"], 3)
        self.assertEqual(restored["expected_start_episode"], 4)
        self.assertEqual(restored["replay_length_after_load"], report["interrupted_before_resume"]["replay_length"])
        self.assertIn("simulation", report["limitations"].lower())
        self.assertIn("does not prove bitwise real SAC learning trajectory equivalence", report["limitations"])

    def test_comparison_reports_restore_relevant_mismatches(self) -> None:
        base = {
            "total_step_num": 100,
            "replay_length": 4,
            "scene_record": [0, 1],
            "scene_success_record": {"0": [1], "1": [0]},
            "dlp_case_record": [7],
            "dlp_case_success_rate": {"7": [1]},
            "checkpoint_names": ["SAC_3.pt"],
            "state_artifact_names": ["stage4_state_3.pt"],
            "tensorboard": {
                "total_reward_steps": [0, 1],
                "step_num_steps": [0, 1],
            },
            "rng_probe": {
                "numpy": 0.1,
                "torch": 0.2,
                "cuda": None,
            },
        }
        changed = dict(base)
        changed["total_step_num"] = 101
        changed["tensorboard"] = {
            "total_reward_steps": [0],
            "step_num_steps": [0, 1],
        }

        comparison = compare_resume_summaries(base, changed)

        self.assertFalse(comparison["match"])
        self.assertIn("total_step_num", comparison["differences"])
        self.assertIn("tensorboard.total_reward_steps", comparison["differences"])


if __name__ == "__main__":
    unittest.main()
