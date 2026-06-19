import tempfile
import unittest
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tools.stage3.tensorboard_stage3_summary import summarize_event_dir


class TensorboardStage3SummaryWindowTests(unittest.TestCase):
    def test_summary_includes_window_means_and_nonfinite_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer = SummaryWriter(str(run_dir))
            for step in range(600):
                writer.add_scalar("avg_reward", float(step), step)
                writer.add_scalar("critic_loss", float(step) / 10.0, step)
            writer.flush()
            writer.close()

            report = summarize_event_dir(run_dir, min_episodes=500)
            reward = report["scalars"]["avg_reward"]
            self.assertEqual(reward["count"], 600)
            self.assertAlmostEqual(reward["mean_first500"], 249.5)
            self.assertAlmostEqual(reward["mean_last100"], 549.5)
            self.assertAlmostEqual(reward["mean_last500"], 349.5)
            self.assertFalse(reward["has_nonfinite"])
            self.assertFalse(report["has_nonfinite"])

    def test_summary_flags_hard_reject_nonfinite_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer = SummaryWriter(str(run_dir))
            writer.add_scalar("actor_loss", float("nan"), 0)
            writer.add_scalar("avg_reward", 1.0, 0)
            writer.flush()
            writer.close()

            report = summarize_event_dir(run_dir, min_episodes=1)
            actor_loss = report["scalars"]["actor_loss"]
            self.assertTrue(actor_loss["has_nonfinite"])
            self.assertTrue(report["has_nonfinite"])
            self.assertIsNone(actor_loss["first_value"])
            self.assertIsNone(actor_loss["last_value"])
            self.assertIn("actor_loss", report["nonfinite_scalar_tags"])
            self.assertIn("actor_loss", report["hard_reject_nonfinite_scalar_tags"])
            self.assertTrue(report["hard_reject_has_nonfinite"])

    def test_scene_success_nonfinite_does_not_trip_hard_reject_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer = SummaryWriter(str(run_dir))
            writer.add_scalar("success_rate_Extrem", float("nan"), 0)
            writer.add_scalar("actor_loss", 1.0, 0)
            writer.add_scalar("critic_loss", 2.0, 0)
            writer.add_scalar("alpha", 0.1, 0)
            writer.add_scalar("action_std0", -1.0, 0)
            writer.add_scalar("action_std1", -1.5, 0)
            writer.flush()
            writer.close()

            report = summarize_event_dir(run_dir, min_episodes=1)
            self.assertTrue(report["has_nonfinite"])
            self.assertIn("success_rate_Extrem", report["nonfinite_scalar_tags"])
            self.assertEqual(report["hard_reject_nonfinite_scalar_tags"], [])
            self.assertFalse(report["hard_reject_has_nonfinite"])


if __name__ == "__main__":
    unittest.main()
