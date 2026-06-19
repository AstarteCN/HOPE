import unittest

from tools.stage3.profile_stage3_components import profile_detail_components, profile_mode_settings


class ProfileStage3ComponentsTests(unittest.TestCase):
    def test_profile_modes_match_original_and_command_only_training_modes(self):
        original = profile_mode_settings("original")
        command_only = profile_mode_settings("command-only")

        self.assertTrue(original["verbose"])
        self.assertIsNone(original["render_mode"])
        self.assertEqual(original["expected_effective_render_mode"], "human")

        self.assertFalse(command_only["verbose"])
        self.assertEqual(command_only["render_mode"], "rgb_array")
        self.assertEqual(command_only["expected_effective_render_mode"], "rgb_array")

    def test_profile_mode_settings_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            profile_mode_settings("unknown")

    def test_env_step_detail_components_cover_render_sim_status_and_rs(self):
        components = profile_detail_components("env-step")

        self.assertIn("env.render.total", components)
        self.assertIn("env.render.img_capture", components)
        self.assertIn("env.render.lidar_total", components)
        self.assertIn("env.render.action_mask", components)
        self.assertIn("env.sim.vehicle_step", components)
        self.assertIn("env.status.detect_collision", components)
        self.assertIn("env.rs.find_path", components)

    def test_profile_detail_components_rejects_unknown_detail(self):
        with self.assertRaises(ValueError):
            profile_detail_components("unknown")


if __name__ == "__main__":
    unittest.main()
