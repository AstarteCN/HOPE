import unittest

from tools.stage3.profile_stage3_components import profile_mode_settings


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


if __name__ == "__main__":
    unittest.main()
