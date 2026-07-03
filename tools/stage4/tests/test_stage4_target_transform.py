import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.stage4.stage4_target_transform import (
    TARGET_MODE_AUTO,
    TARGET_MODE_CORRECTED_COSSIN,
    TARGET_MODE_LEGACY_COSCOS,
    apply_target_mode,
    build_stage4_observation_func,
    resolve_checkpoint_target_mode,
    validate_target_mode,
)


class Stage4TargetTransformTest(unittest.TestCase):
    def test_validate_target_mode_accepts_known_modes(self):
        self.assertEqual(validate_target_mode(TARGET_MODE_LEGACY_COSCOS), TARGET_MODE_LEGACY_COSCOS)
        self.assertEqual(validate_target_mode(TARGET_MODE_CORRECTED_COSSIN), TARGET_MODE_CORRECTED_COSSIN)

    def test_validate_target_mode_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "target_mode"):
            validate_target_mode("unknown")

    def test_legacy_mode_keeps_target_unchanged(self):
        target = np.array([4.0, 0.1, 0.2, 0.3, 0.3], dtype=np.float64)

        result = apply_target_mode(target, TARGET_MODE_LEGACY_COSCOS)

        np.testing.assert_allclose(result, target)
        self.assertIsNot(result, target)

    def test_corrected_mode_replaces_last_dim_with_signed_sine(self):
        angle = -0.7
        target = np.array(
            [4.0, math.cos(0.2), math.sin(0.2), math.cos(angle), math.cos(angle)],
            dtype=np.float64,
        )

        result = apply_target_mode(target, TARGET_MODE_CORRECTED_COSSIN, rel_dest_heading=angle)

        self.assertAlmostEqual(result[3], math.cos(angle))
        self.assertAlmostEqual(result[4], math.sin(angle))

    def test_corrected_mode_requires_relative_heading(self):
        target = np.array([4.0, 1.0, 0.0, 0.5, 0.5], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "rel_dest_heading"):
            apply_target_mode(target, TARGET_MODE_CORRECTED_COSSIN)

    def test_observation_func_transposes_ogm_and_updates_target(self):
        class _Point:
            x = 1.0
            y = 2.0

        class _State:
            loc = _Point()
            heading = 0.5

        class _Dest:
            loc = _Point()
            heading = -0.25

        class _Map:
            dest = _Dest()

        class _Env:
            vehicle = type("Vehicle", (), {"state": _State()})()
            map = _Map()

        obs = {
            "img": None,
            "ogm": np.zeros((64, 64, 2), dtype=np.float32),
            "target": np.array(
                [1.0, 1.0, 0.0, math.cos(-0.75), math.cos(-0.75)],
                dtype=np.float64,
            ),
        }

        transformed = build_stage4_observation_func(_Env(), TARGET_MODE_CORRECTED_COSSIN)(obs)

        self.assertEqual(transformed["ogm"].shape, (2, 64, 64))
        self.assertAlmostEqual(transformed["target"][3], math.cos(-0.75))
        self.assertAlmostEqual(transformed["target"][4], math.sin(-0.75))

    def test_auto_checkpoint_target_mode_reads_matching_stage4_state(self):
        with tempfile.TemporaryDirectory(prefix="stage4 target mode ") as temp_dir:
            temp_root = Path(temp_dir)
            checkpoint_path = temp_root / "SAC_19999.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")
            state_path = temp_root / "stage4_state_19999.pt"
            torch.save({"target_mode": TARGET_MODE_CORRECTED_COSSIN}, state_path)

            mode, source = resolve_checkpoint_target_mode(checkpoint_path, TARGET_MODE_AUTO)

        self.assertEqual(mode, TARGET_MODE_CORRECTED_COSSIN)
        self.assertEqual(source["type"], "stage4_state")
        self.assertEqual(Path(source["path"]).name, "stage4_state_19999.pt")

    def test_auto_checkpoint_target_mode_defaults_to_legacy_without_state(self):
        with tempfile.TemporaryDirectory(prefix="stage4 target mode ") as temp_dir:
            checkpoint_path = Path(temp_dir) / "SAC_best.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")

            mode, source = resolve_checkpoint_target_mode(checkpoint_path, TARGET_MODE_AUTO)

        self.assertEqual(mode, TARGET_MODE_LEGACY_COSCOS)
        self.assertEqual(source["type"], "legacy_default")


if __name__ == "__main__":
    unittest.main()
