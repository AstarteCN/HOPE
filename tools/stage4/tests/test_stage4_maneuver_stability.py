from __future__ import annotations

import unittest

import numpy as np

from tools.stage4.stage4_maneuver_stability import (
    APPLY_TO_ALL,
    APPLY_TO_RL,
    APPLY_TO_RL_RS,
    MODE_DIAGNOSTIC,
    MODE_DIRECTION_HOLD,
    MODE_OFF,
    ManeuverStabilityConfig,
    ManeuverStabilityTracker,
    direction_sign,
)


class Stage4ManeuverStabilityTests(unittest.TestCase):
    def test_direction_sign_ignores_small_speed_by_threshold(self) -> None:
        self.assertEqual(direction_sign(0.0, min_speed=0.05), 0)
        self.assertEqual(direction_sign(0.049, min_speed=0.05), 0)
        self.assertEqual(direction_sign(0.05, min_speed=0.05), 1)
        self.assertEqual(direction_sign(-0.07, min_speed=0.05), -1)

    def test_off_returns_equal_copy_without_counters(self) -> None:
        tracker = ManeuverStabilityTracker(ManeuverStabilityConfig(mode=MODE_OFF))
        raw = np.array([0.25, -0.75], dtype=float)

        result = tracker.apply(raw, source="RL")

        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "off")
        np.testing.assert_array_equal(result.applied_action, raw)
        self.assertIsNot(result.applied_action, raw)
        raw[1] = 0.9
        self.assertAlmostEqual(float(result.applied_action[1]), -0.75)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["last_direction"], 0)
        self.assertEqual(snapshot["held_flip_count"], 0)
        self.assertEqual(
            snapshot["counters"],
            {
                "steps": 0,
                "speed_sign_flips": 0,
                "would_change": 0,
                "interventions": 0,
                "by_source": {},
            },
        )

    def test_config_round_trips_mapping_and_legacy_none_defaults_to_off(self) -> None:
        default_config = ManeuverStabilityConfig.from_mapping(None)
        self.assertEqual(default_config.mode, MODE_OFF)

        config = ManeuverStabilityConfig.from_mapping(
            {
                "mode": MODE_DIRECTION_HOLD,
                "apply_to": APPLY_TO_RL_RS,
                "min_speed": "0.05",
                "hold_steps": "2",
                "max_hold_speed": "0.5",
            }
        )

        self.assertEqual(config.mode, MODE_DIRECTION_HOLD)
        self.assertEqual(config.apply_to, APPLY_TO_RL_RS)
        self.assertAlmostEqual(config.min_speed, 0.05)
        self.assertEqual(config.hold_steps, 2)
        self.assertAlmostEqual(config.max_hold_speed, 0.5)
        self.assertEqual(
            config.to_dict(),
            {
                "schema_version": 1,
                "mode": MODE_DIRECTION_HOLD,
                "apply_to": APPLY_TO_RL_RS,
                "min_speed": 0.05,
                "hold_steps": 2,
                "max_hold_speed": 0.5,
            },
        )

    def test_diagnostic_records_flip_without_changing_action(self) -> None:
        tracker = ManeuverStabilityTracker(ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC))

        tracker.apply(np.array([0.0, 0.5], dtype=float), source="RL")
        result = tracker.apply(np.array([0.0, -0.4], dtype=float), source="RL")

        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "diagnostic_speed_sign_flip")
        np.testing.assert_array_equal(result.applied_action, np.array([0.0, -0.4], dtype=float))
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["counters"]["speed_sign_flips"], 1)
        self.assertEqual(snapshot["counters"]["would_change"], 1)
        self.assertEqual(snapshot["counters"]["by_source"]["RL"]["speed_sign_flips"], 1)

    def test_direction_hold_suppresses_first_rl_flip_then_allows_after_hold_budget(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, apply_to=APPLY_TO_RL, hold_steps=1)
        )

        original_flip_action = np.array([0.0, -0.4], dtype=float)
        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        held = tracker.apply(original_flip_action, source="RL")
        allowed = tracker.apply(np.array([0.0, -0.3], dtype=float), source="RL")

        self.assertTrue(held.changed)
        self.assertEqual(held.reason, "direction_hold")
        self.assertAlmostEqual(float(held.applied_action[1]), 0.4)
        np.testing.assert_array_equal(original_flip_action, np.array([0.0, -0.4], dtype=float))
        self.assertFalse(allowed.changed)
        self.assertAlmostEqual(float(allowed.applied_action[1]), -0.3)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["counters"]["interventions"], 1)
        self.assertEqual(snapshot["counters"]["speed_sign_flips"], 2)

    def test_direction_hold_max_hold_speed_lets_large_flips_pass_through(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(
                mode=MODE_DIRECTION_HOLD,
                apply_to=APPLY_TO_RL,
                hold_steps=1,
                max_hold_speed=0.5,
            )
        )

        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        result = tracker.apply(np.array([0.0, -1.0], dtype=float), source="RL")

        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "direction_hold_speed_not_held")
        self.assertAlmostEqual(float(result.applied_action[1]), -1.0)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["last_direction"], -1)
        self.assertEqual(snapshot["counters"]["speed_sign_flips"], 1)
        self.assertEqual(snapshot["counters"]["would_change"], 0)
        self.assertEqual(snapshot["counters"]["interventions"], 0)

    def test_direction_hold_max_hold_speed_still_holds_small_flips(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(
                mode=MODE_DIRECTION_HOLD,
                apply_to=APPLY_TO_RL,
                hold_steps=1,
                max_hold_speed=0.5,
            )
        )

        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        result = tracker.apply(np.array([0.0, -0.4], dtype=float), source="RL")

        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "direction_hold")
        self.assertAlmostEqual(float(result.applied_action[1]), 0.4)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["counters"]["would_change"], 1)
        self.assertEqual(snapshot["counters"]["interventions"], 1)

    def test_direction_hold_respects_source_filter(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, apply_to=APPLY_TO_RL, hold_steps=3)
        )

        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        rs_result = tracker.apply(np.array([0.0, -0.4], dtype=float), source="RS")

        self.assertFalse(rs_result.changed)
        self.assertEqual(rs_result.reason, "source_not_enabled")
        self.assertAlmostEqual(float(rs_result.applied_action[1]), -0.4)

    def test_disabled_source_does_not_update_hold_state_or_flip_counters(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, apply_to=APPLY_TO_RL, hold_steps=1)
        )

        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        rs_result = tracker.apply(np.array([0.0, -0.4], dtype=float), source="RS")
        rl_result = tracker.apply(np.array([0.0, -0.3], dtype=float), source="RL")

        self.assertFalse(rs_result.changed)
        self.assertEqual(rs_result.reason, "source_not_enabled")
        self.assertTrue(rl_result.changed)
        self.assertEqual(rl_result.reason, "direction_hold")
        self.assertAlmostEqual(float(rl_result.applied_action[1]), 0.3)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["counters"]["speed_sign_flips"], 1)
        self.assertEqual(snapshot["counters"]["interventions"], 1)
        self.assertNotIn("RS", snapshot["counters"]["by_source"])

    def test_diagnostic_ignores_disabled_source_for_flip_state_and_counters(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC, apply_to=APPLY_TO_RL)
        )

        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        rs_result = tracker.apply(np.array([0.0, -0.4], dtype=float), source="RS")
        rl_result = tracker.apply(np.array([0.0, -0.3], dtype=float), source="RL")

        self.assertFalse(rs_result.changed)
        self.assertEqual(rs_result.reason, "diagnostic_source_not_enabled")
        self.assertFalse(rl_result.changed)
        self.assertEqual(rl_result.reason, "diagnostic_speed_sign_flip")
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["counters"]["speed_sign_flips"], 1)
        self.assertEqual(snapshot["counters"]["would_change"], 1)
        self.assertNotIn("RS", snapshot["counters"]["by_source"])

    def test_direction_hold_source_filter_can_include_rs_or_all_sources(self) -> None:
        rl_rs_tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, apply_to=APPLY_TO_RL_RS, hold_steps=1)
        )
        rl_rs_tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        rs_result = rl_rs_tracker.apply(np.array([0.0, -0.4], dtype=float), source="RS")

        self.assertTrue(rs_result.changed)
        self.assertEqual(rs_result.reason, "direction_hold")
        self.assertAlmostEqual(float(rs_result.applied_action[1]), 0.4)

        all_tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, apply_to=APPLY_TO_ALL, hold_steps=1)
        )
        all_tracker.apply(np.array([0.0, 0.6], dtype=float), source="planner")
        planner_result = all_tracker.apply(np.array([0.0, -0.4], dtype=float), source="planner")

        self.assertTrue(planner_result.changed)
        self.assertEqual(planner_result.reason, "direction_hold")
        self.assertAlmostEqual(float(planner_result.applied_action[1]), 0.4)

    def test_reset_episode_clears_direction_state_but_keeps_counters(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, hold_steps=1)
        )

        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        held = tracker.apply(np.array([0.0, -0.4], dtype=float), source="RL")
        tracker.reset_episode()
        after_reset = tracker.apply(np.array([0.0, -0.3], dtype=float), source="RL")

        self.assertTrue(held.changed)
        self.assertFalse(after_reset.changed)
        self.assertAlmostEqual(float(after_reset.applied_action[1]), -0.3)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["last_direction"], -1)
        self.assertEqual(snapshot["held_flip_count"], 0)
        self.assertEqual(snapshot["counters"]["interventions"], 1)
        self.assertEqual(snapshot["counters"]["speed_sign_flips"], 1)

    def test_from_snapshot_restores_tracker_state_and_counters(self) -> None:
        tracker = ManeuverStabilityTracker(ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC))
        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        tracker.apply(np.array([0.0, -0.4], dtype=float), source="RL")

        restored = ManeuverStabilityTracker.from_snapshot(tracker.snapshot())

        self.assertEqual(restored.snapshot(), tracker.snapshot())

    def test_invalid_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            ManeuverStabilityConfig(mode="surprise")
        with self.assertRaisesRegex(ValueError, "hold_steps"):
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, hold_steps=0)
        with self.assertRaisesRegex(ValueError, "apply_to"):
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, apply_to="bad")
        with self.assertRaisesRegex(ValueError, "min_speed"):
            ManeuverStabilityConfig(min_speed=-0.1)
        with self.assertRaisesRegex(ValueError, "min_speed"):
            ManeuverStabilityConfig(min_speed=float("nan"))
        with self.assertRaisesRegex(ValueError, "min_speed"):
            ManeuverStabilityConfig(min_speed=float("inf"))
        with self.assertRaisesRegex(ValueError, "hold_steps"):
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, hold_steps=1.5)
        with self.assertRaisesRegex(ValueError, "max_hold_speed"):
            ManeuverStabilityConfig(max_hold_speed=-0.1)
        with self.assertRaisesRegex(ValueError, "max_hold_speed"):
            ManeuverStabilityConfig(max_hold_speed=float("nan"))


if __name__ == "__main__":
    unittest.main()
