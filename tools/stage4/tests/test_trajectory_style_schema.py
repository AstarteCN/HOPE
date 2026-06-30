import unittest

from tools.stage4.trajectory_style_schema import (
    ClearancePreferences,
    Pose2D,
    ReferenceFamily,
    StyleCaseReport,
    TrajectoryTrace,
    pose_from_mapping,
)


class TrajectoryStyleSchemaTests(unittest.TestCase):
    def test_pose_from_mapping_requires_finite_xy_heading(self):
        pose = pose_from_mapping({"x": 1, "y": -2, "heading": 0.5, "speed": -1.0})

        self.assertEqual(pose, Pose2D(x=1.0, y=-2.0, heading=0.5, speed=-1.0))

        with self.assertRaisesRegex(ValueError, "heading"):
            pose_from_mapping({"x": 0, "y": 0, "heading": float("nan")})

    def test_reference_and_case_report_are_json_serializable(self):
        trace = TrajectoryTrace(
            case_uid="parallel_001",
            source_trace_path="trace.json",
            scene_type="Sim-Complex",
            slot_type="parallel",
            start_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(5.0, 1.0, 0.0),
            poses=[Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.2, 0.1)],
            actions=[[0.1, -0.5], [0.2, -0.5]],
            action_sources=["RL", "RS"],
            planner_route_active=[False, True],
            obstacles=[[[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]],
            outcome={"success": True},
            raw_metrics={"path_length_m": 1.2},
        )
        reference = ReferenceFamily(
            family_id="parallel_standard_reverse_s_curve",
            scene_class="parallel_standard",
            route_family="reverse_s_curve",
            waypoints=[Pose2D(0.0, 0.0, 0.0), Pose2D(5.0, 1.0, 0.0)],
            corridor={"radius_m": 1.0},
            phase_labels=["approach", "reverse_entry"],
            expected_cusp_count=(0, 1),
            expected_gear_shift_count=(1, 2),
            slot_mouth_pose_window={"heading_error_rad": 0.4},
            clearance_preferences=ClearancePreferences(min_clearance_m=0.3, preferred_side="away_from_obstacle"),
            generation_method="analytic_v1",
        )
        report = StyleCaseReport(
            case_uid=trace.case_uid,
            scene_class=reference.scene_class,
            classification_reason="parking_type=parallel; slot length is comfortable",
            selected_reference_family=reference.family_id,
            shape_metrics={"l2_distance": 0.1},
            behavior_metrics={"gear_shifts": 1},
            clearance_metrics={"min_clearance_m": 2.0},
            segment_metrics={"rl_style_score": 0.9},
            style_label="human_like",
            style_score=0.91,
            diagnosis=[],
            unsupported_reason=None,
        )

        payload = report.to_dict()

        self.assertEqual(payload["case_uid"], "parallel_001")
        self.assertEqual(payload["style_label"], "human_like")
        self.assertEqual(reference.to_dict()["clearance_preferences"]["preferred_side"], "away_from_obstacle")
        self.assertEqual(trace.to_dict()["poses"][1]["heading"], 0.1)


if __name__ == "__main__":
    unittest.main()
