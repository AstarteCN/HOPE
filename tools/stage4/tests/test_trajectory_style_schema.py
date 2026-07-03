import json
import unittest

from tools.stage4.trajectory_style_schema import (
    ClearancePreferences,
    Pose2D,
    ReferenceFamily,
    SceneClassification,
    StyleCaseReport,
    TrajectoryTrace,
    pose_from_mapping,
)


class TrajectoryStyleSchemaTests(unittest.TestCase):
    class UnsafeToDict:
        def to_dict(self):
            return {"bad": object()}

    def _make_trace(self, **overrides):
        values = {
            "case_uid": "parallel_001",
            "source_trace_path": "trace.json",
            "scene_type": "Sim-Complex",
            "slot_type": "parallel",
            "start_pose": Pose2D(0.0, 0.0, 0.0),
            "target_pose": Pose2D(5.0, 1.0, 0.0),
            "poses": [Pose2D(0.0, 0.0, 0.0)],
            "actions": [[0.1, -0.5]],
            "action_sources": ["RL"],
            "planner_route_active": [False],
        }
        values.update(overrides)
        return TrajectoryTrace(**values)

    def _make_report(self, **overrides):
        values = {
            "case_uid": "parallel_001",
            "scene_class": "parallel_standard",
            "classification_reason": "parking_type=parallel; slot length is comfortable",
            "selected_reference_family": "parallel_standard_reverse_s_curve",
            "shape_metrics": {"l2_distance": 0.1},
            "behavior_metrics": {"gear_shifts": 1},
            "clearance_metrics": {"min_clearance_m": 2.0},
            "segment_metrics": {"rl_style_score": 0.9},
            "style_label": "human_like",
            "style_score": 0.91,
            "diagnosis": [],
            "unsupported_reason": None,
        }
        values.update(overrides)
        return StyleCaseReport(**values)

    def _make_reference(self, **overrides):
        values = {
            "family_id": "parallel_standard_reverse_s_curve",
            "scene_class": "parallel_standard",
            "route_family": "reverse_s_curve",
            "waypoints": [Pose2D(0.0, 0.0, 0.0), Pose2D(5.0, 1.0, 0.0)],
            "corridor": {"radius_m": 1.0},
            "phase_labels": ["approach", "reverse_entry"],
            "expected_cusp_count": (0, 1),
            "expected_gear_shift_count": (1, 2),
            "slot_mouth_pose_window": {"heading_error_rad": 0.4},
            "clearance_preferences": ClearancePreferences(min_clearance_m=0.3, preferred_side="away_from_obstacle"),
            "generation_method": "analytic_v1",
        }
        values.update(overrides)
        return ReferenceFamily(**values)

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

    def test_scene_classification_constructor_matches_downstream_plan(self):
        classification = SceneClassification("parallel_standard", "parking_type=parallel", 0.7)

        self.assertEqual(classification.scene_class, "parallel_standard")
        self.assertEqual(classification.reason, "parking_type=parallel")
        self.assertEqual(classification.confidence, 0.7)
        self.assertEqual(classification.to_dict()["confidence"], 0.7)

    def test_report_to_dict_supports_strict_json_dump(self):
        report = self._make_report()

        json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)

    def test_nested_unsupported_objects_raise_during_to_dict(self):
        trace = TrajectoryTrace(
            case_uid="parallel_001",
            source_trace_path="trace.json",
            scene_type="Sim-Complex",
            slot_type="parallel",
            start_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(5.0, 1.0, 0.0),
            poses=[Pose2D(0.0, 0.0, 0.0)],
            actions=[[0.1, -0.5]],
            action_sources=["RL"],
            planner_route_active=[False],
            raw_metrics={"bad": object()},
        )
        report = self._make_report(shape_metrics={"bad": object()})

        with self.assertRaisesRegex(TypeError, "JSON-safe"):
            trace.to_dict()
        with self.assertRaisesRegex(TypeError, "JSON-safe"):
            report.to_dict()

    def test_action_sources_reject_non_string_values_during_to_dict(self):
        trace = self._make_trace(action_sources=[object()])

        with self.assertRaisesRegex((TypeError, ValueError), "action_sources"):
            trace.to_dict()

    def test_report_diagnosis_rejects_non_string_values_during_to_dict(self):
        report = self._make_report(diagnosis=[object()])

        with self.assertRaisesRegex((TypeError, ValueError), "diagnosis"):
            report.to_dict()

    def test_planner_route_active_rejects_non_bool_values_during_to_dict(self):
        trace = self._make_trace(planner_route_active=[object()])

        with self.assertRaisesRegex((TypeError, ValueError), "planner_route_active"):
            trace.to_dict()

    def test_scene_classification_rejects_non_finite_confidence(self):
        classification = SceneClassification("parallel_standard", "ok", float("nan"))

        with self.assertRaisesRegex(ValueError, "confidence"):
            classification.to_dict()

    def test_delegated_to_dict_results_are_recursively_validated(self):
        trace = self._make_trace(raw_metrics={"delegated": self.UnsafeToDict()})
        report = self._make_report(shape_metrics={"delegated": self.UnsafeToDict()})

        with self.assertRaisesRegex((TypeError, ValueError), "JSON-safe"):
            trace.to_dict()
        with self.assertRaisesRegex((TypeError, ValueError), "JSON-safe"):
            report.to_dict()

    def test_reference_phase_labels_reject_non_string_values_during_to_dict(self):
        reference = self._make_reference(phase_labels=[object()])

        with self.assertRaisesRegex((TypeError, ValueError), "phase_labels"):
            reference.to_dict()


if __name__ == "__main__":
    unittest.main()
