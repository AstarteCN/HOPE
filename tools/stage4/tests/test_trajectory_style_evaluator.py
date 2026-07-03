import unittest

from tools.stage4.trajectory_style_evaluator import evaluate_trace_style
from tools.stage4.trajectory_style_reference import classify_scene, generate_reference_families
from tools.stage4.trajectory_style_schema import Pose2D, TrajectoryTrace


def _trace_with_poses(case_uid, poses, speeds):
    return TrajectoryTrace(
        case_uid=case_uid,
        source_trace_path="fixture.json",
        scene_type="Sim-Complex",
        slot_type="parallel",
        start_pose=poses[0],
        target_pose=poses[-1],
        poses=poses,
        actions=[[0.0, speed] for speed in speeds],
        action_sources=["RL"] * len(poses),
        planner_route_active=[False] * len(poses),
        obstacles=[],
        outcome={"success": True},
        raw_metrics={"path_length_m": 8.0},
    )


class TrajectoryStyleEvaluatorTests(unittest.TestCase):
    def test_identical_reference_like_path_is_human_like(self):
        trace = _trace_with_poses(
            "parallel_standard",
            [Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 0.8, 0.0), Pose2D(8.0, 0.0, 0.0)],
            [-0.5, -0.5, -0.5],
        )
        classification = classify_scene(trace)
        references = generate_reference_families(trace, classification)
        trace = _trace_with_poses(trace.case_uid, references[0].waypoints, [-0.5] * len(references[0].waypoints))

        report = evaluate_trace_style(trace, classification, references)

        self.assertEqual(report.style_label, "human_like")
        self.assertGreater(report.style_score, 0.8)
        self.assertEqual(report.diagnosis, [])

    def test_extra_cusps_create_style_mismatch_diagnosis(self):
        poses = [Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, 1.0, 0.0), Pose2D(4.0, -1.0, 0.0), Pose2D(8.0, 0.0, 0.0)]
        trace = _trace_with_poses("parallel_standard", poses, [-0.2, 0.2, -0.2, 0.2])
        classification = classify_scene(trace)
        references = generate_reference_families(trace, classification)

        report = evaluate_trace_style(trace, classification, references)

        self.assertIn(report.style_label, {"style_mismatch", "acceptable_variant"})
        self.assertIn("excessive_chatter", report.diagnosis)
        self.assertIn("unplanned_extra_cusp", report.diagnosis)

    def test_unsupported_case_returns_unsupported_report(self):
        trace = TrajectoryTrace(
            case_uid="angled_001",
            source_trace_path="fixture.json",
            scene_type="Sim-Complex",
            slot_type="angled",
            start_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(1.0, 1.0, 0.0),
            poses=[Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 1.0, 0.0)],
            actions=[[0.0, -0.5], [0.0, -0.5]],
            action_sources=["RL", "RL"],
            planner_route_active=[False, False],
            obstacles=[],
            outcome={"success": True},
            raw_metrics={},
        )
        classification = classify_scene(trace)

        report = evaluate_trace_style(trace, classification, [])

        self.assertEqual(report.style_label, "unsupported")
        self.assertEqual(report.unsupported_reason, classification.reason)

    def test_far_off_path_adds_route_family_mismatch_diagnosis(self):
        poses = [
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.0, 20.0, 0.0),
            Pose2D(8.0, 20.0, 0.0),
            Pose2D(8.0, 0.0, 0.0),
        ]
        trace = _trace_with_poses("parallel_standard", poses, [-0.5, -0.5, -0.5, -0.5])
        classification = classify_scene(trace)
        references = generate_reference_families(trace, classification)

        report = evaluate_trace_style(trace, classification, references)

        self.assertEqual(report.style_label, "style_mismatch")
        self.assertIn("route_family_mismatch", report.diagnosis)

    def test_clearance_below_reference_min_adds_diagnosis_without_unsafe_label(self):
        trace = _trace_with_poses(
            "parallel_standard",
            [Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 0.0, 0.0), Pose2D(8.0, 0.0, 0.0)],
            [-0.5, -0.5, -0.5],
        )
        trace = TrajectoryTrace(
            case_uid=trace.case_uid,
            source_trace_path=trace.source_trace_path,
            scene_type=trace.scene_type,
            slot_type=trace.slot_type,
            start_pose=trace.start_pose,
            target_pose=trace.target_pose,
            poses=trace.poses,
            actions=trace.actions,
            action_sources=trace.action_sources,
            planner_route_active=trace.planner_route_active,
            obstacles=[[[3.0, 0.2], [5.0, 0.2], [5.0, 0.4], [3.0, 0.4]]],
            outcome=trace.outcome,
            raw_metrics=trace.raw_metrics,
        )
        classification = classify_scene(trace)
        references = generate_reference_families(trace, classification)

        report = evaluate_trace_style(trace, classification, references)

        self.assertGreaterEqual(report.clearance_metrics["min_clearance_m"], 0.15)
        self.assertIn("object_side_clearance_too_close", report.diagnosis)
        self.assertNotEqual(report.style_label, "unsafe_or_too_close")

    def test_supported_empty_references_preserves_scene_class(self):
        trace = _trace_with_poses(
            "parallel_standard",
            [Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 0.8, 0.0), Pose2D(8.0, 0.0, 0.0)],
            [-0.5, -0.5, -0.5],
        )
        classification = classify_scene(trace)

        report = evaluate_trace_style(trace, classification, [])

        self.assertEqual(report.style_label, "unsupported")
        self.assertEqual(report.scene_class, classification.scene_class)
        self.assertEqual(report.unsupported_reason, "no reference family for parallel_standard")
        self.assertIn("route_family_mismatch", report.diagnosis)

    def test_supported_report_to_dict_contains_clear_shape_metric_keys(self):
        trace = _trace_with_poses(
            "parallel_standard",
            [Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 0.8, 0.0), Pose2D(8.0, 0.0, 0.0)],
            [-0.5, -0.5, -0.5],
        )
        classification = classify_scene(trace)
        references = generate_reference_families(trace, classification)

        report_dict = evaluate_trace_style(trace, classification, references).to_dict()

        self.assertIn("l2_distance", report_dict["shape_metrics"])
        self.assertIn("hausdorff_distance", report_dict["shape_metrics"])
        self.assertIn("fourier_descriptor_distance", report_dict["shape_metrics"])


if __name__ == "__main__":
    unittest.main()
