import unittest

from tools.stage4.trajectory_style_reference import classify_scene, generate_reference_families
from tools.stage4.trajectory_style_schema import Pose2D, TrajectoryTrace


def _trace(case_uid, slot_type, start, target, obstacles=None, path_length=10.0):
    return TrajectoryTrace(
        case_uid=case_uid,
        source_trace_path="fixture.json",
        scene_type="Sim-Complex",
        slot_type=slot_type,
        start_pose=start,
        target_pose=target,
        poses=[start, target],
        actions=[[0.0, -0.5], [0.0, -0.5]],
        action_sources=["RL", "RS"],
        planner_route_active=[False, True],
        obstacles=obstacles or [],
        outcome={"success": True},
        raw_metrics={"path_length_m": path_length},
    )


class TrajectoryStyleReferenceTests(unittest.TestCase):
    def test_classifies_perpendicular_enough_and_limited(self):
        enough = _trace(
            "perpendicular_enough",
            "perpendicular",
            Pose2D(0.0, -8.0, 1.57),
            Pose2D(0.0, 0.0, 1.57),
        )
        limited = _trace(
            "perpendicular_limited",
            "perpendicular",
            Pose2D(0.0, -3.0, 1.57),
            Pose2D(0.0, 0.0, 1.57),
            obstacles=[[[1.0, -2.0], [2.0, -2.0], [2.0, 0.0], [1.0, 0.0]]],
        )

        self.assertEqual(classify_scene(enough).scene_class, "perpendicular_enough_space")
        self.assertEqual(classify_scene(limited).scene_class, "perpendicular_limited_space")

    def test_classifies_parallel_standard_and_tight(self):
        standard = _trace("parallel_standard", "parallel", Pose2D(0.0, 0.0, 0.0), Pose2D(8.0, 0.0, 0.0))
        tight = _trace("parallel_tight", "parallel", Pose2D(0.0, 0.0, 0.0), Pose2D(4.5, 0.0, 0.0))

        self.assertEqual(classify_scene(standard).scene_class, "parallel_standard")
        self.assertEqual(classify_scene(tight).scene_class, "parallel_tight")

    def test_generates_reference_family_with_phase_labels(self):
        trace = _trace(
            "parallel_standard",
            "parallel",
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(8.0, 0.0, 0.0),
        )

        references = generate_reference_families(trace, classify_scene(trace))

        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].scene_class, "parallel_standard")
        self.assertEqual(references[0].route_family, "reverse_s_curve")
        self.assertGreaterEqual(len(references[0].waypoints), 5)
        self.assertIn("reverse_entry", references[0].phase_labels)

    def test_perpendicular_limited_reference_uses_straighten_phase(self):
        trace = _trace(
            "perpendicular_limited",
            "perpendicular",
            Pose2D(0.0, -3.0, 1.57),
            Pose2D(0.0, 0.0, 1.57),
            obstacles=[[[1.0, -2.0], [2.0, -2.0], [2.0, 0.0], [1.0, 0.0]]],
        )

        references = generate_reference_families(trace, classify_scene(trace))

        self.assertEqual(
            references[0].phase_labels,
            ["approach", "reverse_entry", "planned_correction", "straighten"],
        )
        self.assertNotIn("late_straighten", references[0].phase_labels)

    def test_invalid_string_override_raises_immediately(self):
        trace = _trace("override_invalid", "parallel", Pose2D(0.0, 0.0, 0.0), Pose2D(8.0, 0.0, 0.0))

        with self.assertRaisesRegex(ValueError, "parallel_standrad.*override_invalid"):
            classify_scene(trace, label_overrides={trace.case_uid: "parallel_standrad"})

    def test_valid_string_override_serializes(self):
        trace = _trace("override_string", "perpendicular", Pose2D(0.0, -8.0, 1.57), Pose2D(0.0, 0.0, 1.57))

        classification = classify_scene(trace, label_overrides={trace.case_uid: "parallel_tight"})
        payload = classification.to_dict()

        self.assertEqual(classification.scene_class, "parallel_tight")
        self.assertEqual(payload["scene_class"], "parallel_tight")
        self.assertEqual(payload["confidence"], 1.0)

    def test_valid_mapping_override_serializes(self):
        trace = _trace("override_mapping", "perpendicular", Pose2D(0.0, -8.0, 1.57), Pose2D(0.0, 0.0, 1.57))

        classification = classify_scene(trace, label_overrides={trace.case_uid: {"scene_class": "parallel_tight"}})
        payload = classification.to_dict()

        self.assertEqual(classification.scene_class, "parallel_tight")
        self.assertEqual(payload["scene_class"], "parallel_tight")

    def test_unsupported_slot_type_serializes_without_references(self):
        trace = _trace("unsupported_angle", "angled", Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 4.0, 0.5))

        classification = classify_scene(trace)

        self.assertEqual(classification.to_dict()["scene_class"], "unsupported")
        self.assertEqual(generate_reference_families(trace, classification), [])

    def test_all_supported_reference_families_serialize(self):
        traces = [
            _trace("perpendicular_enough", "perpendicular", Pose2D(0.0, -8.0, 1.57), Pose2D(0.0, 0.0, 1.57)),
            _trace(
                "perpendicular_limited",
                "perpendicular",
                Pose2D(0.0, -3.0, 1.57),
                Pose2D(0.0, 0.0, 1.57),
                obstacles=[[[1.0, -2.0], [2.0, -2.0], [2.0, 0.0], [1.0, 0.0]]],
            ),
            _trace("parallel_standard", "parallel", Pose2D(0.0, 0.0, 0.0), Pose2D(8.0, 0.0, 0.0)),
            _trace("parallel_tight", "parallel", Pose2D(0.0, 0.0, 0.0), Pose2D(4.5, 0.0, 0.0)),
        ]

        scene_classes = []
        for trace in traces:
            classification = classify_scene(trace)
            references = generate_reference_families(trace, classification)

            scene_classes.append(classification.to_dict()["scene_class"])
            self.assertEqual(len(references), 1)
            self.assertEqual(references[0].to_dict()["scene_class"], classification.scene_class)

        self.assertEqual(
            scene_classes,
            [
                "perpendicular_enough_space",
                "perpendicular_limited_space",
                "parallel_standard",
                "parallel_tight",
            ],
        )


if __name__ == "__main__":
    unittest.main()
