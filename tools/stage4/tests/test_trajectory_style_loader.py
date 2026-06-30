import json
import tempfile
import unittest
from pathlib import Path

from tools.stage4.debug_trace_export import build_fixture_trace
from tools.stage4.trajectory_style_loader import load_trajectory_traces


class TrajectoryStyleLoaderTests(unittest.TestCase):
    def _write_trace(self, trace):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "trace.json"
        path.write_text(json.dumps(trace, allow_nan=True), encoding="utf-8")
        self.addCleanup(temp_dir.cleanup)
        return path

    def test_loads_fixture_single_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.json"
            trace = build_fixture_trace(frame_count=3)
            path.write_text(json.dumps(trace), encoding="utf-8")

            traces = load_trajectory_traces(path)

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].case_uid, "fixture")
        self.assertEqual(len(traces[0].poses), 3)
        self.assertEqual(traces[0].slot_type, "fixture")
        self.assertEqual(traces[0].action_sources, ["RL", "RL", "RL"])

    def test_loads_collection_and_older_model_action_shape(self):
        trace = build_fixture_trace(frame_count=2)
        trace["case"]["case_uid"] = "parallel_013"
        trace["case"]["parking_type"] = "parallel"
        for frame in trace["frames"]:
            frame["action"] = {"source": "RL", "model": [0.1, -0.5], "env": [0.075, -1.25]}
            frame["status"]["planner_route_active"] = False
        payload = {
            "schema_version": "stage4-ogm-debug-v1",
            "source": {"checkpoint": "fixture"},
            "case_count": 1,
            "case_traces": [trace],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "collection.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            traces = load_trajectory_traces(path)

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].case_uid, "parallel_013")
        self.assertEqual(traces[0].slot_type, "parallel")
        self.assertEqual(traces[0].actions, [[0.1, -0.5], [0.1, -0.5]])
        self.assertEqual(traces[0].planner_route_active, [False, False])

    def test_target_pose_uses_closed_ring_polygon_centroid(self):
        trace = build_fixture_trace(frame_count=1)
        trace["static_geometry"]["target_polygon"] = [[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]
        path = self._write_trace(trace)

        traces = load_trajectory_traces(path)

        self.assertEqual(traces[0].target_pose.x, 2.0)
        self.assertEqual(traces[0].target_pose.y, 1.0)

    def test_rejects_non_string_or_blank_case_uid(self):
        for case_uid in (None, 13, ""):
            with self.subTest(case_uid=case_uid):
                trace = build_fixture_trace(frame_count=1)
                trace["case"]["case_uid"] = case_uid
                path = self._write_trace(trace)

                with self.assertRaisesRegex(ValueError, "case_uid"):
                    load_trajectory_traces(path)

    def test_rejects_non_bool_planner_route_active(self):
        trace = build_fixture_trace(frame_count=1)
        trace["frames"][0]["status"]["planner_route_active"] = "false"
        path = self._write_trace(trace)

        with self.assertRaisesRegex(ValueError, "planner_route_active"):
            load_trajectory_traces(path)

    def test_rejects_nan_summary_metric(self):
        trace = build_fixture_trace(frame_count=1)
        trace["summary"]["path_length_m"] = float("nan")
        path = self._write_trace(trace)

        with self.assertRaisesRegex(ValueError, "path_length_m"):
            load_trajectory_traces(path)

    def test_missing_planner_route_active_defaults_false(self):
        trace = build_fixture_trace(frame_count=2)
        for frame in trace["frames"]:
            frame["status"].pop("planner_route_active", None)
        path = self._write_trace(trace)

        traces = load_trajectory_traces(path)

        self.assertEqual(traces[0].planner_route_active, [False, False])


if __name__ == "__main__":
    unittest.main()
