import json
import tempfile
import unittest
from pathlib import Path

from tools.stage4.debug_trace_export import build_fixture_trace
from tools.stage4.trajectory_style_loader import load_trajectory_traces


class TrajectoryStyleLoaderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
