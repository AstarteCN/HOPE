import json
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from tools.stage4.debug_trace_export import (
    build_fixture_trace,
    build_trace,
    export_checkpoint_traces,
    select_cases_by_uid,
    write_trace_json,
)
from tools.stage4.stage4_maneuver_stability import (
    APPLY_TO_ALL,
    MODE_DIRECTION_HOLD,
    ManeuverStabilityConfig,
)
from tools.stage4.stage4_target_transform import TARGET_MODE_CORRECTED_COSSIN, TARGET_MODE_LEGACY_COSCOS


REPO_ROOT = Path(__file__).resolve().parents[3]


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(seed):
        return None


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def no_grad():
        return _NoGrad()

    @staticmethod
    def manual_seed(seed):
        return None


class _FakeSACAgent:
    def __init__(self, config):
        self.config = config

    def load(self, checkpoint, params_only=True):
        self.checkpoint = checkpoint
        self.params_only = params_only


class _FakeRsPlanner:
    def __init__(self, step_ratio):
        self.step_ratio = step_ratio


class _FakeParkingAgent:
    def __init__(self, rl_agent, planner):
        self.rl_agent = rl_agent
        self.planner = SimpleNamespace(route=None)
        self._actions = [np.array([0.1, 1.0]), np.array([0.2, -1.0])]
        self._index = 0

    @property
    def executing_rs(self):
        return self._index == 1

    def reset(self):
        self._index = 0

    def get_action(self, obs):
        action = self._actions[self._index]
        self._index += 1
        return action.copy(), None

    def set_planner_path(self, path):
        self.planner.route = path


class _FakeStatus:
    CONTINUE = SimpleNamespace(name="CONTINUE")
    ARRIVED = SimpleNamespace(name="ARRIVED")


def _fake_state(x, y, speed=0.0):
    return SimpleNamespace(
        loc=SimpleNamespace(x=float(x), y=float(y)),
        heading=0.0,
        speed=float(speed),
        steering=0.0,
    )


class _FakeEnv:
    def __init__(self):
        self.action_space = SimpleNamespace()
        self.env = SimpleNamespace()
        self.observation_shape = {"ogm": (2, 2, 2)}
        self.vehicle = SimpleNamespace(
            kinetic_model=SimpleNamespace(step_len=1.0, n_step=1),
            state=_fake_state(0.0, 0.0),
            trajectory=[],
        )
        self.vehicle.trajectory = [self.vehicle.state]
        self.stepped_actions = []

    def reset(self, case_id, *_args):
        self.vehicle.state = _fake_state(0.0, 0.0)
        self.vehicle.trajectory = [self.vehicle.state]
        self.stepped_actions = []
        return self._obs(target_x=10.0)

    def step(self, action):
        action_array = np.asarray(action, dtype=float).reshape(-1).copy()
        self.stepped_actions.append(action_array)
        step_number = len(self.stepped_actions)
        self.vehicle.state = _fake_state(step_number, 0.0, speed=action_array[1])
        self.vehicle.trajectory.append(self.vehicle.state)
        done = step_number >= 2
        status = _FakeStatus.ARRIVED if done else _FakeStatus.CONTINUE
        return (
            self._obs(target_x=90.0 + step_number),
            1.0,
            done,
            {"status": status, "path_to_dest": None, "reward_info": {"progress": 1.0}},
        )

    def close(self):
        return None

    @staticmethod
    def _obs(target_x):
        return {
            "target": np.array([target_x, 0.0, 0.0, 1.0, 1.0], dtype=float),
            "action_mask": np.array([1.0, 0.0, 1.0], dtype=float),
            "ogm": np.zeros((2, 2, 2), dtype=float),
            "lidar": np.array([1.0, 2.0], dtype=float),
        }


def _fake_runtime(fake_env, cases):
    return {
        "torch": _FakeTorch(),
        "VALID_SPEED": [0.0, 2.0],
        "build_stage4_env": lambda **_kwargs: fake_env,
        "Status": _FakeStatus,
        "ParkingAgent": _FakeParkingAgent,
        "RsPlanner": _FakeRsPlanner,
        "SACAgent": _FakeSACAgent,
        "action_rescale": lambda action, _action_space, explore=False: np.asarray(action, dtype=float) * 2.0,
        "build_ogm_agent_config": lambda env: {"ogm_shape": env.observation_shape["ogm"]},
        "count_gear_shifts": lambda speeds: int(
            sum(np.sign(speeds[index]) != np.sign(speeds[index - 1]) for index in range(1, len(speeds)))
        ),
        "load_cases": lambda _path: cases,
    }


def _fake_case(case_uid):
    return {
        "case_uid": case_uid,
        "seed": 123,
        "hope_case_id": 0,
        "hope_level": "Normal",
        "split": "Sim-Normal",
        "parking_type": "parallel",
    }


class DebugTraceExportTest(unittest.TestCase):
    def test_select_cases_by_uid_preserves_requested_order(self):
        cases = [
            {"case_uid": "a", "split": "Sim-Complex"},
            {"case_uid": "b", "split": "Sim-Complex"},
            {"case_uid": "c", "split": "Sim-Normal"},
        ]

        selected = select_cases_by_uid(cases, ["c", "a"])

        self.assertEqual([case["case_uid"] for case in selected], ["c", "a"])

    def test_select_cases_by_uid_rejects_missing_case(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            select_cases_by_uid([{"case_uid": "a"}], ["missing"])

    def test_fixture_trace_has_required_shape(self):
        trace = build_fixture_trace(frame_count=6)

        self.assertEqual(trace["schema_version"], "stage4-ogm-debug-v1")
        self.assertIn("vehicle", trace)
        self.assertIn("grid", trace)
        self.assertIn("case", trace)
        self.assertEqual(trace["source"]["target_mode"], TARGET_MODE_LEGACY_COSCOS)
        self.assertEqual(trace["frames"][0]["target"][3:5], [1.0, 1.0])
        self.assertEqual(len(trace["frames"]), 6)
        self.assertEqual([frame["sim_time_s"] for frame in trace["frames"]], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self.assertIn("rl_trajectory", trace["frames"][-1])
        self.assertIn("rs_trajectory", trace["frames"][-1])

    def test_fixture_trace_action_records_raw_applied_env_and_intervention_reason(self):
        trace = build_fixture_trace(frame_count=1)

        action = trace["frames"][0]["action"]

        self.assertIn("raw_model", action)
        self.assertIn("applied_model", action)
        self.assertIn("env", action)
        self.assertIn("intervention_reason", action)
        self.assertIn("intervention_changed", action)
        self.assertEqual(action["raw_model"], action["applied_model"])
        self.assertFalse(action["intervention_changed"])
        self.assertEqual(action["intervention_reason"], "off")

    def test_checkpoint_trace_records_live_raw_applied_env_action_and_pre_step_observation(self):
        fake_env = _FakeEnv()
        runtime = _fake_runtime(fake_env, [_fake_case("case-a")])

        with mock.patch("tools.stage4.debug_trace_export._load_stage4_runtime", return_value=runtime), mock.patch(
            "tools.stage4.debug_trace_export._static_geometry",
            return_value={"obstacle_polygons": [], "target_polygon": [], "start_polygon": [], "map_bounds": []},
        ):
            trace = export_checkpoint_traces(
                checkpoint_path=Path("SAC_79999.pt"),
                cases_path=Path("cases.json"),
                case_uids=None,
                case_indices=None,
                action_selection="get_action",
                max_steps=2,
                max_ogm_cells_per_frame=None,
                target_mode=TARGET_MODE_LEGACY_COSCOS,
                maneuver_stability_config=ManeuverStabilityConfig(
                    mode=MODE_DIRECTION_HOLD,
                    apply_to=APPLY_TO_ALL,
                    hold_steps=1,
                ),
            )

        first_action = trace["frames"][0]["action"]
        second_action = trace["frames"][1]["action"]

        self.assertEqual(trace["frames"][0]["timing"]["observation"], "pre_step")
        self.assertEqual(trace["frames"][0]["target"][0], 10.0)
        self.assertEqual(trace["frames"][1]["target"][0], 91.0)
        self.assertEqual(first_action["raw_model"], [0.1, 1.0])
        self.assertEqual(first_action["applied_model"], [0.1, 1.0])
        self.assertEqual(first_action["env"], [0.2, 2.0])
        self.assertFalse(first_action["intervention_changed"])
        self.assertEqual(second_action["source"], "RS")
        self.assertEqual(second_action["raw_model"], [0.2, -1.0])
        self.assertEqual(second_action["applied_model"], [0.2, 1.0])
        self.assertEqual(second_action["env"], [0.4, 2.0])
        self.assertTrue(second_action["intervention_changed"])
        self.assertEqual(second_action["intervention_reason"], "direction_hold")
        self.assertEqual([float(action[1]) for action in fake_env.stepped_actions], [1.0, 1.0])
        self.assertEqual(trace["summary"]["gear_shifts"], 0)
        self.assertEqual(trace["summary"]["maneuver_stability"]["counters"]["interventions"], 1)

    def test_checkpoint_trace_maneuver_counters_are_per_case(self):
        fake_env = _FakeEnv()
        runtime = _fake_runtime(fake_env, [_fake_case("case-a"), _fake_case("case-b")])

        with mock.patch("tools.stage4.debug_trace_export._load_stage4_runtime", return_value=runtime), mock.patch(
            "tools.stage4.debug_trace_export._static_geometry",
            return_value={"obstacle_polygons": [], "target_polygon": [], "start_polygon": [], "map_bounds": []},
        ):
            trace_bundle = export_checkpoint_traces(
                checkpoint_path=Path("SAC_79999.pt"),
                cases_path=Path("cases.json"),
                case_uids=None,
                case_indices=[0, 1],
                action_selection="get_action",
                max_steps=2,
                max_ogm_cells_per_frame=None,
                target_mode=TARGET_MODE_LEGACY_COSCOS,
                maneuver_stability_config=ManeuverStabilityConfig(
                    mode=MODE_DIRECTION_HOLD,
                    apply_to=APPLY_TO_ALL,
                    hold_steps=1,
                ),
            )

        traces = trace_bundle["case_traces"]

        self.assertEqual([trace["case"]["case_uid"] for trace in traces], ["case-a", "case-b"])
        self.assertEqual(
            [trace["summary"]["maneuver_stability"]["counters"]["interventions"] for trace in traces],
            [1, 1],
        )

    def test_write_trace_json_creates_parent_and_round_trips(self):
        trace = build_trace(
            source={"checkpoint": "fixture"},
            case={"case_uid": "fixture"},
            static_geometry={"obstacle_polygons": [], "target_polygon": []},
            frames=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "trace.json"
            write_trace_json(trace, output)
            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(loaded["schema_version"], "stage4-ogm-debug-v1")
        self.assertEqual(loaded["source"]["checkpoint"], "fixture")

    def test_fixture_cli_succeeds_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fixture.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.stage4.debug_trace_export",
                    "--fixture",
                    "--output",
                    str(output),
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())

    def test_fixture_cli_records_requested_target_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fixture.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.stage4.debug_trace_export",
                    "--fixture",
                    "--target-mode",
                    TARGET_MODE_CORRECTED_COSSIN,
                    "--output",
                    str(output),
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["source"]["target_mode"], TARGET_MODE_CORRECTED_COSSIN)
            self.assertEqual(loaded["frames"][0]["target"][3:5], [1.0, 0.0])

    def test_direct_script_help_succeeds_from_src_working_directory(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "stage4" / "debug_trace_export.py"),
                "--help",
            ],
            cwd=REPO_ROOT / "src",
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--target-mode", completed.stdout)
        self.assertIn("--maneuver_stability_mode", completed.stdout)


if __name__ == "__main__":
    unittest.main()
