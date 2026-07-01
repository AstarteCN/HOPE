from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.stage4.stage4_ogm_targets import (
    BASELINE_FAST_ACTION_MASK_20K,
    OGM_SIM_TARGETS,
    relative_band,
)
from tools.stage4.stage4_ogm_progress import (
    build_progress_gate_input,
    decide_progress_gate,
    recommend_recovery_action,
)


class Stage4TargetTests(unittest.TestCase):
    def test_ogm_sim_targets_match_prd_values(self) -> None:
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["psr"], 0.9933)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["angs"], 1.5)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["pl"], 20.3)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["psr"], 0.977)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["angs"], 1.9)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["pl"], 23.6)

    def test_relative_band_uses_three_percent_window(self) -> None:
        low, high = relative_band(100.0)
        self.assertAlmostEqual(low, 97.0)
        self.assertAlmostEqual(high, 103.0)

    def test_fast_action_mask_20k_baseline_is_recorded(self) -> None:
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["id"], "hope-fast-action-mask-20k")
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["run_dir"], "src/log/exp/sac_20260620_085208")
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["checkpoint"], "SAC_19999.pt")
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["wall_time_hours"], 10.947863)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["episodes_per_hour"], 1826.840564)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["env_steps_per_second"], 46.223481)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Normal"], 0.985)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Complex"], 0.945)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Extrem"], 0.655)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["DLP"], 0.960)


class Stage4ProgressGateTests(unittest.TestCase):
    def _tensorboard_summary(
        self,
        *,
        avg_reward_delta: float = -0.2,
        psr_deltas: dict[str, float] | None = None,
        step_first_mean: float = 80.0,
        step_last_mean: float = 90.0,
        has_nonfinite: bool = False,
        hard_reject_has_nonfinite: bool | None = None,
    ) -> dict:
        scalars = {
            "avg_reward": {"trend": {"delta": avg_reward_delta}},
            "step_num": {"trend": {"first_mean": step_first_mean, "last_mean": step_last_mean}},
        }
        for name, delta in (psr_deltas or {}).items():
            scalars[name] = {"trend": {"delta": delta}}
        result = {"has_nonfinite": has_nonfinite, "scalars": scalars}
        if hard_reject_has_nonfinite is not None:
            result["hard_reject_has_nonfinite"] = hard_reject_has_nonfinite
        return result

    def _current_eval(self) -> dict:
        return {
            "target_mode": "corrected-cossin",
            "summary": {
                "Sim-Normal": {"psr": 0.96, "angs": 2.0, "pl": 20.0},
                "Sim-Complex": {"psr": 0.70, "angs": 8.0, "pl": 33.0},
            },
        }

    def _previous_eval(self) -> dict:
        return {
            "target_mode": "corrected-cossin",
            "summary": {
                "Sim-Normal": {"psr": 0.95, "angs": 2.5, "pl": 24.0},
                "Sim-Complex": {"psr": 0.75, "angs": 7.0, "pl": 31.0},
            },
        }

    def test_build_gate_input_80k_like_weak_trend_gets_grace(self) -> None:
        current = build_progress_gate_input(
            episode=80000,
            tensorboard_summary=self._tensorboard_summary(
                psr_deltas={
                    "success_rate_Normal": 0.02,
                    "success_rate_Complex": 0.01,
                    "success_rate_Extrem": 0.03,
                    "success_rate_dlp": 0.02,
                },
            ),
            fixed_eval=self._current_eval(),
            checkpoint_exists=True,
            state_snapshot_required=True,
            state_snapshot_exists=True,
        )

        self.assertEqual(current["episode"], 80000)
        self.assertFalse(current["has_nonfinite"])
        self.assertTrue(current["checkpoint_exists"])
        self.assertTrue(current["state_snapshot_required"])
        self.assertTrue(current["state_snapshot_exists"])
        self.assertAlmostEqual(current["trend"]["avg_reward_delta"], -0.2)
        self.assertAlmostEqual(current["trend"]["mean_psr_delta"], 0.02)
        self.assertAlmostEqual(current["trend"]["step_num_improvement"], -0.125)
        self.assertEqual(current["summary"], self._current_eval()["summary"])

        decision = decide_progress_gate(
            current=current,
            previous=build_progress_gate_input(
                episode=70000,
                tensorboard_summary=self._tensorboard_summary(),
                fixed_eval=self._previous_eval(),
                checkpoint_exists=True,
            ),
            previous_bad_gate=False,
        )
        self.assertEqual(decision["decision"], "grace-10k")
        self.assertIn("first_bad_progress_gate", decision["reasons"])

    def test_previous_bad_gate_plus_same_weak_trend_stops(self) -> None:
        current = build_progress_gate_input(
            episode=80000,
            tensorboard_summary=self._tensorboard_summary(
                psr_deltas={
                    "success_rate_Normal": 0.02,
                    "success_rate_Complex": 0.01,
                },
            ),
            fixed_eval=self._current_eval(),
            checkpoint_exists=True,
        )
        previous = build_progress_gate_input(
            episode=70000,
            tensorboard_summary=self._tensorboard_summary(),
            fixed_eval=self._previous_eval(),
            checkpoint_exists=True,
        )

        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)

        self.assertEqual(decision["decision"], "stop")
        self.assertIn("second_consecutive_bad_progress_gate", decision["reasons"])

    def test_hard_reject_has_nonfinite_overrides_regular_nonfinite_flag(self) -> None:
        current = build_progress_gate_input(
            episode=90000,
            tensorboard_summary=self._tensorboard_summary(
                has_nonfinite=False,
                hard_reject_has_nonfinite=True,
                avg_reward_delta=0.5,
                psr_deltas={"success_rate_Normal": 0.5, "success_rate_Complex": 0.5},
                step_first_mean=100.0,
                step_last_mean=50.0,
            ),
            fixed_eval=self._current_eval(),
            checkpoint_exists=True,
        )

        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)

        self.assertTrue(current["has_nonfinite"])
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("nonfinite_tensorboard_or_eval_metric", decision["reasons"])

    def test_missing_required_state_snapshot_from_builder_stops(self) -> None:
        current = build_progress_gate_input(
            episode=90000,
            tensorboard_summary=self._tensorboard_summary(
                avg_reward_delta=0.5,
                psr_deltas={"success_rate_Normal": 0.5, "success_rate_Complex": 0.5},
                step_first_mean=100.0,
                step_last_mean=50.0,
            ),
            fixed_eval=self._current_eval(),
            checkpoint_exists=True,
            state_snapshot_required=True,
            state_snapshot_exists=False,
        )

        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)

        self.assertEqual(decision["decision"], "stop")
        self.assertIn("missing_state_snapshot", decision["reasons"])

    def test_decide_stage4_ogm_gate_cli_emits_decision_and_trend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            previous_eval_path = root / "previous.eval.json"
            checkpoint_path = root / "SAC_79999.pt"
            state_path = root / "stage4_state_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            state_path.write_text("state-placeholder", encoding="utf-8")
            tensorboard_summary = self._tensorboard_summary(
                psr_deltas={
                    "success_rate_Normal": 0.02,
                    "success_rate_Complex": 0.01,
                    "success_rate_Extrem": 0.03,
                    "success_rate_dlp": 0.02,
                }
            )
            tensorboard_summary["log_dir"] = str(root)
            tensorboard_summary["event_dir"] = str(root)
            tensorboard_path.write_text(
                json.dumps(tensorboard_summary),
                encoding="utf-8",
            )
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")
            previous_eval_path.write_text(json.dumps(self._previous_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--previous-eval-json",
                    str(previous_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--state-snapshot",
                    str(state_path),
                    "--state-snapshot-required",
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
                check=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["decision"]["decision"], "grace-10k")
        self.assertEqual(payload["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(payload["state_snapshot_path"], str(state_path))
        self.assertEqual(payload["current_target_mode"], "corrected-cossin")
        self.assertEqual(payload["previous_target_mode"], "corrected-cossin")
        self.assertAlmostEqual(payload["current"]["trend"]["mean_psr_delta"], 0.02)
        self.assertAlmostEqual(payload["current"]["trend"]["step_num_improvement"], -0.125)

    def test_decide_stage4_ogm_gate_cli_rejects_mismatched_target_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            previous_eval_path = root / "previous.eval.json"
            checkpoint_path = root / "SAC_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            tensorboard_path.write_text(
                json.dumps(
                    self._tensorboard_summary(
                        avg_reward_delta=0.1,
                        psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
                    )
                ),
                encoding="utf-8",
            )
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")
            previous_eval = self._previous_eval()
            previous_eval["target_mode"] = "legacy-coscos"
            previous_eval_path.write_text(json.dumps(previous_eval), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--previous-eval-json",
                    str(previous_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("target_mode", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_decide_stage4_ogm_gate_cli_rejects_tensorboard_log_dir_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_run_dir = root / "current_run"
            stale_run_dir = root / "stale_run"
            current_run_dir.mkdir()
            stale_run_dir.mkdir()
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            checkpoint_path = current_run_dir / "SAC_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            tensorboard_summary = self._tensorboard_summary(
                avg_reward_delta=0.1,
                psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
            )
            tensorboard_summary["log_dir"] = str(stale_run_dir)
            tensorboard_path.write_text(json.dumps(tensorboard_summary), encoding="utf-8")
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("TensorBoard run dir mismatch", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_decide_stage4_ogm_gate_cli_rejects_tensorboard_log_and_event_dir_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_run_dir = root / "current_run"
            stale_run_dir = root / "stale_run"
            current_run_dir.mkdir()
            stale_run_dir.mkdir()
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            checkpoint_path = current_run_dir / "SAC_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            tensorboard_summary = self._tensorboard_summary(
                avg_reward_delta=0.1,
                psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
            )
            tensorboard_summary["log_dir"] = str(current_run_dir)
            tensorboard_summary["event_dir"] = str(stale_run_dir)
            tensorboard_path.write_text(json.dumps(tensorboard_summary), encoding="utf-8")
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("TensorBoard run dir mismatch", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_decide_stage4_ogm_gate_cli_requires_tensorboard_provenance_for_full_state_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            checkpoint_path = root / "SAC_79999.pt"
            state_path = root / "stage4_state_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            state_path.write_text("state-placeholder", encoding="utf-8")
            tensorboard_path.write_text(
                json.dumps(
                    self._tensorboard_summary(
                        avg_reward_delta=0.1,
                        psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
                    )
                ),
                encoding="utf-8",
            )
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--state-snapshot",
                    str(state_path),
                    "--state-snapshot-required",
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("TensorBoard run dir mismatch", completed.stderr)
        self.assertIn("missing provenance", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_decide_stage4_ogm_gate_cli_requires_event_dir_for_full_state_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            checkpoint_path = root / "SAC_79999.pt"
            state_path = root / "stage4_state_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            state_path.write_text("state-placeholder", encoding="utf-8")
            tensorboard_summary = self._tensorboard_summary(
                avg_reward_delta=0.1,
                psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
            )
            tensorboard_summary["log_dir"] = str(root)
            tensorboard_path.write_text(json.dumps(tensorboard_summary), encoding="utf-8")
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--state-snapshot",
                    str(state_path),
                    "--state-snapshot-required",
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("TensorBoard run dir mismatch", completed.stderr)
        self.assertIn("missing provenance", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_decide_stage4_ogm_gate_cli_requires_log_dir_for_full_state_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            checkpoint_path = root / "SAC_79999.pt"
            state_path = root / "stage4_state_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            state_path.write_text("state-placeholder", encoding="utf-8")
            tensorboard_summary = self._tensorboard_summary(
                avg_reward_delta=0.1,
                psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
            )
            tensorboard_summary["event_dir"] = str(root)
            tensorboard_path.write_text(json.dumps(tensorboard_summary), encoding="utf-8")
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--state-snapshot",
                    str(state_path),
                    "--state-snapshot-required",
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("TensorBoard run dir mismatch", completed.stderr)
        self.assertIn("missing provenance", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_decide_stage4_ogm_gate_cli_resolves_repo_relative_provenance_from_non_root_cwd(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix="stage4_gate_test_", dir=repo_root) as temp_dir:
            run_dir = Path(temp_dir)
            rel_run_dir = run_dir.relative_to(repo_root)
            tensorboard_path = run_dir / "gate_80000.tensorboard.json"
            current_eval_path = run_dir / "current.eval.json"
            checkpoint_rel_path = rel_run_dir / "SAC_79999.pt"
            state_snapshot_rel_path = rel_run_dir / "stage4_state_79999.pt"
            checkpoint_path = repo_root / checkpoint_rel_path
            state_snapshot_path = repo_root / state_snapshot_rel_path
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            state_snapshot_path.write_text("state-placeholder", encoding="utf-8")
            tensorboard_summary = self._tensorboard_summary(
                avg_reward_delta=0.1,
                psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
            )
            tensorboard_summary["log_dir"] = str(rel_run_dir)
            tensorboard_summary["event_dir"] = str(rel_run_dir)
            tensorboard_path.write_text(json.dumps(tensorboard_summary), encoding="utf-8")
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "tools/stage4/decide_stage4_ogm_gate.py"),
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--checkpoint",
                    str(checkpoint_rel_path),
                    "--state-snapshot",
                    str(state_snapshot_rel_path),
                    "--state-snapshot-required",
                ],
                cwd=repo_root / "tools",
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["checkpoint_path"], str(checkpoint_rel_path))
        self.assertTrue(payload["current"]["checkpoint_exists"])
        self.assertEqual(payload["state_snapshot_path"], str(state_snapshot_rel_path))
        self.assertTrue(payload["current"]["state_snapshot_exists"])
        self.assertNotIn("missing_checkpoint", payload["decision"]["reasons"])
        self.assertNotIn("missing_state_snapshot", payload["decision"]["reasons"])

    def test_decide_stage4_ogm_gate_cli_rejects_previous_eval_missing_summary_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tensorboard_path = root / "gate_80000.tensorboard.json"
            current_eval_path = root / "current.eval.json"
            previous_eval_path = root / "previous.eval.json"
            checkpoint_path = root / "SAC_79999.pt"
            checkpoint_path.write_text("checkpoint-placeholder", encoding="utf-8")
            tensorboard_path.write_text(
                json.dumps(
                    self._tensorboard_summary(
                        avg_reward_delta=0.1,
                        psr_deltas={"success_rate_Normal": 0.1, "success_rate_Complex": 0.1},
                    )
                ),
                encoding="utf-8",
            )
            current_eval_path.write_text(json.dumps(self._current_eval()), encoding="utf-8")
            previous_eval_path.write_text(json.dumps({"target_mode": "corrected-cossin"}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/decide_stage4_ogm_gate.py",
                    "--episode",
                    "80000",
                    "--tensorboard-json",
                    str(tensorboard_path),
                    "--current-eval-json",
                    str(current_eval_path),
                    "--previous-eval-json",
                    str(previous_eval_path),
                    "--checkpoint",
                    str(checkpoint_path),
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("previous eval JSON missing summary", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_hard_rejects_nonfinite_metrics(self) -> None:
        current = {
            "episode": 30000,
            "has_nonfinite": True,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 1.0, "mean_psr_delta": 1.0, "step_num_improvement": 1.0},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("nonfinite_tensorboard_or_eval_metric", decision["reasons"])

    def test_hard_rejects_missing_checkpoint(self) -> None:
        current = {
            "episode": 30000,
            "has_nonfinite": False,
            "checkpoint_exists": False,
            "trend": {"avg_reward_delta": 1.0, "mean_psr_delta": 1.0, "step_num_improvement": 1.0},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("missing_checkpoint", decision["reasons"])

    def test_hard_rejects_missing_required_state_snapshot(self) -> None:
        current = {
            "episode": 80000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "state_snapshot_required": True,
            "state_snapshot_exists": False,
            "trend": {"avg_reward_delta": 1.0, "mean_psr_delta": 1.0, "step_num_improvement": 1.0},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("missing_state_snapshot", decision["reasons"])

    def test_hard_rejects_provided_missing_state_snapshot_without_required_flag(self) -> None:
        current = {
            "episode": 80000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "state_snapshot_exists": False,
            "trend": {"avg_reward_delta": 1.0, "mean_psr_delta": 1.0, "step_num_improvement": 1.0},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("missing_state_snapshot", decision["reasons"])

    def test_state_snapshot_is_backward_compatible_when_no_state_info_is_provided(self) -> None:
        current = {
            "episode": 80000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 1.0, "mean_psr_delta": 1.0, "step_num_improvement": 1.0},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "continue")

    def test_20k_gate_allows_weaker_than_hope_baseline_when_trend_improves(self) -> None:
        current = {
            "episode": 20000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 0.08, "mean_psr_delta": 0.05, "step_num_improvement": 0.08},
            "summary": {"Sim-Normal": {"psr": 0.70, "angs": 3.0, "pl": 35.0}},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "continue")
        self.assertIn("early_ogm_can_lag_hope_20k_baseline", decision["notes"])

    def test_weak_20k_trend_gets_one_10k_grace_window(self) -> None:
        current = {
            "episode": 20000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 0.0, "mean_psr_delta": 0.01, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.40, "angs": 6.0, "pl": 55.0}},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "grace-10k")
        self.assertIn("weak_20k_trend", decision["reasons"])

    def test_19999_episode_uses_first_gate_policy(self) -> None:
        current = {
            "episode": 19999,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 0.08, "mean_psr_delta": 0.05, "step_num_improvement": 0.08},
            "summary": {"Sim-Normal": {"psr": 0.70, "angs": 3.0, "pl": 35.0}},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "continue")
        self.assertIn("early_ogm_can_lag_hope_20k_baseline", decision["notes"])

    def test_no_success_summary_values_do_not_crash_progress_gate(self) -> None:
        current = {
            "episode": 20000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 0.08, "mean_psr_delta": 0.05, "step_num_improvement": 0.08},
            "summary": {"Sim-Extreme": {"psr": 0.0, "angs": None, "pl": None}},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "continue")

    def test_first_stagnant_gate_gets_one_10k_grace_window(self) -> None:
        current = {
            "episode": 30000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": -0.01, "mean_psr_delta": 0.0, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}},
        }
        previous = {"episode": 20000, "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}}}
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "grace-10k")

    def test_second_consecutive_stagnant_gate_stops_before_100k(self) -> None:
        current = {
            "episode": 40000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": -0.02, "mean_psr_delta": 0.0, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.2, "pl": 46.0}},
        }
        previous = {"episode": 30000, "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}}}
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("second_consecutive_bad_progress_gate", decision["reasons"])

    def test_fixed_eval_path_quality_recovery_can_clear_grace_window(self) -> None:
        previous = {
            "episode": 50000,
            "summary": {
                "Sim-Normal": {"psr": 1.0, "angs": 1.88, "pl": 13.569036900165345},
                "Sim-Complex": {"psr": 1.0, "angs": 15.4, "pl": 21.395820940693326},
            },
        }
        current = {
            "episode": 60000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {
                "avg_reward_delta": 0.012659,
                "mean_psr_delta": 0.0,
                "step_num_improvement": -0.158945,
            },
            "summary": {
                "Sim-Normal": {"psr": 1.0, "angs": 2.08, "pl": 13.759971948042594},
                "Sim-Complex": {"psr": 1.0, "angs": 9.45, "pl": 16.59542645844011},
            },
        }
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)
        self.assertEqual(decision["decision"], "continue")
        self.assertIn("fixed_eval_quality_improved", decision["notes"])

    def test_second_grace_window_still_stops_when_fixed_eval_quality_regresses(self) -> None:
        previous = {
            "episode": 50000,
            "summary": {
                "Sim-Normal": {"psr": 1.0, "angs": 1.88, "pl": 13.57},
                "Sim-Complex": {"psr": 1.0, "angs": 15.4, "pl": 21.4},
            },
        }
        current = {
            "episode": 60000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {
                "avg_reward_delta": 0.0,
                "mean_psr_delta": 0.0,
                "step_num_improvement": -0.1,
            },
            "summary": {
                "Sim-Normal": {"psr": 1.0, "angs": 2.2, "pl": 14.5},
                "Sim-Complex": {"psr": 1.0, "angs": 18.0, "pl": 25.0},
            },
        }
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("second_consecutive_bad_progress_gate", decision["reasons"])

    def test_fixed_eval_quality_does_not_clear_gate_when_psr_regresses(self) -> None:
        previous = {
            "episode": 50000,
            "summary": {
                "Sim-Normal": {"psr": 1.0, "angs": 1.88, "pl": 13.57},
                "Sim-Complex": {"psr": 1.0, "angs": 15.4, "pl": 21.4},
            },
        }
        current = {
            "episode": 60000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {
                "avg_reward_delta": 0.0,
                "mean_psr_delta": 0.0,
                "step_num_improvement": -0.1,
            },
            "summary": {
                "Sim-Normal": {"psr": 1.0, "angs": 1.5, "pl": 12.0},
                "Sim-Complex": {"psr": 0.98, "angs": 9.0, "pl": 16.0},
            },
        }
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("second_consecutive_bad_progress_gate", decision["reasons"])

    def test_recovery_restarts_when_observation_or_rasterizer_changes(self) -> None:
        self.assertEqual(recommend_recovery_action(["ogm_rasterizer_changed"]), "restart-from-scratch")
        self.assertEqual(recommend_recovery_action(["reward_logging_only"]), "resume-from-checkpoint")

    def test_recovery_restarts_for_known_semantic_aliases(self) -> None:
        for reason in (
            "action_semantics_changed",
            "network_input_shape_changed",
            "replay_meaning_changed",
            "state_normalization_changed",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(recommend_recovery_action([reason]), "restart-from-scratch")

    def test_recovery_sends_unknown_reasons_to_manual_review(self) -> None:
        self.assertEqual(recommend_recovery_action(["new_unclassified_training_fix"]), "manual-review")

    def test_recovery_resumes_for_known_logging_eval_and_report_reasons(self) -> None:
        for reason in (
            "reward_logging_only",
            "monitoring_only",
            "eval_parser_fix",
            "report_format_only",
            "resource_monitor_fix",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(recommend_recovery_action([reason]), "resume-from-checkpoint")


if __name__ == "__main__":
    unittest.main()
