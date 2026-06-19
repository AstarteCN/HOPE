from __future__ import annotations

from statistics import mean
from typing import Mapping


SCENES = ("Normal", "Complex", "Extrem", "DLP")

PROTECTED_SOURCE_PATHS = (
    "src/train",
    "src/env",
    "src/model",
)

BASELINE_ID = "stage3_command_only_20k_20260619"

BASELINE_20K = {
    "id": BASELINE_ID,
    "run_dir": r"D:\Github\HOPE\src\log\exp\sac_20260619_004316",
    "checkpoint": r"D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt",
    "stop_episode_snapshot": 20038,
    "env_steps": 1824904,
    "wall_time_hours": 12.964,
    "episodes_per_hour": 1545.71,
    "env_steps_per_second": 39.10,
    "resource_profile": {
        "avg_process_cpu_percent": 37.76,
        "avg_whole_gpu_util_percent": 29.42,
    },
    "tensorboard": {
        "avg_reward": {"latest": 0.076828, "recent_mean": 0.061052},
        "actor_loss": {"latest": -0.471469, "recent_mean": -0.543129},
        "critic_loss": {"latest": 0.080817, "recent_mean": 0.103880},
        "success_rate_Normal": {"latest": 1.0, "recent_mean": 1.0},
        "success_rate_Complex": {"latest": 0.94, "recent_mean": 0.94},
        "success_rate_Extrem": {"latest": 0.69, "recent_mean": 0.7108},
        "success_rate_dlp": {"latest": 0.82, "recent_mean": 0.7924},
        "step_num": {"latest": 52.0, "recent_mean": 77.22},
    },
    "external_eval": {
        "Normal": 0.985,
        "Complex": 0.945,
        "Extrem": 0.655,
        "DLP": 0.960,
        "mean": 0.88625,
    },
}

QUALITY_THRESHOLDS = {
    "Normal": 0.95,
    "Complex": 0.90,
    "Extrem": 0.58,
    "DLP": 0.91,
    "mean": 0.85625,
}

SPEED_MIN_IMPROVEMENT_RATIO = 0.10


def scene_mean(eval_success: Mapping[str, float]) -> float:
    missing = [scene for scene in SCENES if scene not in eval_success]
    if missing:
        raise ValueError(f"missing scene results: {', '.join(missing)}")
    return mean(float(eval_success[scene]) for scene in SCENES)


def quality_gate_status(
    *,
    eval_success: Mapping[str, float],
    tensorboard_has_nonfinite: bool,
    multi_scene_collapse: bool,
    checkpoint_missing: bool,
    parity_failed: bool,
) -> dict:
    hard_reject_reasons: list[str] = []
    if checkpoint_missing:
        hard_reject_reasons.append("missing_20k_checkpoint")
    if tensorboard_has_nonfinite:
        hard_reject_reasons.append("nonfinite_tensorboard_metric")
    if multi_scene_collapse:
        hard_reject_reasons.append("multi_scene_collapse")
    if parity_failed:
        hard_reject_reasons.append("parity_failed")

    per_scene = {
        scene: float(eval_success.get(scene, -1.0)) >= QUALITY_THRESHOLDS[scene]
        for scene in SCENES
    }
    mean_success = scene_mean(eval_success) if all(scene in eval_success for scene in SCENES) else -1.0
    mean_pass = mean_success >= QUALITY_THRESHOLDS["mean"]
    quality_pass = not hard_reject_reasons and all(per_scene.values()) and mean_pass

    return {
        "quality_pass": quality_pass,
        "mean_success": mean_success,
        "mean_pass": mean_pass,
        "per_scene_pass": per_scene,
        "hard_reject_reasons": hard_reject_reasons,
    }


def speed_gate_passed(candidate_metrics: Mapping[str, float]) -> dict:
    baseline_hours = float(BASELINE_20K["wall_time_hours"])
    baseline_eps_hour = float(BASELINE_20K["episodes_per_hour"])
    baseline_env_steps_sec = float(BASELINE_20K["env_steps_per_second"])

    wall_time_hours = float(candidate_metrics.get("wall_time_hours", baseline_hours))
    episodes_per_hour = float(candidate_metrics.get("episodes_per_hour", baseline_eps_hour))
    env_steps_per_second = float(candidate_metrics.get("env_steps_per_second", baseline_env_steps_sec))

    wall_time_improvement = (baseline_hours - wall_time_hours) / baseline_hours
    eps_hour_improvement = (episodes_per_hour - baseline_eps_hour) / baseline_eps_hour
    env_steps_improvement = (env_steps_per_second - baseline_env_steps_sec) / baseline_env_steps_sec

    metrics = {
        "wall_time_improvement_ratio": wall_time_improvement,
        "episodes_per_hour_improvement_ratio": eps_hour_improvement,
        "env_steps_per_second_improvement_ratio": env_steps_improvement,
    }
    speed_pass = any(value >= SPEED_MIN_IMPROVEMENT_RATIO for value in metrics.values())
    return {"speed_pass": speed_pass, **metrics}
