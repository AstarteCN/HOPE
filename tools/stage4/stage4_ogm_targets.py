from __future__ import annotations

from typing import Mapping

RELATIVE_TOLERANCE = 0.03
TRAIN_EPISODE_MIN = 80_000
TRAIN_EPISODE_TARGET = 100_000
TRAIN_EPISODE_MAX = 120_000
FIRST_GATE_EPISODE = 20_000
PROGRESS_GATE_INTERVAL = 10_000

OGM_SIM_TARGETS: dict[str, dict[str, float]] = {
    "Sim-Normal": {"psr": 0.9933, "angs": 1.5, "pl": 20.3},
    "Sim-Complex": {"psr": 0.977, "angs": 1.9, "pl": 23.6},
}

BASELINE_FAST_ACTION_MASK_20K: dict[str, object] = {
    "id": "hope-fast-action-mask-20k",
    "run_dir": "src/log/exp/sac_20260620_085208",
    "checkpoint": "SAC_19999.pt",
    "wall_time_hours": 10.947863,
    "episodes_per_hour": 1826.840564,
    "env_steps_per_second": 46.223481,
    "eval_success": {
        "Normal": 0.985,
        "Complex": 0.945,
        "Extrem": 0.655,
        "DLP": 0.960,
    },
}


def relative_band(target: float, tolerance: float = RELATIVE_TOLERANCE) -> tuple[float, float]:
    return target * (1.0 - tolerance), target * (1.0 + tolerance)


def metric_within_relative_band(value: float, target: float, tolerance: float = RELATIVE_TOLERANCE) -> bool:
    low, high = relative_band(target, tolerance)
    return low <= value <= high


def all_ogm_targets_met(results: Mapping[str, Mapping[str, float]]) -> bool:
    for split, targets in OGM_SIM_TARGETS.items():
        observed = results.get(split, {})
        for metric, target in targets.items():
            if metric not in observed:
                return False
            if not metric_within_relative_band(float(observed[metric]), target):
                return False
    return True
