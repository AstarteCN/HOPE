from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TARGET_MODE_LEGACY_COSCOS = "legacy-coscos"
TARGET_MODE_CORRECTED_COSSIN = "corrected-cossin"
TARGET_MODE_AUTO = "auto"
TARGET_MODES = (TARGET_MODE_LEGACY_COSCOS, TARGET_MODE_CORRECTED_COSSIN)
TARGET_MODE_REQUEST_CHOICES = (TARGET_MODE_AUTO,) + TARGET_MODES


def validate_target_mode(target_mode: str) -> str:
    if target_mode not in TARGET_MODES:
        raise ValueError("target_mode must be one of: %s" % ", ".join(TARGET_MODES))
    return target_mode


def validate_requested_target_mode(target_mode: str) -> str:
    if target_mode not in TARGET_MODE_REQUEST_CHOICES:
        raise ValueError(
            "target_mode must be one of: %s" % ", ".join(TARGET_MODE_REQUEST_CHOICES)
        )
    return target_mode


def _load_state_target_mode(state_path: Path) -> str | None:
    if not state_path.exists():
        return None
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return validate_target_mode(state.get("target_mode", TARGET_MODE_LEGACY_COSCOS))


def _candidate_state_paths_for_checkpoint(checkpoint_path: Path) -> list[Path]:
    candidates: list[Path] = []
    match = re.fullmatch(r"SAC_(\d+)\.pt", checkpoint_path.name)
    if match:
        candidates.append(checkpoint_path.with_name("stage4_state_%s.pt" % match.group(1)))
    candidates.append(checkpoint_path.with_name("stage4_state_latest.pt"))
    return candidates


def resolve_checkpoint_target_mode(
    checkpoint_path: str | Path,
    requested_target_mode: str = TARGET_MODE_AUTO,
) -> tuple[str, dict[str, str]]:
    requested = validate_requested_target_mode(requested_target_mode)
    if requested != TARGET_MODE_AUTO:
        return validate_target_mode(requested), {"type": "explicit", "value": requested}

    checkpoint = Path(checkpoint_path)
    for state_path in _candidate_state_paths_for_checkpoint(checkpoint):
        mode = _load_state_target_mode(state_path)
        if mode is not None:
            return mode, {"type": "stage4_state", "path": str(state_path)}
    return TARGET_MODE_LEGACY_COSCOS, {"type": "legacy_default"}


def relative_dest_heading(raw_env: Any) -> float:
    return float(raw_env.map.dest.heading) - float(raw_env.vehicle.state.heading)


def apply_target_mode(
    target: np.ndarray,
    target_mode: str,
    rel_dest_heading: float | None = None,
) -> np.ndarray:
    mode = validate_target_mode(target_mode)
    transformed = np.asarray(target, dtype=np.float64).copy()
    if transformed.shape != (5,):
        raise ValueError("target observation must have shape (5,), got %r" % (transformed.shape,))
    if mode == TARGET_MODE_LEGACY_COSCOS:
        return transformed
    if rel_dest_heading is None:
        raise ValueError("rel_dest_heading is required for corrected-cossin target mode.")
    transformed[3] = math.cos(float(rel_dest_heading))
    transformed[4] = math.sin(float(rel_dest_heading))
    return transformed


def build_stage4_observation_func(raw_env: Any, target_mode: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    from env.env_wrapper import observation_rescale

    mode = validate_target_mode(target_mode)

    def _stage4_observation_func(obs: dict[str, Any]) -> dict[str, Any]:
        scaled = observation_rescale(obs)
        scaled["target"] = apply_target_mode(
            scaled["target"],
            mode,
            relative_dest_heading(raw_env) if mode == TARGET_MODE_CORRECTED_COSSIN else None,
        )
        return scaled

    return _stage4_observation_func
