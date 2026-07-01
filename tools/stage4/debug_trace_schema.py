from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs import (  # noqa: E402
    FRONT_HANG,
    LENGTH,
    LIDAR_NUM,
    OGM_CHANNELS,
    OGM_RESOLUTION,
    OGM_SIZE,
    REAR_HANG,
    VALID_SPEED,
    VALID_STEER,
    WHEEL_BASE,
    WIDTH,
)


SCHEMA_VERSION = "stage4-ogm-debug-v1"
OGM_CHANNEL_NAMES = ("obstacle", "target")


def vehicle_params_from_configs() -> dict[str, float]:
    return {
        "length_m": float(LENGTH),
        "width_m": float(WIDTH),
        "wheelbase_m": float(WHEEL_BASE),
        "front_hang_m": float(FRONT_HANG),
        "rear_hang_m": float(REAR_HANG),
        "max_velocity_mps": float(VALID_SPEED[1]),
        "max_steer_rad": float(VALID_STEER[1]),
    }


def vehicle_body_corners_local(params: dict[str, float]) -> list[list[float]]:
    half_width = params["width_m"] / 2.0
    return [
        [-params["rear_hang_m"], -half_width],
        [params["wheelbase_m"] + params["front_hang_m"], -half_width],
        [params["wheelbase_m"] + params["front_hang_m"], half_width],
        [-params["rear_hang_m"], half_width],
    ]


def ogm_default_grid() -> dict[str, object]:
    return {
        "size": int(OGM_SIZE),
        "resolution_m": float(OGM_RESOLUTION),
        "origin": "rear_axle_center",
        "x_forward": True,
        "y_left": True,
        "channels": list(OGM_CHANNEL_NAMES[: int(OGM_CHANNELS)]),
    }


def default_layer_groups() -> dict[str, list[str]]:
    return {
        "environment_model": [
            "map_bounds",
            "obstacle_polygons",
            "target_slot",
            "ogm_obstacle",
            "ogm_target",
        ],
        "perception": [
            "lidar_rays",
            "lidar_hit_points",
            "minimum_distance_marker",
            "action_mask_feasible_sectors",
        ],
        "vehicle_model": [
            "body",
            "front_wheels",
            "rear_wheels",
            "front_axle",
            "rear_axle",
            "rear_axle_center",
            "heading_arrow",
        ],
        "trajectory_control": [
            "rl_trajectory",
            "rl_waypoints",
            "rs_trajectory",
            "rs_waypoints",
            "current_step_marker",
        ],
        "debug_annotations": [
            "executed_action_vector",
            "terminal_status_marker",
            "collision_outbound_marker",
            "reward_label",
            "case_metadata",
        ],
    }


def densify_polyline(points: Sequence[Sequence[float]], max_spacing_m: float = 0.4) -> list[list[float]]:
    if max_spacing_m <= 0:
        raise ValueError("max_spacing_m must be positive.")
    if len(points) <= 1:
        return [list(point) for point in points]

    dense: list[list[float]] = [list(points[0])]
    for start, end in zip(points, points[1:]):
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end_arr[:2] - start_arr[:2]))
        segments = max(1, int(math.ceil(distance / max_spacing_m)))
        for idx in range(1, segments + 1):
            ratio = idx / segments
            interpolated = start_arr + (end_arr - start_arr) * ratio
            dense.append([float(value) for value in interpolated.tolist()])
    return dense


def project_lidar_hits(
    distances_m: Iterable[float],
    vehicle_boundary_m: Iterable[float] | None = None,
) -> list[list[float]]:
    distances = np.asarray(list(distances_m), dtype=float)
    if vehicle_boundary_m is not None:
        distances = distances + np.asarray(list(vehicle_boundary_m), dtype=float)
    angles = np.arange(len(distances), dtype=float) * (2.0 * math.pi / float(LIDAR_NUM))
    points = np.stack([distances * np.cos(angles), distances * np.sin(angles)], axis=1)
    return [[float(x), float(y)] for x, y in points]


def _as_channel_first(ogm: np.ndarray) -> np.ndarray:
    array = np.asarray(ogm)
    if array.ndim != 3:
        raise ValueError("OGM must be a 3D array.")
    expected_channels = len(OGM_CHANNEL_NAMES)
    if array.shape[0] == expected_channels:
        return array
    if array.shape[-1] == expected_channels:
        return np.moveaxis(array, -1, 0)
    raise ValueError("OGM array must be channel-first or channel-last with known channels.")


def sparsify_ogm(ogm: np.ndarray, max_cells: int | None = None) -> list[dict[str, int | float | str]]:
    array = _as_channel_first(ogm)
    cells: list[dict[str, int | float | str]] = []
    for channel_index, channel_name in enumerate(OGM_CHANNEL_NAMES[: array.shape[0]]):
        rows, cols = np.nonzero(array[channel_index])
        for row, col in zip(rows.tolist(), cols.tolist()):
            cells.append(
                {
                    "channel": channel_name,
                    "row": int(row),
                    "col": int(col),
                    "value": float(array[channel_index, row, col]),
                }
            )
            if max_cells is not None and len(cells) >= max_cells:
                return cells
    return cells
