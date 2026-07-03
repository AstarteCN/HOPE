from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from shapely.geometry import LinearRing, Polygon, box
from shapely.prepared import prep


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_positive_integer(name: str, value: object) -> None:
    if not _is_integer(value) or int(value) <= 0:
        raise ValueError(f"OGMConfig.{name} must be a positive integer.")


def _validate_channel(name: str, value: object, channels: int) -> None:
    if not _is_integer(value) or not 0 <= int(value) < channels:
        raise ValueError(f"OGMConfig.{name} must be an integer in [0, channels).")


def _validate_finite_float(name: str, value: object) -> None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OGMConfig.{name} must be finite.") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"OGMConfig.{name} must be finite.")


@dataclass(frozen=True)
class OGMConfig:
    size: int = 64
    resolution: float = 1.0 / 3.0
    channels: int = 2
    obstacle_channel: int = 0
    target_channel: int = 1
    obstacle_value: float = 1.0
    target_value: float = 1.0
    dtype: type = np.float32

    def __post_init__(self) -> None:
        _validate_positive_integer("size", self.size)
        _validate_positive_integer("channels", self.channels)
        _validate_finite_float("resolution", self.resolution)
        if float(self.resolution) <= 0:
            raise ValueError("OGMConfig.resolution must be positive.")
        channels = int(self.channels)
        _validate_channel("obstacle_channel", self.obstacle_channel, channels)
        _validate_channel("target_channel", self.target_channel, channels)
        if int(self.obstacle_channel) == int(self.target_channel):
            raise ValueError("OGMConfig obstacle and target channels must be distinct.")
        _validate_finite_float("obstacle_value", self.obstacle_value)
        _validate_finite_float("target_value", self.target_value)
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("OGMConfig.dtype must resolve to a floating numeric dtype.") from exc
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("OGMConfig.dtype must resolve to a floating numeric dtype.")


DEFAULT_OGM_CONFIG = OGMConfig()


def _ring_from_object(item: object) -> LinearRing:
    if isinstance(item, LinearRing):
        return item
    shape = getattr(item, "shape", None)
    if isinstance(shape, LinearRing):
        return shape
    raise TypeError(f"Unsupported OGM geometry object: {type(item)!r}")


def world_to_local(ego_state: object, x: float, y: float) -> tuple[float, float]:
    ego_x = float(ego_state.loc.x)
    ego_y = float(ego_state.loc.y)
    heading = float(ego_state.heading)
    dx = float(x) - ego_x
    dy = float(y) - ego_y
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return float(local_x), float(local_y)


def _snap_near_integer(value: float) -> float:
    rounded = np.round(value)
    if np.isclose(value, rounded, atol=1.0e-9):
        return float(rounded)
    return float(value)


def world_to_grid(
    ego_state: object,
    x: float,
    y: float,
    config: OGMConfig = DEFAULT_OGM_CONFIG,
) -> tuple[int, int]:
    local_x, local_y = world_to_local(ego_state, x, y)
    center = config.size // 2
    col_value = _snap_near_integer(local_x / config.resolution + center)
    row_value = _snap_near_integer(center - local_y / config.resolution)
    col = int(np.floor(col_value))
    row = int(np.floor(row_value))
    row = int(np.clip(row, 0, config.size - 1))
    col = int(np.clip(col, 0, config.size - 1))
    return row, col


def _local_to_world(ego_state: object, local_x: float, local_y: float) -> tuple[float, float]:
    heading = float(ego_state.heading)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    world_x = float(ego_state.loc.x) + cos_h * local_x - sin_h * local_y
    world_y = float(ego_state.loc.y) + sin_h * local_x + cos_h * local_y
    return float(world_x), float(world_y)


def _cell_center_world(ego_state: object, row: int, col: int, config: OGMConfig) -> tuple[float, float]:
    center = float(config.size) / 2.0
    local_x = (float(col) + 0.5 - center) * float(config.resolution)
    local_y = (center - float(row) - 0.5) * float(config.resolution)
    return _local_to_world(ego_state, local_x, local_y)


def _cell_footprint_world(ego_state: object, row: int, col: int, config: OGMConfig) -> Polygon:
    center = float(config.size) / 2.0
    resolution = float(config.resolution)
    left = (float(col) - center) * resolution
    right = (float(col) + 1.0 - center) * resolution
    top = (center - float(row)) * resolution
    bottom = (center - float(row) - 1.0) * resolution
    corners = [
        _local_to_world(ego_state, left, bottom),
        _local_to_world(ego_state, right, bottom),
        _local_to_world(ego_state, right, top),
        _local_to_world(ego_state, left, top),
    ]
    return Polygon(corners)


@lru_cache(maxsize=16)
def _cached_cell_footprints(size: int, resolution: float) -> tuple[tuple[Polygon, ...], ...]:
    center = float(size) / 2.0
    resolution = float(resolution)
    rows: list[tuple[Polygon, ...]] = []
    for row in range(size):
        top = (center - float(row)) * resolution
        bottom = (center - float(row) - 1.0) * resolution
        cells = []
        for col in range(size):
            left = (float(col) - center) * resolution
            right = (float(col) + 1.0 - center) * resolution
            cells.append(box(left, bottom, right, top))
        rows.append(tuple(cells))
    return tuple(rows)


def _local_candidate_bounds(local_coords: np.ndarray, config: OGMConfig) -> tuple[int, int, int, int]:
    center = float(config.size) / 2.0
    resolution = float(config.resolution)
    cols = local_coords[:, 0] / resolution + center
    rows = center - local_coords[:, 1] / resolution
    padding = 2
    min_row = max(int(np.floor(float(np.min(rows)))) - padding, 0)
    max_row = min(int(np.floor(float(np.max(rows)))) + padding, config.size - 1)
    min_col = max(int(np.floor(float(np.min(cols)))) - padding, 0)
    max_col = min(int(np.floor(float(np.max(cols)))) + padding, config.size - 1)
    return min_row, max_row, min_col, max_col


def _rasterize_ring(
    grid: np.ndarray,
    ego_state: object,
    ring: LinearRing,
    channel: int,
    value: float,
    config: OGMConfig,
) -> None:
    local_coords = np.asarray(
        [world_to_local(ego_state, x, y) for x, y in ring.coords],
        dtype=float,
    )
    local_polygon = Polygon(local_coords)
    if local_polygon.is_empty:
        return
    prepared = prep(local_polygon)
    min_row, max_row, min_col, max_col = _local_candidate_bounds(local_coords, config)
    cell_footprints = _cached_cell_footprints(int(config.size), float(config.resolution))
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            cell = cell_footprints[row][col]
            if prepared.intersects(cell):
                grid[row, col, channel] = value


def build_ego_ogm(
    ego_state: object,
    obstacles: Iterable[object],
    target_box: LinearRing,
    config: OGMConfig = DEFAULT_OGM_CONFIG,
) -> np.ndarray:
    grid = np.zeros((config.size, config.size, config.channels), dtype=config.dtype)
    for obstacle in obstacles:
        _rasterize_ring(
            grid=grid,
            ego_state=ego_state,
            ring=_ring_from_object(obstacle),
            channel=config.obstacle_channel,
            value=config.obstacle_value,
            config=config,
        )
    _rasterize_ring(
        grid=grid,
        ego_state=ego_state,
        ring=_ring_from_object(target_box),
        channel=config.target_channel,
        value=config.target_value,
        config=config,
    )
    return grid
