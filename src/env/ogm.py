from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import LinearRing, Point, Polygon
from shapely.prepared import prep


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
        if self.size <= 0:
            raise ValueError("OGMConfig.size must be positive.")
        if self.resolution <= 0:
            raise ValueError("OGMConfig.resolution must be positive.")
        if self.channels < 2:
            raise ValueError("OGMConfig.channels must be at least 2.")


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


def _cell_center_world(ego_state: object, row: int, col: int, config: OGMConfig) -> tuple[float, float]:
    center = config.size / 2.0
    local_x = (float(col) + 0.5 - center) * config.resolution
    local_y = (center - float(row) - 0.5) * config.resolution
    heading = float(ego_state.heading)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    world_x = float(ego_state.loc.x) + cos_h * local_x - sin_h * local_y
    world_y = float(ego_state.loc.y) + sin_h * local_x + cos_h * local_y
    return float(world_x), float(world_y)


def _rasterize_ring(
    grid: np.ndarray,
    ego_state: object,
    ring: LinearRing,
    channel: int,
    value: float,
    config: OGMConfig,
) -> None:
    polygon = Polygon(ring)
    if polygon.is_empty:
        return
    prepared = prep(polygon)
    coords = np.asarray(ring.coords, dtype=float)
    grid_points = [world_to_grid(ego_state, x, y, config) for x, y in coords]
    rows = [p[0] for p in grid_points]
    cols = [p[1] for p in grid_points]
    min_row = max(min(rows) - 2, 0)
    max_row = min(max(rows) + 2, config.size - 1)
    min_col = max(min(cols) - 2, 0)
    max_col = min(max(cols) + 2, config.size - 1)
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            world_x, world_y = _cell_center_world(ego_state, row, col, config)
            point = Point(world_x, world_y)
            if prepared.contains(point) or polygon.touches(point):
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
