"""Offline numeric helpers for comparing Stage 4 trajectory style."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _xy_array(points: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError("points must be a non-empty Nx2 sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite coordinates")
    return array


def resample_polyline(points: Iterable[Iterable[float]], sample_count: int = 64) -> list[list[float]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    array = _xy_array(points)
    if sample_count == 1:
        return [array[0].tolist()]
    if len(array) == 1:
        return np.repeat(array, sample_count, axis=0).tolist()

    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length == 0.0:
        sampled = np.repeat(array[:1], sample_count, axis=0)
    else:
        targets = np.linspace(0.0, total_length, sample_count)
        sampled = np.column_stack(
            [
                np.interp(targets, cumulative, array[:, 0]),
                np.interp(targets, cumulative, array[:, 1]),
            ]
        )

    sampled[0] = array[0]
    sampled[-1] = array[-1]
    return sampled.tolist()


def mean_l2_distance(
    first: Iterable[Iterable[float]],
    second: Iterable[Iterable[float]],
    sample_count: int = 64,
) -> float:
    first_sampled = np.asarray(resample_polyline(first, sample_count), dtype=float)
    second_sampled = np.asarray(resample_polyline(second, sample_count), dtype=float)
    return float(np.mean(np.linalg.norm(first_sampled - second_sampled, axis=1)))


def hausdorff_distance(
    first: Iterable[Iterable[float]],
    second: Iterable[Iterable[float]],
    sample_count: int = 64,
) -> float:
    first_array = np.asarray(resample_polyline(first, sample_count), dtype=float)
    second_array = np.asarray(resample_polyline(second, sample_count), dtype=float)
    distances = np.linalg.norm(first_array[:, None, :] - second_array[None, :, :], axis=2)
    first_to_second = float(np.max(np.min(distances, axis=1)))
    second_to_first = float(np.max(np.min(distances, axis=0)))
    return max(first_to_second, second_to_first)


def fourier_descriptor_distance(
    first: Iterable[Iterable[float]],
    second: Iterable[Iterable[float]],
    sample_count: int = 64,
    descriptor_count: int = 10,
) -> float:
    if descriptor_count <= 0:
        raise ValueError("descriptor_count must be positive")

    first_descriptor = _fourier_descriptor(first, sample_count, descriptor_count)
    second_descriptor = _fourier_descriptor(second, sample_count, descriptor_count)
    return float(np.linalg.norm(first_descriptor - second_descriptor))


def curvature_stats(points: Iterable[Iterable[float]]) -> dict[str, float]:
    array = _xy_array(points)
    if len(array) < 3:
        return _zero_curvature_stats()

    previous_vectors = array[1:-1] - array[:-2]
    next_vectors = array[2:] - array[1:-1]
    previous_lengths = np.linalg.norm(previous_vectors, axis=1)
    next_lengths = np.linalg.norm(next_vectors, axis=1)
    usable = (previous_lengths > 0.0) & (next_lengths > 0.0)
    if not np.any(usable):
        return _zero_curvature_stats()

    previous_unit = previous_vectors[usable] / previous_lengths[usable, None]
    next_unit = next_vectors[usable] / next_lengths[usable, None]
    cross = previous_unit[:, 0] * next_unit[:, 1] - previous_unit[:, 1] * next_unit[:, 0]
    dot = np.sum(previous_unit * next_unit, axis=1)
    angles = np.arctan2(cross, dot)
    local_lengths = (previous_lengths[usable] + next_lengths[usable]) * 0.5
    curvatures = angles / local_lengths
    abs_curvatures = np.abs(curvatures)

    if len(curvatures) > 1:
        variation = float(np.mean(np.abs(np.diff(curvatures))))
    else:
        variation = 0.0
    return {
        "mean_abs_curvature": float(np.mean(abs_curvatures)),
        "max_abs_curvature": float(np.max(abs_curvatures)),
        "curvature_variation": variation,
    }


def count_gear_shifts(speeds: Iterable[float]) -> int:
    return _count_sign_changes(speeds)


def count_cusps(speeds: Iterable[float]) -> int:
    return _count_sign_changes(speeds)


def count_low_speed_chatter(speeds: Iterable[float], max_abs_speed: float = 0.2) -> int:
    if max_abs_speed < 0.0:
        raise ValueError("max_abs_speed must be non-negative")

    values = _finite_values(speeds)
    count = 0
    previous_sign = 0
    previous_low_speed = False
    for value in values:
        sign = _sign(value)
        low_speed = abs(value) <= max_abs_speed
        if sign != 0 and previous_sign != 0 and low_speed and previous_low_speed and sign != previous_sign:
            count += 1
        if sign != 0:
            previous_sign = sign
            previous_low_speed = low_speed
    return count


def steering_sign_changes(steering_values: Iterable[float]) -> int:
    return _count_sign_changes(steering_values)


def min_obstacle_clearance(
    path_points: Iterable[Iterable[float]],
    obstacle_polygons: Iterable[Iterable[Iterable[float]]],
) -> float | None:
    path = _xy_array(path_points)
    minimum = math.inf

    for polygon in obstacle_polygons:
        try:
            vertices = _xy_array(polygon)
        except (TypeError, ValueError):
            continue
        if len(vertices) < 2:
            continue

        for start, end in _polygon_edges(vertices):
            distances = [_point_to_segment_distance(point, start, end) for point in path]
            minimum = min(minimum, *distances)

    if math.isinf(minimum):
        return None
    return float(minimum)


def _fourier_descriptor(
    points: Iterable[Iterable[float]],
    sample_count: int,
    descriptor_count: int,
) -> np.ndarray:
    sampled = np.asarray(resample_polyline(points, sample_count), dtype=float)
    complex_points = sampled[:, 0] + 1j * sampled[:, 1]
    centered = complex_points - np.mean(complex_points)
    scale = float(np.linalg.norm(centered))
    if scale > 0.0:
        centered = centered / scale
    descriptors = np.fft.fft(centered)[1 : descriptor_count + 1]
    if len(descriptors) < descriptor_count:
        descriptors = np.pad(descriptors, (0, descriptor_count - len(descriptors)))
    return np.concatenate((descriptors.real, descriptors.imag))


def _zero_curvature_stats() -> dict[str, float]:
    return {
        "mean_abs_curvature": 0.0,
        "max_abs_curvature": 0.0,
        "curvature_variation": 0.0,
    }


def _finite_values(values: Iterable[float]) -> list[float]:
    finite_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("values must be finite")
    return finite_values


def _count_sign_changes(values: Iterable[float]) -> int:
    count = 0
    previous = 0
    for value in _finite_values(values):
        sign = _sign(value)
        if sign == 0:
            continue
        if previous != 0 and sign != previous:
            count += 1
        previous = sign
    return count


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _polygon_edges(vertices: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for index in range(len(vertices)):
        yield vertices[index], vertices[(index + 1) % len(vertices)]


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / segment_length_squared)
    projection = min(1.0, max(0.0, projection))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))
