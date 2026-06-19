from __future__ import annotations

import csv
import re
from pathlib import Path


SCENE_DIRS = {
    "Normal": "normalize",
    "Complex": "complex",
    "Extrem": "extreme",
    "DLP": "dlp",
}


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_eval_result_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    success_match = re.search(r"success rate:\s*([0-9.]+)", text)
    step_match = re.search(r"step num:\s*([0-9.]+)", text)
    if not success_match:
        raise ValueError(f"missing success rate in {path}")
    return {
        "success_rate": float(success_match.group(1)),
        "step_num_mean": float(step_match.group(1)) if step_match else float("nan"),
    }


def collect_eval_success(eval_root: Path) -> dict[str, float]:
    success: dict[str, float] = {}
    for scene, directory in SCENE_DIRS.items():
        result_path = eval_root / directory / "result.txt"
        success[scene] = parse_eval_result_file(result_path)["success_rate"]
    return success


def summarize_resource_csv(path: Path) -> dict[str, float | int | str]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    cpu_values = [_float_or_none(row.get("cpu_percent", "")) for row in rows]
    working_set_values = [_float_or_none(row.get("working_set_mb", "")) for row in rows]
    private_memory_values = [_float_or_none(row.get("private_memory_mb", "")) for row in rows]
    gpu_values = [_float_or_none(row.get("gpu_util_percent", "")) for row in rows]
    memory_values = [_float_or_none(row.get("gpu_memory_used_mb", "")) for row in rows]
    memory_total_values = [_float_or_none(row.get("gpu_memory_total_mb", "")) for row in rows]
    power_values = [_float_or_none(row.get("gpu_power_w", "")) for row in rows]

    def average(values: list[float | None]) -> float:
        finite = [value for value in values if value is not None]
        return sum(finite) / len(finite) if finite else 0.0

    def maximum(values: list[float | None]) -> float:
        finite = [value for value in values if value is not None]
        return max(finite) if finite else 0.0

    return {
        "path": str(path),
        "sample_count": len(rows),
        "avg_process_cpu_percent": average(cpu_values),
        "max_process_cpu_percent": maximum(cpu_values),
        "avg_working_set_mb": average(working_set_values),
        "max_working_set_mb": maximum(working_set_values),
        "avg_private_memory_mb": average(private_memory_values),
        "max_private_memory_mb": maximum(private_memory_values),
        "avg_whole_gpu_util_percent": average(gpu_values),
        "max_whole_gpu_util_percent": maximum(gpu_values),
        "avg_gpu_power_w": average(power_values),
        "avg_gpu_memory_used_mb": average(memory_values),
        "max_gpu_memory_used_mb": maximum(memory_values),
        "avg_gpu_memory_total_mb": average(memory_total_values),
        "max_gpu_memory_total_mb": maximum(memory_total_values),
        "max_gpu_power_w": maximum(power_values),
    }
