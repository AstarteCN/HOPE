from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tools.stage3.stage3_baseline import (
    BASELINE_20K,
    quality_gate_status,
    scene_mean,
    speed_gate_passed,
)
from tools.stage3.stage3_manifest import read_manifest, update_manifest
from tools.stage3.stage3_result_parsers import collect_eval_success, summarize_resource_csv


SCENE_SUCCESS_TAGS = (
    "success_rate_Normal",
    "success_rate_Complex",
    "success_rate_Extrem",
    "success_rate_dlp",
)


def classify_decision(quality: Mapping[str, Any], speed: Mapping[str, Any]) -> str:
    if quality.get("hard_reject_reasons"):
        return "reject"
    if not quality.get("quality_pass", False):
        return "reject"
    if quality.get("investigate_reasons"):
        return "investigate"
    if speed.get("speed_pass", False):
        return "pass"
    return "quality-pass-speed-neutral"


def _first_numeric(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def _multi_scene_collapse(tensorboard_summary: Mapping[str, Any]) -> dict[str, Any]:
    scalars = tensorboard_summary.get("scalars", {})
    low_scene_success_values: dict[str, float] = {}
    for tag in SCENE_SUCCESS_TAGS:
        scalar = scalars.get(tag, {})
        if not isinstance(scalar, Mapping):
            continue
        value = _first_numeric(scalar, ("mean_last100", "recent_mean", "latest"))
        if value is not None and value < 0.5:
            low_scene_success_values[tag] = value

    return {
        "multi_scene_collapse": len(low_scene_success_values) >= 2,
        "low_scene_success_tags": list(low_scene_success_values),
        "low_scene_success_values": low_scene_success_values,
    }


def _core_scene_collapse_reasons(collapse: Mapping[str, Any]) -> list[str]:
    low_values = collapse.get("low_scene_success_values", {})
    if not isinstance(low_values, Mapping):
        return []

    reasons = []
    for scene in ("Normal", "Complex"):
        tag = f"success_rate_{scene}"
        if tag in low_values:
            reasons.append(f"severe_core_scene_collapse_{scene}")
    return reasons


def build_comparison_report(
    manifest: Mapping[str, Any],
    tensorboard_summary: Mapping[str, Any],
    resource_csv_path: str | Path,
    eval_success: Mapping[str, float],
    candidate_speed: Mapping[str, float],
    parity_status: str,
    checkpoint_missing: bool,
) -> dict[str, Any]:
    collapse = _multi_scene_collapse(tensorboard_summary)
    hard_reject_has_nonfinite = bool(tensorboard_summary.get("hard_reject_has_nonfinite", False))
    quality = quality_gate_status(
        eval_success=eval_success,
        tensorboard_has_nonfinite=hard_reject_has_nonfinite,
        multi_scene_collapse=collapse["multi_scene_collapse"],
        checkpoint_missing=checkpoint_missing,
        parity_failed=parity_status == "fail",
    )
    core_scene_reasons = _core_scene_collapse_reasons(collapse)
    if core_scene_reasons:
        quality["hard_reject_reasons"].extend(core_scene_reasons)
        quality["quality_pass"] = False

    speed = speed_gate_passed(candidate_speed)
    decision = classify_decision(quality, speed)

    report = {
        "schema_version": 1,
        "decision": decision,
        "baseline_id": manifest.get("baseline_id", BASELINE_20K["id"]),
        "baseline": BASELINE_20K,
        "candidate": {
            "candidate_type": manifest.get("candidate_type", ""),
            "run_dir": manifest.get("run_dir", ""),
            "changed_knobs": manifest.get("changed_knobs", {}),
        },
        "command": manifest.get("command", []),
        "parity_status": parity_status,
        "checkpoint_missing": bool(checkpoint_missing),
        "speed": {
            "candidate_metrics": dict(candidate_speed),
            **speed,
        },
        "quality": quality,
        "scene_eval": {
            "success": dict(eval_success),
            "mean_success": scene_mean(eval_success),
        },
        "tensorboard": {
            "has_nonfinite": bool(tensorboard_summary.get("has_nonfinite", False)),
            "hard_reject_has_nonfinite": hard_reject_has_nonfinite,
            "nonfinite_scalar_tags": list(tensorboard_summary.get("nonfinite_scalar_tags", [])),
            "hard_reject_nonfinite_scalar_tags": list(
                tensorboard_summary.get("hard_reject_nonfinite_scalar_tags", [])
            ),
            "core_scene_collapse_reasons": core_scene_reasons,
            **collapse,
        },
        "resource_profile": summarize_resource_csv(Path(resource_csv_path)),
    }
    return report


def write_markdown(report: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = report.get("command", [])
    command_text = " ".join(str(part) for part in command) if isinstance(command, list) else str(command)
    candidate = report.get("candidate", {})
    speed = report.get("speed", {})
    quality = report.get("quality", {})
    scene_eval = report.get("scene_eval", {})
    resource = report.get("resource_profile", {})
    tensorboard = report.get("tensorboard", {})

    lines = [
        "# Stage 3 20K Comparison Report",
        "",
        f"- Decision: `{report.get('decision', '')}`",
        f"- Baseline id: `{report.get('baseline_id', '')}`",
        f"- Candidate type: `{candidate.get('candidate_type', '')}`",
        f"- Run dir: `{candidate.get('run_dir', '')}`",
        f"- Parity status: `{report.get('parity_status', '')}`",
        f"- Checkpoint missing: `{report.get('checkpoint_missing', False)}`",
        "",
        "## Changed Knobs",
        "",
        "```json",
        json.dumps(candidate.get("changed_knobs", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Speed",
        "",
        f"- Speed pass: `{speed.get('speed_pass', False)}`",
        f"- Candidate metrics: `{json.dumps(speed.get('candidate_metrics', {}), ensure_ascii=False)}`",
        f"- Wall-time improvement ratio: `{speed.get('wall_time_improvement_ratio', '')}`",
        f"- Episodes/hour improvement ratio: `{speed.get('episodes_per_hour_improvement_ratio', '')}`",
        f"- Env steps/second improvement ratio: `{speed.get('env_steps_per_second_improvement_ratio', '')}`",
        "",
        "## Quality",
        "",
        f"- Quality pass: `{quality.get('quality_pass', False)}`",
        f"- Mean success: `{quality.get('mean_success', '')}`",
        f"- Mean pass: `{quality.get('mean_pass', False)}`",
        f"- Hard reject reasons: `{json.dumps(quality.get('hard_reject_reasons', []), ensure_ascii=False)}`",
        "",
        "## Scene Eval",
        "",
    ]

    success = scene_eval.get("success", {})
    for scene, value in success.items():
        lines.append(f"- {scene}: `{value}`")
    lines.extend(
        [
            f"- Mean: `{scene_eval.get('mean_success', '')}`",
            "",
            "## Resource Summary",
            "",
            f"- Sample count: `{resource.get('sample_count', '')}`",
            f"- Avg process CPU percent: `{resource.get('avg_process_cpu_percent', '')}`",
            f"- Avg whole GPU util percent: `{resource.get('avg_whole_gpu_util_percent', '')}`",
            f"- Avg GPU memory used MB: `{resource.get('avg_gpu_memory_used_mb', '')}`",
            f"- Max GPU memory used MB: `{resource.get('max_gpu_memory_used_mb', '')}`",
            "",
            "## TensorBoard",
            "",
            f"- Has nonfinite: `{tensorboard.get('has_nonfinite', False)}`",
            f"- Hard-reject has nonfinite: `{tensorboard.get('hard_reject_has_nonfinite', False)}`",
            f"- Nonfinite scalar tags: `{json.dumps(tensorboard.get('nonfinite_scalar_tags', []), ensure_ascii=False)}`",
            "- Hard-reject nonfinite scalar tags: "
            f"`{json.dumps(tensorboard.get('hard_reject_nonfinite_scalar_tags', []), ensure_ascii=False)}`",
            f"- Multi-scene collapse: `{tensorboard.get('multi_scene_collapse', False)}`",
            f"- Low scene success tags: `{json.dumps(tensorboard.get('low_scene_success_tags', []), ensure_ascii=False)}`",
            "- Core scene collapse reasons: "
            f"`{json.dumps(tensorboard.get('core_scene_collapse_reasons', []), ensure_ascii=False)}`",
            "",
            "## Command",
            "",
            "```powershell",
            command_text,
            "```",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a Stage 3 20K candidate run against the command-only baseline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tensorboard-json", required=True, type=Path)
    parser.add_argument("--resource-csv", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--wall-time-hours", required=True, type=float)
    parser.add_argument("--episodes-per-hour", required=True, type=float)
    parser.add_argument("--env-steps-per-second", required=True, type=float)
    parser.add_argument("--parity-status", choices=("not-required", "pass", "fail"), default="not-required")
    parser.add_argument("--checkpoint-missing", action="store_true")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    manifest = read_manifest(args.manifest)
    tensorboard_summary = json.loads(args.tensorboard_json.read_text(encoding="utf-8"))
    eval_success = collect_eval_success(args.eval_root)
    candidate_speed = {
        "wall_time_hours": args.wall_time_hours,
        "episodes_per_hour": args.episodes_per_hour,
        "env_steps_per_second": args.env_steps_per_second,
    }

    report = build_comparison_report(
        manifest=manifest,
        tensorboard_summary=tensorboard_summary,
        resource_csv_path=args.resource_csv,
        eval_success=eval_success,
        candidate_speed=candidate_speed,
        parity_status=args.parity_status,
        checkpoint_missing=args.checkpoint_missing,
    )

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    update_manifest(
        args.manifest,
        {
            "gate_status": report["decision"],
            "comparison_report_json": str(args.json),
            "comparison_report_markdown": str(args.markdown),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
