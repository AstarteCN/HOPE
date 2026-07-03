from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.stage4.trajectory_style_evaluator import evaluate_trace_style
from tools.stage4.trajectory_style_reference import classify_scene, generate_reference_families
from tools.stage4.trajectory_style_schema import StyleCaseReport, TrajectoryTrace


SCHEMA_VERSION = "trajectory-style-eval-v1"
WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
)


def evaluate_batch(
    traces: Sequence[TrajectoryTrace],
    *,
    slot_types: Sequence[str],
    label_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    allowed_slot_types = {slot_type.lower() for slot_type in slot_types}
    cases: list[dict[str, Any]] = []
    supported_cases: list[dict[str, Any]] = []
    unsupported_cases: list[dict[str, Any]] = []
    scene_class_counts: Counter[str] = Counter()

    for trace in traces:
        if trace.slot_type.lower() not in allowed_slot_types:
            reason = f"slot_type filtered out: {trace.slot_type}"
            case_report = _filtered_out_report(trace, reason)
        else:
            classification = classify_scene(trace, label_overrides)
            references = generate_reference_families(trace, classification)
            case_report = evaluate_trace_style(trace, classification, references)

        case_payload = case_report.to_dict()
        cases.append(case_payload)
        scene_class_counts[case_payload["scene_class"]] += 1
        if case_payload["unsupported_reason"] is None:
            supported_cases.append(case_payload)
        else:
            unsupported_cases.append(case_payload)

    top_style_mismatches = sorted(
        supported_cases,
        key=lambda case: (case["style_score"], case["case_uid"]),
    )[:10]

    return {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(traces),
        "supported_case_count": len(supported_cases),
        "unsupported_case_count": len(unsupported_cases),
        "scene_class_counts": dict(sorted(scene_class_counts.items())),
        "top_style_mismatches": top_style_mismatches,
        "cases": cases,
        "unsupported_cases": unsupported_cases,
    }


def write_json_reports(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    case_summary_dir = output_path / "case_summaries"
    case_summary_dir.mkdir(parents=True, exist_ok=True)

    case_paths: list[tuple[Path, Mapping[str, Any]]] = []
    seen_case_summary_names: set[str] = set()
    for case in report.get("cases", []):
        case_uid = str(case["case_uid"])
        case_path = _case_summary_path(case_summary_dir, case_uid)
        if case_path.name in seen_case_summary_names:
            raise ValueError(f"duplicate case summary filename after sanitizing case_uid: {case_path.name}")
        seen_case_summary_names.add(case_path.name)
        case_paths.append((case_path, case))

    _write_json(output_path / "style_eval_report.json", report)
    _write_json(output_path / "unsupported_cases.json", report.get("unsupported_cases", []))

    for case_path, case in case_paths:
        _write_json(case_path, case)


def write_markdown_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Trajectory Style Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Schema version: {report.get('schema_version')}",
        f"- Cases: {report.get('case_count')}",
        f"- Supported cases: {report.get('supported_case_count')}",
        f"- Unsupported cases: {report.get('unsupported_case_count')}",
        "",
        "## Scene Class Counts",
        "",
    ]

    scene_class_counts = report.get("scene_class_counts", {})
    if scene_class_counts:
        for scene_class, count in sorted(scene_class_counts.items()):
            lines.append(f"- {scene_class}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Style Mismatches", ""])
    top_style_mismatches = report.get("top_style_mismatches", [])
    if top_style_mismatches:
        for case in top_style_mismatches:
            lines.extend(_case_markdown_lines(case))
    else:
        lines.append("- none")

    lines.extend(["", "## Unsupported Cases", ""])
    unsupported_cases = report.get("unsupported_cases", [])
    if unsupported_cases:
        for case in unsupported_cases:
            lines.append(
                "- {case_uid}: {reason}".format(
                    case_uid=case.get("case_uid"),
                    reason=case.get("unsupported_reason"),
                )
            )
    else:
        lines.append("- none")

    (output_path / "style_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filtered_out_report(trace: TrajectoryTrace, reason: str) -> StyleCaseReport:
    return StyleCaseReport(
        case_uid=trace.case_uid,
        scene_class="unsupported",
        classification_reason=reason,
        selected_reference_family=None,
        shape_metrics={},
        behavior_metrics={},
        clearance_metrics={},
        segment_metrics={
            "rl_count": sum(1 for source in trace.action_sources if str(source).upper() == "RL"),
            "rs_count": sum(1 for source in trace.action_sources if str(source).upper() == "RS"),
            "planner_active_count": sum(1 for active in trace.planner_route_active if active),
            "total_count": len(trace.poses),
        },
        style_label="unsupported",
        style_score=0.0,
        diagnosis=["route_family_mismatch"],
        unsupported_reason=reason,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_summary_path(case_summary_dir: Path, case_uid: str) -> Path:
    filename = _case_summary_filename(case_uid)
    root = case_summary_dir.resolve()
    candidate = (case_summary_dir / filename).resolve()
    if candidate.parent != root:
        raise ValueError(f"unsafe case_uid for case summary path: {case_uid!r}")
    return candidate


def _case_summary_filename(case_uid: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", case_uid).strip("._")
    changed = safe_stem != case_uid
    if not safe_stem:
        safe_stem = "case"
        changed = True

    stem_for_reserved_check = safe_stem.split(".", 1)[0].upper()
    if stem_for_reserved_check in WINDOWS_RESERVED_FILENAMES:
        changed = True

    if changed:
        digest = hashlib.sha1(case_uid.encode("utf-8")).hexdigest()[:10]
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.json"


def _case_markdown_lines(case: Mapping[str, Any]) -> list[str]:
    lines = [
        "- {case_uid}: score={style_score:.6f}, label={style_label}, scene={scene_class}".format(
            case_uid=case.get("case_uid"),
            style_score=float(case.get("style_score", 0.0)),
            style_label=case.get("style_label"),
            scene_class=case.get("scene_class"),
        )
    ]
    reference_family = case.get("selected_reference_family")
    if reference_family:
        lines.append(f"  - reference: {reference_family}")
    diagnosis = case.get("diagnosis") or []
    if diagnosis:
        lines.append("  - diagnosis: " + ", ".join(str(item) for item in diagnosis))
    shape_metrics = case.get("shape_metrics") or {}
    if shape_metrics:
        rendered_metrics = ", ".join(
            f"{key}={_format_metric(value)}"
            for key, value in sorted(shape_metrics.items())
        )
        lines.append(f"  - shape metrics: {rendered_metrics}")
    return lines


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
