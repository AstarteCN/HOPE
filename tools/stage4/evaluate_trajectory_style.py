from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stage4.trajectory_style_loader import load_trajectory_traces
from tools.stage4.trajectory_style_report import (
    evaluate_batch,
    write_json_reports,
    write_markdown_report,
)


VALID_SLOT_TYPES = frozenset({"perpendicular", "parallel"})


def _slot_types(value: str) -> list[str]:
    slot_types = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not slot_types:
        raise argparse.ArgumentTypeError("--slot-types must include perpendicular or parallel")
    invalid = sorted(set(slot_types) - VALID_SLOT_TYPES)
    if invalid:
        raise argparse.ArgumentTypeError(
            "--slot-types only accepts comma-separated perpendicular/parallel values; got %s"
            % ", ".join(invalid)
        )
    return slot_types


def _load_scene_labels(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    payload = _load_json_object(path, "scene labels")
    labels: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str):
            raise ValueError(f"scene label for {key!r} must be a string")
        labels[str(key)] = value
    return labels


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    json_path = Path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} file does not exist: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a JSON object")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate offline trajectory style traces.")
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slot-types", required=True, type=_slot_types)
    parser.add_argument("--scene-labels")
    parser.add_argument("--reference-annotations")
    parser.add_argument("--case-filter")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument("--visualization-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        traces = load_trajectory_traces(args.trace_json)
        if args.case_filter:
            traces = [trace for trace in traces if args.case_filter in trace.case_uid]

        scene_labels = _load_scene_labels(args.scene_labels)
        if args.reference_annotations:
            _load_json_object(args.reference_annotations, "reference annotations")

        report = evaluate_batch(
            traces,
            slot_types=args.slot_types,
            label_overrides=scene_labels,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.write_json or not args.write_markdown:
            write_json_reports(report, output_dir)
        if args.write_markdown:
            write_markdown_report(report, output_dir)

        if report["unsupported_case_count"] and args.strict and not args.allow_unsupported:
            print(
                "unsupported cases present: %d" % report["unsupported_case_count"],
                file=sys.stderr,
            )
            return 2

        print(output_dir)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
