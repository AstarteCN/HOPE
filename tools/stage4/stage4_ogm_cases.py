from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_CASE_FILE = Path("docs/research/stage4_ogm_fixed_eval_cases_20260620.json")
PARALLEL_COUNT = 20
PERPENDICULAR_COUNT = 50
BASE_SEED = 20260620


def build_fixed_eval_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index in range(PARALLEL_COUNT):
        cases.append(
            {
                "case_uid": f"parallel_{index:03d}",
                "split": "Sim-Complex",
                "parking_type": "parallel",
                "hope_level": "Complex",
                "hope_case_id": 1,
                "seed": BASE_SEED + index,
            }
        )
    for index in range(PERPENDICULAR_COUNT):
        cases.append(
            {
                "case_uid": f"perpendicular_{index:03d}",
                "split": "Sim-Normal",
                "parking_type": "perpendicular",
                "hope_level": "Normal",
                "hope_case_id": 0,
                "seed": BASE_SEED + PARALLEL_COUNT + index,
            }
        )
    return cases


def split_case_counts(cases: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {"parallel": 0, "perpendicular": 0}
    for case in cases:
        counts[str(case["parking_type"])] += 1
    return counts


def write_cases(path: str | Path = DEFAULT_CASE_FILE) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = build_fixed_eval_cases()
    output_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_cases(path: str | Path = DEFAULT_CASE_FILE) -> list[dict[str, object]]:
    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_CASE_FILE)
    args = parser.parse_args()
    path = write_cases(args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
