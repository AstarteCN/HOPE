from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.stage3.stage3_baseline import BASELINE_ID, PROTECTED_SOURCE_PATHS


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout or completed.stderr).strip()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_branch(repo_root: Path) -> str:
    return _run_git(repo_root, ["branch", "--show-current"])


def git_status_short(repo_root: Path) -> str:
    return _run_git(repo_root, ["status", "-sb"])


def protected_source_diff(repo_root: Path) -> str:
    return _run_git(repo_root, ["diff", "HEAD", "--", *PROTECTED_SOURCE_PATHS])


def build_manifest(
    *,
    repo_root: Path,
    run_name: str,
    candidate_type: str,
    changed_knobs: Mapping[str, Any],
    command: Sequence[str],
    python_executable: str,
    env: Mapping[str, str],
    run_dir: str,
    stdout_path: str,
    stderr_path: str,
    resource_csv_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "baseline_id": BASELINE_ID,
        "run_name": run_name,
        "candidate_type": candidate_type,
        "changed_knobs": dict(changed_knobs),
        "git_branch": git_branch(repo_root),
        "git_status_sb": git_status_short(repo_root),
        "protected_source_diff": protected_source_diff(repo_root),
        "command": list(command),
        "python_executable": python_executable,
        "environment": dict(env),
        "run_dir": run_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "resource_csv_path": resource_csv_path,
        "tensorboard_summary_json": "",
        "tensorboard_summary_markdown": "",
        "expected_checkpoint_20k": str(Path(run_dir) / "SAC_19999.pt"),
        "eval_result_dirs": {},
        "launcher_pid": None,
        "workload_pid": None,
        "resource_monitor_pid": None,
        "gate_status": "created",
        "notes": [],
    }


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(path)
    manifest.update(updates)
    manifest["updated_at_utc"] = utc_timestamp()
    write_manifest(manifest, path)
    return manifest
