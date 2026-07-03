from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model.replay_memory import ReplayMemory  # noqa: E402
from tools.stage4.train_HOPE_sac_ogm import (  # noqa: E402
    DlpCaseChoose,
    SceneChoose,
    load_stage4_state,
    periodic_checkpoint_name,
    save_stage4_state,
    stage4_state_name,
)


class _FakeInnerAgent:
    def __init__(self, memory: ReplayMemory) -> None:
        self.memory = memory
        self.actor_loss_list: list[float] = []
        self.critic_loss_list: list[float] = []


class _FakeParkingAgent:
    def __init__(self) -> None:
        self.agent = _FakeInnerAgent(ReplayMemory(128, ["log_prob", "next_obs"]))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_context() -> dict[str, Any]:
    return {
        "parking_agent": _FakeParkingAgent(),
        "scene_chooser": SceneChoose(),
        "dlp_case_chooser": DlpCaseChoose(),
        "total_step_num": 0,
        "best_success_rate": [0.0, 0.0, 0.0, 0.0],
        "reward_list": [],
        "reward_per_state_list": [],
        "reward_info_list": [],
        "case_id_list": [],
        "succ_record": [],
        "checkpoint_names": [],
        "state_artifact_names": [],
        "last_state_path": None,
    }


def _success_rates(scene_chooser: SceneChoose) -> list[float]:
    rates = []
    for scene_id in sorted(scene_chooser.scene_types):
        records = scene_chooser.success_record[scene_id]
        rates.append(float(np.mean(records[-100:]))) if records else rates.append(0.0)
    return rates


def _update_best_success_rate(context: dict[str, Any]) -> None:
    raw_rates = np.array(_success_rates(context["scene_chooser"]))
    target_rates = context["scene_chooser"].target_success_rate
    context["best_success_rate"] = list(np.minimum(raw_rates, target_rates))


def _maybe_save_state(context: dict[str, Any], save_dir: Path, global_episode: int) -> None:
    checkpoint_name = periodic_checkpoint_name(global_episode)
    checkpoint_path = save_dir / checkpoint_name
    checkpoint_path.write_text("bounded full-state resume verifier checkpoint\n", encoding="utf-8")
    state_path = save_stage4_state(
        save_dir=save_dir,
        global_episode=global_episode,
        parking_agent=context["parking_agent"],
        total_step_num=context["total_step_num"],
        scene_chooser=context["scene_chooser"],
        dlp_case_chooser=context["dlp_case_chooser"],
        best_success_rate=context["best_success_rate"],
        reward_list=context["reward_list"],
        reward_per_state_list=context["reward_per_state_list"],
        reward_info_list=context["reward_info_list"],
        case_id_list=context["case_id_list"],
        succ_record=context["succ_record"],
        sac_checkpoint_path=checkpoint_path,
    )
    context["checkpoint_names"].append(checkpoint_name)
    context["state_artifact_names"].append(stage4_state_name(global_episode))
    context["last_state_path"] = state_path


def _run_episode(context: dict[str, Any], writer: SummaryWriter, global_episode: int) -> None:
    scene = context["scene_chooser"].choose_case()
    case_id = context["dlp_case_chooser"].choose_case() if scene == "dlp" else None
    np_draw = float(np.random.random())
    torch_draw = float(torch.rand(()).item())
    cuda_draw = float(torch.rand((), device="cuda").item()) if torch.cuda.is_available() else 0.0
    step_num = int(3 + np.random.randint(0, 5) + (case_id % 3 if case_id is not None else 0))
    reward = float(round(np_draw + torch_draw + cuda_draw - 0.5, 8))
    success = int(reward > 0.0)

    context["total_step_num"] += step_num
    context["reward_list"].append(reward)
    context["reward_per_state_list"].extend([reward / step_num for _ in range(step_num)])
    context["reward_info_list"].append([round(reward, 4), step_num, success])
    context["case_id_list"].append("dlp-%s" % case_id if case_id is not None else scene)
    context["succ_record"].append(success)
    context["scene_chooser"].update_success_record(success)
    if scene == "dlp":
        context["dlp_case_chooser"].update_success_record(success, int(case_id))
    context["parking_agent"].agent.memory.push(
        (
            "obs-%s" % global_episode,
            "action-%s" % global_episode,
            reward,
            True,
            "log-%s" % global_episode,
            "next-%s" % global_episode,
        )
    )
    context["parking_agent"].agent.actor_loss_list.append(round(torch_draw, 8))
    context["parking_agent"].agent.critic_loss_list.append(round(np_draw, 8))
    _update_best_success_rate(context)

    writer.add_scalar("total_reward", reward, global_episode)
    writer.add_scalar("step_num", step_num, global_episode)
    writer.add_scalar("replay_length", len(context["parking_agent"].agent.memory), global_episode)


def _run_segment(
    context: dict[str, Any],
    writer: SummaryWriter,
    save_dir: Path,
    start_episode: int,
    end_episode: int,
    save_interval: int,
) -> None:
    for global_episode in range(start_episode, end_episode):
        _run_episode(context, writer, global_episode)
        if (global_episode + 1) % save_interval == 0:
            _maybe_save_state(context, save_dir, global_episode)
    writer.flush()


def _scalar_steps(event_dirs: list[Path], tag: str) -> list[int]:
    steps: list[int] = []
    for event_dir in event_dirs:
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        steps.extend(int(event.step) for event in accumulator.Scalars(tag))
    return sorted(steps)


def _rng_probe() -> dict[str, Any]:
    cuda_probe = None
    if torch.cuda.is_available():
        cuda_probe = float(torch.rand((), device="cuda").item())
    return {
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(()).item()),
        "cuda": cuda_probe,
    }


def _summarize_context(
    context: dict[str, Any],
    *,
    event_dirs: list[Path],
    include_rng_probe: bool,
) -> dict[str, Any]:
    scene_chooser = context["scene_chooser"]
    dlp_case_chooser = context["dlp_case_chooser"]
    parking_agent = context["parking_agent"]
    summary = {
        "total_step_num": context["total_step_num"],
        "replay_length": len(parking_agent.agent.memory),
        "scene_record": list(scene_chooser.scene_record),
        "scene_success_record": {str(key): list(value) for key, value in scene_chooser.success_record.items()},
        "dlp_case_record": list(dlp_case_chooser.case_record),
        "dlp_case_success_rate": {
            str(key): list(value)
            for key, value in dlp_case_chooser.case_success_rate.items()
            if value
        },
        "best_success_rate": list(context["best_success_rate"]),
        "reward_list": list(context["reward_list"]),
        "succ_record": list(context["succ_record"]),
        "checkpoint_names": list(context["checkpoint_names"]),
        "state_artifact_names": list(context["state_artifact_names"]),
        "tensorboard": {
            "total_reward_steps": _scalar_steps(event_dirs, "total_reward"),
            "step_num_steps": _scalar_steps(event_dirs, "step_num"),
        },
    }
    if include_rng_probe:
        summary["rng_probe"] = _rng_probe()
    return summary


def _restore_into_new_context(state_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _new_context()
    restored = load_stage4_state(
        state_path=state_path,
        parking_agent=context["parking_agent"],
        scene_chooser=context["scene_chooser"],
        dlp_case_chooser=context["dlp_case_chooser"],
    )
    context["total_step_num"] = restored["total_step_num"]
    context["best_success_rate"] = restored["best_success_rate"]
    context["reward_list"] = restored["reward_list"]
    context["reward_per_state_list"] = restored["reward_per_state_list"]
    context["reward_info_list"] = restored["reward_info_list"]
    context["case_id_list"] = restored["case_id_list"]
    context["succ_record"] = restored["succ_record"]
    if restored.get("sac_checkpoint_name"):
        context["checkpoint_names"].append(restored["sac_checkpoint_name"])
    context["state_artifact_names"].append(stage4_state_name(restored["global_episode"]))
    restored_summary = {
        "schema_version": restored["schema_version"],
        "global_episode": restored["global_episode"],
        "expected_start_episode": int(restored["global_episode"]) + 1,
        "total_step_num": restored["total_step_num"],
        "replay_length_after_load": len(context["parking_agent"].agent.memory),
        "scene_record_after_load": list(context["scene_chooser"].scene_record),
        "dlp_case_record_after_load": list(context["dlp_case_chooser"].case_record),
    }
    return context, restored_summary


def compare_resume_summaries(direct: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    missing = object()

    def get_path(summary: dict[str, Any], field: str) -> Any:
        value: Any = summary
        for part in field.split("."):
            if not isinstance(value, dict) or part not in value:
                return missing
            value = value[part]
        return value

    fields = [
        "total_step_num",
        "replay_length",
        "scene_record",
        "scene_success_record",
        "dlp_case_record",
        "dlp_case_success_rate",
        "best_success_rate",
        "reward_list",
        "succ_record",
        "checkpoint_names",
        "state_artifact_names",
        "tensorboard.total_reward_steps",
        "tensorboard.step_num_steps",
        "rng_probe",
    ]
    differences = []
    for field in fields:
        left = get_path(direct, field)
        right = get_path(resumed, field)
        if left != right:
            differences.append(field)
    return {
        "match": not differences,
        "differences": differences,
    }


def run_bounded_resume_simulation(
    *,
    work_dir: Path,
    episode_count: int = 8,
    interrupt_after_episode: int = 3,
    seed: int = 2468,
) -> dict[str, Any]:
    if interrupt_after_episode < 0 or interrupt_after_episode >= episode_count - 1:
        raise ValueError("interrupt_after_episode must leave at least one resumed episode.")

    save_interval = interrupt_after_episode + 1
    work_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)
    direct_dir = work_dir / "direct"
    direct_writer = SummaryWriter(str(direct_dir / "events"))
    direct_context = _new_context()
    _run_segment(direct_context, direct_writer, direct_dir, 0, episode_count, save_interval)
    direct_writer.close()
    direct_summary = _summarize_context(
        direct_context,
        event_dirs=[direct_dir / "events"],
        include_rng_probe=True,
    )

    _set_seed(seed)
    part1_dir = work_dir / "interrupted_part1"
    part1_writer = SummaryWriter(str(part1_dir / "events"))
    part1_context = _new_context()
    _run_segment(
        part1_context,
        part1_writer,
        part1_dir,
        0,
        interrupt_after_episode + 1,
        save_interval,
    )
    part1_writer.close()
    interrupted_before_resume = _summarize_context(
        part1_context,
        event_dirs=[part1_dir / "events"],
        include_rng_probe=False,
    )

    state_path = part1_context["last_state_path"]
    if state_path is None:
        raise RuntimeError("interrupted segment did not produce a Stage 4 state artifact.")

    np.random.random(25)
    torch.rand(25)
    if torch.cuda.is_available():
        torch.rand(25, device="cuda")

    resumed_context, restored_summary = _restore_into_new_context(state_path)
    part2_dir = work_dir / "interrupted_part2"
    part2_writer = SummaryWriter(str(part2_dir / "events"))
    _run_segment(
        resumed_context,
        part2_writer,
        part2_dir,
        restored_summary["expected_start_episode"],
        episode_count,
        save_interval,
    )
    part2_writer.close()
    resumed_summary = _summarize_context(
        resumed_context,
        event_dirs=[part1_dir / "events", part2_dir / "events"],
        include_rng_probe=True,
    )
    comparison = compare_resume_summaries(direct_summary, resumed_summary)
    cuda_note = (
        "CUDA RNG progression was included because torch.cuda.is_available() returned true."
        if torch.cuda.is_available()
        else "CUDA RNG progression was not exercised in this run because CUDA was unavailable."
    )
    return {
        "schema_version": 1,
        "verification_type": "deterministic_runner_state_simulation",
        "episode_count": episode_count,
        "interrupt_after_episode": interrupt_after_episode,
        "seed": seed,
        "direct": direct_summary,
        "interrupted_before_resume": interrupted_before_resume,
        "restored_state": restored_summary,
        "interrupted_resumed": resumed_summary,
        "comparison": comparison,
        "limitations": (
            "This is a bounded deterministic runner-state simulation. It exercises the actual "
            "Stage 4 full-state save/load helpers, checkpoint naming helpers, chooser state, "
            "replay memory, RNG capture/restore, and TensorBoard global-step writes, but it "
            "does not prove bitwise real SAC learning trajectory equivalence."
        ),
        "cuda_note": cuda_note,
    }


def write_report(report: dict[str, Any], output_json: Path, output_md: Path) -> None:
    def json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError("Object of type %s is not JSON serializable" % value.__class__.__name__)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    status = "PASS" if report["comparison"]["match"] else "FAIL"
    lines = [
        "# Stage 4 Full-State Resume Verification",
        "",
        f"- Status: {status}",
        f"- Verification type: `{report['verification_type']}`",
        f"- Episodes: {report['episode_count']}",
        f"- Interrupt after episode: {report['interrupt_after_episode']}",
        f"- Seed: {report['seed']}",
        f"- Differences: {report['comparison']['differences']}",
        f"- Direct total steps: {report['direct']['total_step_num']}",
        f"- Resumed total steps: {report['interrupted_resumed']['total_step_num']}",
        f"- Direct replay length: {report['direct']['replay_length']}",
        f"- Resumed replay length: {report['interrupted_resumed']['replay_length']}",
        f"- TensorBoard total_reward steps: {report['interrupted_resumed']['tensorboard']['total_reward_steps']}",
        f"- Checkpoint names: {report['interrupted_resumed']['checkpoint_names']}",
        f"- State artifact names: {report['interrupted_resumed']['state_artifact_names']}",
        "",
        "## Restored State",
        "",
        f"- Restored global episode: {report['restored_state']['global_episode']}",
        f"- Expected resumed start episode: {report['restored_state']['expected_start_episode']}",
        f"- Restored replay length: {report['restored_state']['replay_length_after_load']}",
        f"- Restored total step count: {report['restored_state']['total_step_num']}",
        "",
        "## Limitations",
        "",
        report["limitations"],
        "",
        report["cuda_note"],
        "",
        f"JSON evidence: `{output_json}`",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Stage 4 full-state resume behavior with a bounded simulation.")
    parser.add_argument("--episode-count", type=int, default=8)
    parser.add_argument("--interrupt-after-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "2026-06-22-stage4-full-state-resume-verification.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "2026-06-22-stage4-full-state-resume-verification.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="stage4 full-state resume verify ") as temp_dir:
            report = run_bounded_resume_simulation(
                work_dir=Path(temp_dir),
                episode_count=args.episode_count,
                interrupt_after_episode=args.interrupt_after_episode,
                seed=args.seed,
            )
    else:
        report = run_bounded_resume_simulation(
            work_dir=args.work_dir,
            episode_count=args.episode_count,
            interrupt_after_episode=args.interrupt_after_episode,
            seed=args.seed,
        )
    write_report(report, args.output_json, args.output_md)
    print(args.output_json)
    print(args.output_md)
    return 0 if report["comparison"]["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
