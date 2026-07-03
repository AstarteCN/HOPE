from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


WATCHED_SCALARS = [
    "total_reward",
    "avg_reward",
    "actor_loss",
    "critic_loss",
    "action_std0",
    "action_std1",
    "alpha",
    "success_rate_Normal",
    "success_rate_Complex",
    "success_rate_Extrem",
    "success_rate_dlp",
    "step_num",
]

HARD_REJECT_NONFINITE_SCALARS = ("actor_loss", "critic_loss", "alpha", "action_std0", "action_std1")


def _finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _trend(values: list[float], window: int = 50) -> dict[str, float | None]:
    finite = _finite_values(values)
    if len(finite) < 2:
        return {"first_mean": None, "last_mean": None, "delta": None}
    first_window = finite[: min(window, len(finite))]
    last_window = finite[-min(window, len(finite)) :]
    first_mean = statistics.fmean(first_window)
    last_mean = statistics.fmean(last_window)
    return {
        "first_mean": _json_number(first_mean),
        "last_mean": _json_number(last_mean),
        "delta": _json_number(last_mean - first_mean),
    }


def _summarize_values(events: list[Any]) -> dict[str, Any]:
    values = [float(event.value) for event in events]
    finite_values = _finite_values(values)
    has_nonfinite = len(finite_values) != len(values)
    summary: dict[str, Any] = {
        "count": len(values),
        "first_step": int(events[0].step) if events else None,
        "last_step": int(events[-1].step) if events else None,
        "first_value": _json_number(values[0]) if values else None,
        "last_value": _json_number(values[-1]) if values else None,
        "has_nan_or_inf": has_nonfinite,
        "has_nonfinite": has_nonfinite,
        "mean_first500": statistics.fmean(finite_values[:500]) if finite_values[:500] else None,
        "mean_last100": statistics.fmean(finite_values[-100:]) if finite_values[-100:] else None,
        "mean_last500": statistics.fmean(finite_values[-500:]) if finite_values[-500:] else None,
        "trend": _trend(values),
    }
    if finite_values:
        summary.update(
            {
                "min": _json_number(min(finite_values)),
                "max": _json_number(max(finite_values)),
                "mean": _json_number(statistics.fmean(finite_values)),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None})
    return summary


def _has_event_file(path: Path) -> bool:
    return any(
        child.name.startswith("events.out.tfevents")
        for child in path.iterdir()
        if child.is_file()
    )


def _newest_event_mtime(path: Path) -> float:
    return max(
        child.stat().st_mtime
        for child in path.iterdir()
        if child.is_file() and child.name.startswith("events.out.tfevents")
    )


def _find_event_directory(log_dir: Path) -> Path:
    if _has_event_file(log_dir):
        return log_dir
    candidates = [path for path in log_dir.rglob("*") if path.is_dir() and _has_event_file(path)]
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under {log_dir}")
    candidates.sort(key=lambda path: (-_newest_event_mtime(path), str(path.resolve())))
    return candidates[0]


def _nonnegative_finite_sum(events: list[Any]) -> int:
    values = [float(event.value) for event in events]
    return int(sum(value for value in values if math.isfinite(value) and value >= 0))


def summarize_event_dir(log_dir: Path, min_episodes: int) -> dict[str, Any]:
    event_dir = _find_event_directory(log_dir)
    accumulator = event_accumulator.EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))

    scalar_summaries: dict[str, Any] = {}
    missing_scalars: list[str] = []
    for tag in WATCHED_SCALARS:
        if tag not in tags:
            missing_scalars.append(tag)
            continue
        scalar_summaries[tag] = _summarize_values(accumulator.Scalars(tag))

    step_events = accumulator.Scalars("step_num") if "step_num" in tags else []
    episode_count = len(step_events)
    env_step_count = _nonnegative_finite_sum(step_events)
    updates_after_warmup = max(0, (env_step_count - 10240) // 10)
    warnings = build_warnings(scalar_summaries, episode_count, min_episodes)
    nonfinite_scalar_tags = [
        tag
        for tag in WATCHED_SCALARS
        if scalar_summaries.get(tag, {}).get("has_nonfinite", False)
    ]
    hard_reject_nonfinite_scalar_tags = [
        tag
        for tag in HARD_REJECT_NONFINITE_SCALARS
        if scalar_summaries.get(tag, {}).get("has_nonfinite", False)
    ]

    report = {
        "log_dir": str(log_dir.resolve()),
        "event_dir": str(event_dir.resolve()),
        "min_episodes": min_episodes,
        "episode_count": episode_count,
        "training_budget_met": episode_count >= min_episodes,
        "env_step_count": env_step_count,
        "estimated_sac_updates_after_warmup": updates_after_warmup,
        "missing_scalars": missing_scalars,
        "scalars": scalar_summaries,
        "nonfinite_scalar_tags": nonfinite_scalar_tags,
        "hard_reject_nonfinite_scalar_tags": hard_reject_nonfinite_scalar_tags,
        "hard_reject_has_nonfinite": bool(hard_reject_nonfinite_scalar_tags),
        "warnings": warnings,
    }
    report["has_nonfinite"] = any(
        scalar.get("has_nonfinite", False)
        for scalar in report["scalars"].values()
    )
    return report


def _last_value(report: dict[str, Any], tag: str) -> float | None:
    scalar = report.get("scalars", {}).get(tag)
    if not scalar:
        return None
    value = scalar.get("last_value")
    return float(value) if value is not None else None


def build_warnings(scalars: dict[str, Any], episode_count: int, min_episodes: int) -> list[str]:
    report = {"scalars": scalars}
    warnings: list[str] = []
    if episode_count < min_episodes:
        warnings.append(f"episode_count {episode_count} is below required minimum {min_episodes}")

    for tag in ["actor_loss", "critic_loss", "alpha", "action_std0", "action_std1"]:
        if scalars.get(tag, {}).get("has_nan_or_inf"):
            warnings.append(f"{tag} contains NaN or infinity")

    avg_reward = scalars.get("avg_reward", {})
    reward_delta = avg_reward.get("trend", {}).get("delta") if avg_reward else None
    if reward_delta is not None and reward_delta <= 0 and episode_count >= min_episodes:
        warnings.append("avg_reward did not improve between the first and last trend windows")

    for tag in [
        "success_rate_Normal",
        "success_rate_Complex",
        "success_rate_Extrem",
        "success_rate_dlp",
    ]:
        value = _last_value(report, tag)
        if value is not None and value < 0.2 and episode_count >= min_episodes:
            warnings.append(f"{tag} remains below 0.2 at the end of the run")

    step_num = _last_value(report, "step_num")
    if step_num is not None and step_num >= 190:
        warnings.append("final step_num is near the 200-step timeout")

    for tag in ["action_std0", "action_std1"]:
        value = _last_value(report, tag)
        if value is not None and value < -6:
            warnings.append(f"{tag} is very low, suggesting early exploration collapse")

    alpha = _last_value(report, "alpha")
    if alpha is not None and (alpha < 1e-5 or alpha > 10):
        warnings.append("alpha is outside the expected early-training range")

    return warnings


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Stage 3 TensorBoard Summary",
        "",
        f"- Log dir: `{report['log_dir']}`",
        f"- Event dir: `{report['event_dir']}`",
        f"- Episode count: `{report['episode_count']}`",
        f"- Required minimum episodes: `{report['min_episodes']}`",
        f"- Training budget met: `{report['training_budget_met']}`",
        f"- Environment step count: `{report['env_step_count']}`",
        f"- Estimated SAC updates after warmup: `{report['estimated_sac_updates_after_warmup']}`",
        f"- Has nonfinite watched scalar: `{report['has_nonfinite']}`",
        f"- Hard-reject has nonfinite scalar: `{report['hard_reject_has_nonfinite']}`",
        f"- Nonfinite scalar tags: `{report['nonfinite_scalar_tags']}`",
        f"- Hard-reject nonfinite scalar tags: `{report['hard_reject_nonfinite_scalar_tags']}`",
        "",
        "## Watched Scalars",
        "",
    ]
    for tag in WATCHED_SCALARS:
        scalar = report["scalars"].get(tag)
        if scalar is None:
            lines.append(f"- `{tag}`: missing")
            continue
        lines.append(
            "- `{tag}`: count `{count}`, first step `{first_step}`, last step `{last_step}`, "
            "first `{first}`, last `{last}`, mean `{mean}`, min `{minv}`, max `{maxv}`, "
            "has nonfinite `{has_nonfinite}`, mean first500 `{mean_first500}`, "
            "mean last100 `{mean_last100}`, mean last500 `{mean_last500}`, "
            "trend delta `{trend_delta}`".format(
                tag=tag,
                count=scalar["count"],
                first_step=scalar["first_step"],
                last_step=scalar["last_step"],
                first=scalar["first_value"],
                last=scalar["last_value"],
                mean=scalar["mean"],
                minv=scalar["min"],
                maxv=scalar["max"],
                has_nonfinite=scalar["has_nonfinite"],
                mean_first500=scalar["mean_first500"],
                mean_last100=scalar["mean_last100"],
                mean_last500=scalar["mean_last500"],
                trend_delta=scalar["trend"]["delta"],
            )
        )
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No automatic TensorBoard early-warning rule fired.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=40000)
    args = parser.parse_args()

    report = summarize_event_dir(args.log_dir, args.min_episodes)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(report, args.markdown)
    print(
        json.dumps(
            {
                "episode_count": report["episode_count"],
                "training_budget_met": report["training_budget_met"],
                "env_step_count": report["env_step_count"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
