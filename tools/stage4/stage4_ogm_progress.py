from __future__ import annotations

from typing import Any, Mapping

FIRST_GATE_MIN_EPISODE = 19_500
FIRST_GATE_MAX_EPISODE = 20_500
FIXED_EVAL_RELATIVE_IMPROVEMENT = 0.03

_RESTART_REASONS = {
    "ogm_rasterizer_changed",
    "observation_semantics_changed",
    "reward_semantics_changed",
    "action_semantics_changed",
    "network_shape_changed",
    "network_input_shape_changed",
    "replay_buffer_contaminated",
    "replay_meaning_changed",
    "state_norm_semantics_changed",
    "state_normalization_changed",
}

_RESUME_REASONS = {
    "logging_only",
    "reward_logging_only",
    "tensorboard_logging_only",
    "monitoring_only",
    "monitoring_fix",
    "eval_only_fix",
    "eval_parser_fix",
    "evaluation_only_fix",
    "report_format_only",
    "reporting_only",
    "resource_monitor_fix",
    "resource_monitoring_only",
}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_scalar_trend_delta(summary: Mapping[str, Any], scalar_name: str) -> float | None:
    scalars = summary.get("scalars", {})
    if not isinstance(scalars, Mapping):
        return None
    scalar = scalars.get(scalar_name, {})
    if not isinstance(scalar, Mapping):
        return None
    trend = scalar.get("trend", {})
    if not isinstance(trend, Mapping):
        return None
    return _optional_float(trend.get("delta"))


def _scalar_trend_delta(summary: Mapping[str, Any], scalar_name: str) -> float:
    return _optional_scalar_trend_delta(summary, scalar_name) or 0.0


def _mean_success_rate_delta(summary: Mapping[str, Any]) -> float:
    deltas = [
        _optional_scalar_trend_delta(summary, name)
        for name in (
            "success_rate_Normal",
            "success_rate_Complex",
            "success_rate_Extrem",
            "success_rate_dlp",
        )
    ]
    available = [delta for delta in deltas if delta is not None]
    if not available:
        return 0.0
    return sum(available) / len(available)


def _step_num_improvement(summary: Mapping[str, Any]) -> float:
    scalars = summary.get("scalars", {})
    if not isinstance(scalars, Mapping):
        return 0.0
    step_num = scalars.get("step_num", {})
    if not isinstance(step_num, Mapping):
        return 0.0
    trend = step_num.get("trend", {})
    if not isinstance(trend, Mapping):
        return 0.0
    first_mean = _optional_float(trend.get("first_mean"))
    last_mean = _optional_float(trend.get("last_mean"))
    if first_mean is None or last_mean is None or first_mean == 0.0:
        return 0.0
    return (first_mean - last_mean) / abs(first_mean)


def build_progress_gate_input(
    *,
    episode: int,
    tensorboard_summary: Mapping[str, Any],
    fixed_eval: Mapping[str, Any],
    checkpoint_exists: bool,
    state_snapshot_required: bool | None = None,
    state_snapshot_exists: bool | None = None,
) -> dict[str, Any]:
    """Build the pure input mapping consumed by ``decide_progress_gate``."""
    result: dict[str, Any] = {
        "episode": int(episode),
        "has_nonfinite": bool(
            tensorboard_summary.get(
                "hard_reject_has_nonfinite",
                tensorboard_summary.get("has_nonfinite", False),
            )
        ),
        "checkpoint_exists": bool(checkpoint_exists),
        "trend": {
            "avg_reward_delta": _scalar_trend_delta(tensorboard_summary, "avg_reward"),
            "mean_psr_delta": _mean_success_rate_delta(tensorboard_summary),
            "step_num_improvement": _step_num_improvement(tensorboard_summary),
        },
        "summary": fixed_eval["summary"],
    }
    if state_snapshot_required is not None:
        result["state_snapshot_required"] = bool(state_snapshot_required)
    if state_snapshot_exists is not None:
        result["state_snapshot_exists"] = bool(state_snapshot_exists)
    return result


def _lower_is_better_relative_improvements(
    current_summary: Mapping[str, Any],
    previous_summary: Mapping[str, Any],
    metric: str,
) -> list[float]:
    improvements: list[float] = []
    for split, current_metrics in current_summary.items():
        previous_metrics = previous_summary.get(split)
        if not isinstance(current_metrics, Mapping) or not isinstance(previous_metrics, Mapping):
            continue
        previous_value = _optional_float(previous_metrics.get(metric))
        current_value = _optional_float(current_metrics.get(metric))
        if previous_value is None or current_value is None or previous_value == 0.0:
            continue
        improvements.append((previous_value - current_value) / abs(previous_value))
    return improvements


def _psr_deltas(
    current_summary: Mapping[str, Any],
    previous_summary: Mapping[str, Any],
) -> list[float]:
    deltas: list[float] = []
    for split, current_metrics in current_summary.items():
        previous_metrics = previous_summary.get(split)
        if not isinstance(current_metrics, Mapping) or not isinstance(previous_metrics, Mapping):
            continue
        previous_value = _optional_float(previous_metrics.get("psr"))
        current_value = _optional_float(current_metrics.get("psr"))
        if previous_value is None or current_value is None:
            continue
        deltas.append(current_value - previous_value)
    return deltas


def _fixed_eval_quality_signals(
    current: Mapping[str, Any] | None,
    previous: Mapping[str, Any] | None,
) -> list[str]:
    if current is None or previous is None:
        return []
    current_summary = current.get("summary", {})
    previous_summary = previous.get("summary", {})
    if not isinstance(current_summary, Mapping) or not isinstance(previous_summary, Mapping):
        return []

    psr_deltas = _psr_deltas(current_summary, previous_summary)
    if psr_deltas and min(psr_deltas) < 0.0:
        return []

    signals: list[str] = []
    mean_angs_improvement = _mean(
        _lower_is_better_relative_improvements(current_summary, previous_summary, "angs")
    )
    if (
        mean_angs_improvement is not None
        and mean_angs_improvement > FIXED_EVAL_RELATIVE_IMPROVEMENT
    ):
        signals.append("fixed_eval_angs_improved")

    mean_pl_improvement = _mean(
        _lower_is_better_relative_improvements(current_summary, previous_summary, "pl")
    )
    if mean_pl_improvement is not None and mean_pl_improvement > FIXED_EVAL_RELATIVE_IMPROVEMENT:
        signals.append("fixed_eval_pl_improved")

    return signals


def _positive_trend_signals(
    trend: Mapping[str, float],
    current: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> list[str]:
    positive: list[str] = []
    if float(trend.get("avg_reward_delta", 0.0)) > 0.02:
        positive.append("avg_reward_improved")
    if float(trend.get("mean_psr_delta", 0.0)) > 0.02:
        positive.append("mean_psr_improved")
    if float(trend.get("step_num_improvement", 0.0)) > 0.05:
        positive.append("step_num_improved")
    positive.extend(_fixed_eval_quality_signals(current, previous))
    return positive


def _trend_is_improving(
    trend: Mapping[str, float],
    current: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> bool:
    return len(_positive_trend_signals(trend, current=current, previous=previous)) >= 2


def _trend_is_bad(
    trend: Mapping[str, float],
    current: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> bool:
    return not _trend_is_improving(trend, current=current, previous=previous)


def _is_first_gate_episode(episode: int) -> bool:
    return FIRST_GATE_MIN_EPISODE <= episode <= FIRST_GATE_MAX_EPISODE


def decide_progress_gate(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    previous_bad_gate: bool,
) -> dict[str, Any]:
    """Return stop/continue policy from hard rejects and trend signals.

    Baseline speed/resource comparison is intentionally handled by monitor and
    report tooling; this function only encodes progress-gate control flow.
    """
    reasons: list[str] = []
    notes: list[str] = []
    episode = int(current.get("episode", 0))
    if bool(current.get("has_nonfinite", False)):
        reasons.append("nonfinite_tensorboard_or_eval_metric")
    if not bool(current.get("checkpoint_exists", False)):
        reasons.append("missing_checkpoint")
    has_state_snapshot_info = "state_snapshot_exists" in current
    if (bool(current.get("state_snapshot_required", False)) or has_state_snapshot_info) and not bool(
        current.get("state_snapshot_exists", False)
    ):
        reasons.append("missing_state_snapshot")
    if reasons:
        return {"decision": "stop", "reasons": reasons, "notes": notes, "episode": episode}

    trend = current.get("trend", {})
    positive_signals = _positive_trend_signals(trend, current=current, previous=previous)
    if any(signal.startswith("fixed_eval_") for signal in positive_signals):
        notes.append("fixed_eval_quality_improved")
    if _is_first_gate_episode(episode):
        notes.append("early_ogm_can_lag_hope_20k_baseline")
        if len(positive_signals) >= 2:
            return {"decision": "continue", "reasons": [], "notes": notes, "episode": episode}
        return {"decision": "grace-10k", "reasons": ["weak_20k_trend"], "notes": notes, "episode": episode}

    if len(positive_signals) < 2:
        if previous_bad_gate:
            reasons.append("second_consecutive_bad_progress_gate")
            return {"decision": "stop", "reasons": reasons, "notes": notes, "episode": episode}
        return {"decision": "grace-10k", "reasons": ["first_bad_progress_gate"], "notes": notes, "episode": episode}

    return {"decision": "continue", "reasons": [], "notes": notes, "episode": episode}


def recommend_recovery_action(change_reasons: list[str]) -> str:
    if any(reason in _RESTART_REASONS for reason in change_reasons):
        return "restart-from-scratch"
    if change_reasons and all(reason in _RESUME_REASONS for reason in change_reasons):
        return "resume-from-checkpoint"
    return "manual-review"
