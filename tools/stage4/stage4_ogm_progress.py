from __future__ import annotations

from typing import Any, Mapping

FIRST_GATE_MIN_EPISODE = 19_500
FIRST_GATE_MAX_EPISODE = 20_500

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


def _trend_is_improving(trend: Mapping[str, float]) -> bool:
    positive = 0
    if float(trend.get("avg_reward_delta", 0.0)) > 0.02:
        positive += 1
    if float(trend.get("mean_psr_delta", 0.0)) > 0.02:
        positive += 1
    if float(trend.get("step_num_improvement", 0.0)) > 0.05:
        positive += 1
    return positive >= 2


def _trend_is_bad(trend: Mapping[str, float]) -> bool:
    return not _trend_is_improving(trend)


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
    if reasons:
        return {"decision": "stop", "reasons": reasons, "notes": notes, "episode": episode}

    trend = current.get("trend", {})
    if _is_first_gate_episode(episode):
        notes.append("early_ogm_can_lag_hope_20k_baseline")
        if _trend_is_improving(trend):
            return {"decision": "continue", "reasons": [], "notes": notes, "episode": episode}
        return {"decision": "grace-10k", "reasons": ["weak_20k_trend"], "notes": notes, "episode": episode}

    if _trend_is_bad(trend):
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
