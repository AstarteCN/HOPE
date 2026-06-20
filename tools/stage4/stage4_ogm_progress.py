from __future__ import annotations

from typing import Any, Mapping


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


def decide_progress_gate(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    previous_bad_gate: bool,
) -> dict[str, Any]:
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
    if episode == 20_000:
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
    restart_reasons = {
        "ogm_rasterizer_changed",
        "observation_semantics_changed",
        "reward_semantics_changed",
        "network_shape_changed",
        "replay_buffer_contaminated",
        "state_norm_semantics_changed",
    }
    if any(reason in restart_reasons for reason in change_reasons):
        return "restart-from-scratch"
    return "resume-from-checkpoint"
