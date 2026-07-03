from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

MODE_OFF = "off"
MODE_DIAGNOSTIC = "diagnostic"
MODE_DIRECTION_HOLD = "direction_hold"
MODES = (MODE_OFF, MODE_DIAGNOSTIC, MODE_DIRECTION_HOLD)

APPLY_TO_RL = "rl"
APPLY_TO_RL_RS = "rl-rs"
APPLY_TO_ALL = "all"
APPLY_TO_MODES = (APPLY_TO_RL, APPLY_TO_RL_RS, APPLY_TO_ALL)


def direction_sign(speed: float, *, min_speed: float = 1e-6) -> int:
    threshold = float(min_speed)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("min_speed must be finite and non-negative")
    if abs(float(speed)) < threshold:
        return 0
    return 1 if float(speed) > 0.0 else -1


@dataclass(frozen=True)
class ManeuverStabilityConfig:
    mode: str = MODE_OFF
    apply_to: str = APPLY_TO_RL
    min_speed: float = 1e-6
    hold_steps: int = 1
    max_hold_speed: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("Unsupported maneuver stability mode: %r" % self.mode)
        if self.apply_to not in APPLY_TO_MODES:
            raise ValueError("Unsupported maneuver stability apply_to: %r" % self.apply_to)
        min_speed = float(self.min_speed)
        if not math.isfinite(min_speed) or min_speed < 0:
            raise ValueError("maneuver stability min_speed must be finite and non-negative")
        if not isinstance(self.hold_steps, int) or isinstance(self.hold_steps, bool):
            raise ValueError("maneuver stability hold_steps must be an integer")
        if self.mode == MODE_DIRECTION_HOLD and self.hold_steps < 1:
            raise ValueError("direction_hold requires hold_steps >= 1")
        if self.max_hold_speed is not None:
            max_hold_speed = float(self.max_hold_speed)
            if not math.isfinite(max_hold_speed) or max_hold_speed < 0:
                raise ValueError("maneuver stability max_hold_speed must be finite and non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ManeuverStabilityConfig":
        if value is None:
            return cls()
        return cls(
            mode=str(value.get("mode", MODE_OFF)),
            apply_to=str(value.get("apply_to", APPLY_TO_RL)),
            min_speed=float(value.get("min_speed", 1e-6)),
            hold_steps=int(value.get("hold_steps", 1)),
            max_hold_speed=None
            if value.get("max_hold_speed") is None
            else float(value.get("max_hold_speed")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "apply_to": self.apply_to,
            "min_speed": float(self.min_speed),
            "hold_steps": int(self.hold_steps),
            "max_hold_speed": None if self.max_hold_speed is None else float(self.max_hold_speed),
        }


@dataclass(frozen=True)
class ManeuverStabilityResult:
    raw_action: np.ndarray
    applied_action: np.ndarray
    source: str
    raw_direction: int
    applied_direction: int
    changed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "raw_action": [float(value) for value in self.raw_action.tolist()],
            "applied_action": [float(value) for value in self.applied_action.tolist()],
            "raw_direction": int(self.raw_direction),
            "applied_direction": int(self.applied_direction),
            "changed": bool(self.changed),
            "reason": self.reason,
        }


class ManeuverStabilityTracker:
    def __init__(self, config: ManeuverStabilityConfig | None = None) -> None:
        self.config = config or ManeuverStabilityConfig()
        self._last_direction = 0
        self._held_flip_count = 0
        self._counters: dict[str, Any] = {
            "steps": 0,
            "speed_sign_flips": 0,
            "would_change": 0,
            "interventions": 0,
            "by_source": {},
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any] | None) -> "ManeuverStabilityTracker":
        if snapshot is None:
            return cls()
        tracker = cls(ManeuverStabilityConfig.from_mapping(snapshot.get("config")))
        tracker._last_direction = int(snapshot.get("last_direction", 0))
        tracker._held_flip_count = int(snapshot.get("held_flip_count", 0))
        counters = snapshot.get("counters", {})
        tracker._counters = {
            "steps": int(counters.get("steps", 0)),
            "speed_sign_flips": int(counters.get("speed_sign_flips", 0)),
            "would_change": int(counters.get("would_change", 0)),
            "interventions": int(counters.get("interventions", 0)),
            "by_source": {
                str(source): {
                    "speed_sign_flips": int(source_counters.get("speed_sign_flips", 0)),
                    "would_change": int(source_counters.get("would_change", 0)),
                    "interventions": int(source_counters.get("interventions", 0)),
                }
                for source, source_counters in counters.get("by_source", {}).items()
            },
        }
        return tracker

    def apply(self, action: Any, *, source: str = "RL") -> ManeuverStabilityResult:
        raw = np.asarray(action, dtype=float).reshape(-1).copy()
        if raw.size < 2:
            raise ValueError("maneuver stability action must include speed at index 1")

        applied = raw.copy()
        source_name = str(source)
        raw_direction = direction_sign(float(raw[1]), min_speed=self.config.min_speed)
        enabled = self._source_enabled(source_name)
        changed = False
        reason = self.config.mode

        if self.config.mode == MODE_OFF:
            return ManeuverStabilityResult(
                raw_action=raw,
                applied_action=applied,
                source=source_name,
                raw_direction=raw_direction,
                applied_direction=raw_direction,
                changed=False,
                reason="off",
            )

        self._counters["steps"] += 1
        if not enabled:
            reason = "diagnostic_source_not_enabled" if self.config.mode == MODE_DIAGNOSTIC else "source_not_enabled"
            return ManeuverStabilityResult(
                raw_action=raw,
                applied_action=applied,
                source=source_name,
                raw_direction=raw_direction,
                applied_direction=raw_direction,
                changed=False,
                reason=reason,
            )

        flip = self._is_flip(raw_direction)
        if flip:
            self._counters["speed_sign_flips"] += 1
            self._source_counters(source_name)["speed_sign_flips"] += 1

        if self.config.mode == MODE_DIAGNOSTIC:
            if flip and enabled:
                self._record_would_change(source_name)
                reason = "diagnostic_speed_sign_flip"
            else:
                reason = "diagnostic_no_flip"
            self._accept_direction(raw_direction)
        elif flip and not self._within_hold_speed_limit(float(raw[1])):
            reason = "direction_hold_speed_not_held"
            self._accept_direction(raw_direction)
        elif flip and self._held_flip_count < self.config.hold_steps:
            applied[1] = abs(float(raw[1])) * self._last_direction
            changed = True
            reason = "direction_hold"
            self._held_flip_count += 1
            self._record_would_change(source_name)
            self._counters["interventions"] += 1
            self._source_counters(source_name)["interventions"] += 1
        else:
            reason = "direction_hold_no_change"
            self._accept_direction(raw_direction)

        applied_direction = direction_sign(float(applied[1]), min_speed=self.config.min_speed)
        return ManeuverStabilityResult(
            raw_action=raw,
            applied_action=applied,
            source=source_name,
            raw_direction=raw_direction,
            applied_direction=applied_direction,
            changed=changed,
            reason=reason,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "last_direction": int(self._last_direction),
            "held_flip_count": int(self._held_flip_count),
            "counters": self._copy_counters(),
        }

    def reset_episode(self) -> None:
        self._last_direction = 0
        self._held_flip_count = 0

    def _is_flip(self, direction: int) -> bool:
        return self._last_direction != 0 and direction != 0 and direction != self._last_direction

    def _source_enabled(self, source: str) -> bool:
        normalized = source.upper()
        if self.config.apply_to == APPLY_TO_ALL:
            return True
        if self.config.apply_to == APPLY_TO_RL_RS:
            return normalized in {"RL", "RS"}
        return normalized == "RL"

    def _within_hold_speed_limit(self, speed: float) -> bool:
        if self.config.max_hold_speed is None:
            return True
        return abs(float(speed)) <= float(self.config.max_hold_speed)

    def _accept_direction(self, direction: int) -> None:
        if direction == 0:
            return
        self._last_direction = direction
        self._held_flip_count = 0

    def _record_would_change(self, source: str) -> None:
        self._counters["would_change"] += 1
        self._source_counters(source)["would_change"] += 1

    def _source_counters(self, source: str) -> dict[str, int]:
        by_source = self._counters["by_source"]
        if source not in by_source:
            by_source[source] = {"speed_sign_flips": 0, "would_change": 0, "interventions": 0}
        return by_source[source]

    def _copy_counters(self) -> dict[str, Any]:
        return {
            "steps": int(self._counters["steps"]),
            "speed_sign_flips": int(self._counters["speed_sign_flips"]),
            "would_change": int(self._counters["would_change"]),
            "interventions": int(self._counters["interventions"]),
            "by_source": {
                source: {name: int(value) for name, value in counters.items()}
                for source, counters in self._counters["by_source"].items()
            },
        }
