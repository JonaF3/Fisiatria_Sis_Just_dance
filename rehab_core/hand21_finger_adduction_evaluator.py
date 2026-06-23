from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FAN_KEY = "finger_fan_angle_2d"


@dataclass
class Hand21AdductionEvalState:
    phase: str = "waiting_spread"
    completed_reps: int = 0
    best_score_current_rep: float = 0.0
    last_rep: dict = field(default_factory=dict)
    spread_streak: int = 0
    together_streak: int = 0
    transition_streak: int = 0
    cooldown_frames_left: int = 0
    rep_start_timestamp_s: float = 0.0


class Hand21FingerAdductionEvaluator:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or {}
        self.spread_range = self.cfg.get("spread_range", [50.0, 90.0])
        self.together_range = self.cfg.get("together_range", [0.0, 30.0])
        self.min_stable_frames = int(self.cfg.get("min_stable_frames", 5))
        self.cooldown_frames = int(self.cfg.get("cooldown_frames", 10))
        self.min_tracking_quality = float(self.cfg.get("min_tracking_quality", 0.55))
        self.target_repetitions = int(self.cfg.get("target_repetitions", 5))
        self.stop_at_target = bool(self.cfg.get("stop_at_target", True))
        rep_duration = float(self.cfg.get("rep_duration", 8.0))
        self.min_rep_duration = float(
            self.cfg.get("min_rep_duration", max(2.0, rep_duration * 0.40))
        )
        self.state = Hand21AdductionEvalState()

    def reset_current_rep(self):
        self.state.phase = "waiting_spread"
        self.state.best_score_current_rep = 0.0
        self.state.spread_streak = 0
        self.state.together_streak = 0
        self.state.transition_streak = 0
        self.state.cooldown_frames_left = self.cooldown_frames
        self.state.rep_start_timestamp_s = 0.0

    def _update_streaks(self, is_spread: bool, is_together: bool):
        self.state.spread_streak = self.state.spread_streak + 1 if is_spread else 0
        self.state.together_streak = self.state.together_streak + 1 if is_together else 0
        self.state.transition_streak = self.state.transition_streak + 1 if (not is_spread and not is_together) else 0

    def _score_fan(self, fan_value: Optional[float], together_range: list) -> float:
        if fan_value is None:
            return 0.0
        lo, hi = float(together_range[0]), float(together_range[1])
        if lo <= fan_value <= hi:
            return 100.0
        dist = min(abs(fan_value - lo), abs(fan_value - hi))
        return max(0.0, min(100.0, 100.0 * (1.0 - dist / 60.0)))

    def evaluate(self, hand_result=None, angles: Optional[dict] = None, frame_index: int = 0, timestamp_s: float = 0.0) -> dict:
        if hand_result is not None:
            angles = getattr(hand_result, "angles", angles) or {}
            tracking_quality = float(getattr(hand_result, "tracking_quality", 0.0))
            handedness_label = getattr(hand_result, "handedness_label", "unknown")
        else:
            angles = angles or {}
            tracking_quality = 1.0 if angles else 0.0
            handedness_label = "unknown"

        fan_value = angles.get(FAN_KEY)

        is_spread_raw = fan_value is not None and self.spread_range[0] <= fan_value <= self.spread_range[1]
        is_together_raw = fan_value is not None and self.together_range[0] <= fan_value <= self.together_range[1]
        self._update_streaks(is_spread_raw, is_together_raw)

        is_spread_stable = self.state.spread_streak >= self.min_stable_frames
        is_together_stable = self.state.together_streak >= self.min_stable_frames

        together_score = self._score_fan(fan_value, self.together_range)
        self.state.best_score_current_rep = max(self.state.best_score_current_rep, together_score)

        if self.state.cooldown_frames_left > 0:
            self.state.cooldown_frames_left -= 1

        base = {
            "completed_reps": self.state.completed_reps,
            "finger_fan_angle": None if fan_value is None else round(float(fan_value), 2),
            "score": round(float(together_score), 2),
            "best_score_current_rep": round(float(self.state.best_score_current_rep), 2),
            "last_rep": dict(self.state.last_rep),
            "handedness_label": handedness_label,
            "tracking_quality": round(float(tracking_quality), 4),
            "spread_streak": self.state.spread_streak,
            "together_streak": self.state.together_streak,
            "min_stable_frames": self.min_stable_frames,
        }

        if self.stop_at_target and self.state.completed_reps >= self.target_repetitions:
            self.state.phase = "done"
            return {**base, "ok": True, "phase": "done", "rep_completed": False, "hand_state": "completado", "feedback": "Ejercicio completado."}

        if tracking_quality < self.min_tracking_quality:
            return {**base, "ok": False, "phase": self.state.phase, "rep_completed": False, "hand_state": "tracking_lost", "feedback": "Acerca la mano a la camara."}

        hand_state = "transicion"
        if is_spread_raw:
            hand_state = "dedos separados"
        elif is_together_raw:
            hand_state = "dedos juntos"

        rep_completed = False
        feedback = "Separa los dedos."

        if self.state.phase == "waiting_spread":
            if is_spread_stable:
                self.state.phase = "going_together"
                self.state.rep_start_timestamp_s = timestamp_s
                feedback = "Junta los dedos."
            else:
                feedback = f"Separa bien los dedos. Ángulo: {fan_value:.0f}° (necesitas >{self.spread_range[0]:.0f}°)"

        elif self.state.phase == "going_together":
            if is_together_stable:
                self.state.phase = "returning_spread"
                feedback = "Ahora separa los dedos otra vez."
            else:
                feedback = f"Junta mas los dedos. Ángulo: {fan_value:.0f}° (necesitas <{self.together_range[1]:.0f}°)"

        elif self.state.phase == "returning_spread":
            if self.state.cooldown_frames_left > 0:
                feedback = "Manten la mano visible."
            elif is_spread_stable:
                rep_duration_actual = timestamp_s - self.state.rep_start_timestamp_s
                too_fast = rep_duration_actual < self.min_rep_duration and self.state.rep_start_timestamp_s > 0.0

                rep_completed = True
                self.state.completed_reps += 1
                final_score = self.state.best_score_current_rep

                if too_fast:
                    speed_penalty = min(25.0, (self.min_rep_duration - rep_duration_actual) * 5.0)
                    final_score = max(0.0, final_score - speed_penalty)

                self.state.last_rep = {
                    "rep_idx": self.state.completed_reps,
                    "best_score": round(final_score, 2),
                    "raw_best_score": round(self.state.best_score_current_rep, 2),
                    "timestamp_s": float(timestamp_s),
                    "duration_s": round(rep_duration_actual, 2),
                    "too_fast": too_fast,
                    "min_duration_s": self.min_rep_duration,
                    "frame_index": int(frame_index),
                    "finger_fan_angle": None if fan_value is None else round(float(fan_value), 2),
                }
                self.reset_current_rep()
                if too_fast:
                    feedback = f"Movimiento muy rapido ({rep_duration_actual:.1f}s). Separa y junta los dedos con mas control."
                else:
                    feedback = "Repeticion valida."
            else:
                feedback = f"Separa los dedos otra vez. Ángulo: {fan_value:.0f}° (necesitas >{self.spread_range[0]:.0f}°)"

        return {**base, "ok": True, "phase": self.state.phase, "rep_completed": rep_completed, "hand_state": hand_state, "feedback": feedback}
