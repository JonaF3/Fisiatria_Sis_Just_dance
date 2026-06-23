"""
rehab_core/hand21_evaluator.py

Evaluador Hand21 v3 con validación de 5 dedos:
    mano abierta estable -> mano cerrada estable -> mano abierta estable = 1 repetición

Valida:
    - 4 dedos largos: index, middle, ring, pinky usando PIP.
    - Pulgar: thumb_ip_flexion_2d usando rango separado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PIP_KEYS = [
    "index_pip_flexion_2d",
    "middle_pip_flexion_2d",
    "ring_pip_flexion_2d",
    "pinky_pip_flexion_2d",
]

THUMB_KEY = "thumb_ip_flexion_2d"


def average_pip_flexion(angles: dict) -> Optional[float]:
    values = [float(angles[k]) for k in PIP_KEYS if k in angles and angles[k] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def value_in_range(value: Optional[float], range_pair: list | tuple) -> bool:
    if value is None or not range_pair or len(range_pair) != 2:
        return False
    lo, hi = float(range_pair[0]), float(range_pair[1])
    return lo <= float(value) <= hi


def count_in_range(angles: dict, keys: list[str], range_pair: list | tuple) -> int:
    return sum(1 for key in keys if value_in_range(angles.get(key), range_pair))


def score_closed(avg: Optional[float], closed_range: list | tuple) -> float:
    if avg is None:
        return 0.0
    lo, hi = float(closed_range[0]), float(closed_range[1])
    if lo <= avg <= hi:
        return 100.0
    dist = min(abs(avg - lo), abs(avg - hi))
    return max(0.0, min(100.0, 100.0 * (1.0 - dist / 60.0)))


@dataclass
class Hand21EvalState:
    phase: str = "waiting_open"
    completed_reps: int = 0
    reached_closed: bool = False
    best_score_current_rep: float = 0.0
    last_rep: dict = field(default_factory=dict)
    open_streak: int = 0
    closed_streak: int = 0
    transition_streak: int = 0
    cooldown_frames_left: int = 0
    rep_start_timestamp_s: float = 0.0   # marca de tiempo cuando empieza la rep


class Hand21OpenCloseEvaluator:
    """Evaluador para abrir/cerrar mano validando 5 dedos."""

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or {}
        self.open_range = self.cfg.get("open_range", [145.0, 180.0])
        self.closed_range = self.cfg.get("closed_range", [0.0, 105.0])
        self.min_open_fingers = int(self.cfg.get("min_open_fingers", 4))
        self.min_closed_fingers = int(self.cfg.get("min_closed_fingers", 4))

        self.require_thumb_open = bool(self.cfg.get("require_thumb_open", True))
        self.require_thumb_closed = bool(self.cfg.get("require_thumb_closed", True))
        self.thumb_open_range = self.cfg.get("thumb_open_range", [145.0, 180.0])
        self.thumb_closed_range = self.cfg.get("thumb_closed_range", [0.0, 165.0])

        self.min_stable_frames = int(self.cfg.get("min_stable_frames", 5))
        self.cooldown_frames = int(self.cfg.get("cooldown_frames", 10))
        self.min_tracking_quality = float(self.cfg.get("min_tracking_quality", 0.55))
        self.target_repetitions = int(self.cfg.get("target_repetitions", 5))
        self.stop_at_target = bool(self.cfg.get("stop_at_target", True))
        # Duración mínima de una rep (segundos).
        # Por defecto: 40% del rep_duration esperado, mínimo 2s.
        rep_duration = float(self.cfg.get("rep_duration", 8.0))
        self.min_rep_duration = float(
            self.cfg.get("min_rep_duration", max(2.0, rep_duration * 0.40))
        )
        self.state = Hand21EvalState()

    def reset_current_rep(self):
        self.state.phase = "waiting_open"
        self.state.reached_closed = False
        self.state.best_score_current_rep = 0.0
        self.state.open_streak = 0
        self.state.closed_streak = 0
        self.state.transition_streak = 0
        self.state.cooldown_frames_left = self.cooldown_frames
        self.state.rep_start_timestamp_s = 0.0

    def _update_streaks(self, is_open_raw: bool, is_closed_raw: bool):
        self.state.open_streak = self.state.open_streak + 1 if is_open_raw else 0
        self.state.closed_streak = self.state.closed_streak + 1 if is_closed_raw else 0
        self.state.transition_streak = self.state.transition_streak + 1 if (not is_open_raw and not is_closed_raw) else 0

    def evaluate(self, hand_result=None, angles: Optional[dict] = None, frame_index: int = 0, timestamp_s: float = 0.0) -> dict:
        if hand_result is not None:
            angles = getattr(hand_result, "angles", angles) or {}
            tracking_quality = float(getattr(hand_result, "tracking_quality", 0.0))
            handedness_label = getattr(hand_result, "handedness_label", "unknown")
        else:
            angles = angles or {}
            tracking_quality = 1.0 if angles else 0.0
            handedness_label = "unknown"

        avg = average_pip_flexion(angles)
        thumb_value = angles.get(THUMB_KEY)
        open_count = count_in_range(angles, PIP_KEYS, self.open_range)
        closed_count = count_in_range(angles, PIP_KEYS, self.closed_range)
        thumb_open = value_in_range(thumb_value, self.thumb_open_range)
        thumb_closed = value_in_range(thumb_value, self.thumb_closed_range)

        is_open_raw = (open_count >= self.min_open_fingers) and (thumb_open or not self.require_thumb_open)
        is_closed_raw = (closed_count >= self.min_closed_fingers) and (thumb_closed or not self.require_thumb_closed)
        self._update_streaks(is_open_raw, is_closed_raw)

        is_open_stable = self.state.open_streak >= self.min_stable_frames
        is_closed_stable = self.state.closed_streak >= self.min_stable_frames

        closed_score = score_closed(avg, self.closed_range)
        if self.require_thumb_closed and not thumb_closed:
            closed_score = max(0.0, closed_score - 25.0)
        self.state.best_score_current_rep = max(self.state.best_score_current_rep, closed_score)

        if self.state.cooldown_frames_left > 0:
            self.state.cooldown_frames_left -= 1

        base = {
            "completed_reps": self.state.completed_reps,
            "avg_pip_flexion": None if avg is None else round(float(avg), 2),
            "score": round(float(closed_score), 2),
            "best_score_current_rep": round(float(self.state.best_score_current_rep), 2),
            "last_rep": dict(self.state.last_rep),
            "handedness_label": handedness_label,
            "tracking_quality": round(float(tracking_quality), 4),
            "open_count": open_count,
            "closed_count": closed_count,
            "thumb_value": None if thumb_value is None else round(float(thumb_value), 2),
            "thumb_open": thumb_open,
            "thumb_closed": thumb_closed,
            "open_streak": self.state.open_streak,
            "closed_streak": self.state.closed_streak,
            "min_open_fingers": self.min_open_fingers,
            "min_closed_fingers": self.min_closed_fingers,
            "require_thumb_open": self.require_thumb_open,
            "require_thumb_closed": self.require_thumb_closed,
            "min_stable_frames": self.min_stable_frames,
        }

        if self.stop_at_target and self.state.completed_reps >= self.target_repetitions:
            self.state.phase = "done"
            return {**base, "ok": True, "phase": "done", "rep_completed": False, "hand_state": "completado", "feedback": "Ejercicio completado."}

        if tracking_quality < self.min_tracking_quality:
            return {**base, "ok": False, "phase": self.state.phase, "rep_completed": False, "hand_state": "tracking_lost", "feedback": "Acerca la mano a la camara y manten los dedos visibles."}

        hand_state = "transicion"
        if is_open_raw:
            hand_state = "mano abierta"
        elif is_closed_raw:
            hand_state = "mano cerrada"

        rep_completed = False
        feedback = "Coloca la mano abierta."

        if self.state.phase == "waiting_open":
            if is_open_stable:
                self.state.phase = "going_closed"
                self.state.rep_start_timestamp_s = timestamp_s  # inicio del cronómetro
                feedback = "Cierra los 5 dedos, incluido el pulgar."
            elif self.require_thumb_open and not thumb_open:
                feedback = f"Abre tambien el pulgar. Dedos abiertos: {open_count}/4."
            else:
                feedback = f"Abre la mano completa: {open_count}/4 dedos abiertos."

        elif self.state.phase == "going_closed":
            if is_closed_stable:
                self.state.reached_closed = True
                self.state.phase = "returning_open"
                feedback = "Ahora abre completamente los 5 dedos."
            elif self.require_thumb_closed and not thumb_closed:
                feedback = f"Cierra tambien el pulgar. Dedos cerrados: {closed_count}/4."
            else:
                feedback = f"Cierra mas los dedos: {closed_count}/4 dedos cerrados."

        elif self.state.phase == "returning_open":
            if self.state.cooldown_frames_left > 0:
                feedback = "Manten la mano visible."
            elif is_open_stable and self.state.reached_closed:
                rep_duration_actual = timestamp_s - self.state.rep_start_timestamp_s
                too_fast = rep_duration_actual < self.min_rep_duration and self.state.rep_start_timestamp_s > 0.0

                rep_completed = True
                self.state.completed_reps += 1
                final_score = self.state.best_score_current_rep

                # Penalizar reps muy rápidas proporcionalmente
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
                    "avg_pip_flexion": None if avg is None else round(float(avg), 2),
                    "open_count": open_count,
                    "closed_count": closed_count,
                    "thumb_value": None if thumb_value is None else round(float(thumb_value), 2),
                    "thumb_closed": thumb_closed,
                }
                self.reset_current_rep()
                if too_fast:
                    feedback = f"Movimiento muy rapido ({rep_duration_actual:.1f}s). Realiza la apertura y cierre con mas control."
                else:
                    feedback = "Repeticion valida con 5 dedos."
            elif self.require_thumb_open and not thumb_open:
                feedback = f"Abre tambien el pulgar. Dedos abiertos: {open_count}/4."
            else:
                feedback = f"Abre completamente la mano: {open_count}/4 dedos abiertos."

        return {**base, "ok": True, "phase": self.state.phase, "rep_completed": rep_completed, "hand_state": hand_state, "feedback": feedback}
