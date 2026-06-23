from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

PIP_KEYS = ["index_pip_flexion_2d", "middle_pip_flexion_2d", "ring_pip_flexion_2d", "pinky_pip_flexion_2d"]
THUMB_KEY = "thumb_ip_flexion_2d"
EVALUATOR_VERSION = "static_wrist_no_openclose_release_step25"

def _in_range(value: Optional[float], range_pair) -> bool:
    if value is None or range_pair is None or len(range_pair) != 2:
        return False
    return float(range_pair[0]) <= float(value) <= float(range_pair[1])

def _angle_of_vector_deg(a, b):
    dx = float(b.x - a.x); dy = float(b.y - a.y)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return round(math.degrees(math.atan2(dy, dx)), 2)

def _from_horizontal(angle):
    if angle is None:
        return None
    a = abs(float(angle))
    if a > 90.0:
        a = 180.0 - a
    return round(a, 2)

def _avg_pip(angles):
    vals = [float(angles[k]) for k in PIP_KEYS if angles.get(k) is not None]
    return None if not vals else sum(vals) / len(vals)

def _open_count(angles, open_range):
    return sum(1 for k in PIP_KEYS if _in_range(angles.get(k), open_range))

def compute_static_wrist_metrics(hand_result):
    lm = getattr(hand_result, "landmarks", {}) or {}
    angles = getattr(hand_result, "angles", {}) or {}
    metrics = {}
    wrist = lm.get("wrist"); middle_mcp = lm.get("middle_mcp"); middle_tip = lm.get("middle_tip")
    if wrist is not None and middle_mcp is not None:
        axis = _angle_of_vector_deg(wrist, middle_mcp)
        metrics["hand_axis_angle_2d"] = axis
        metrics["hand_axis_from_horizontal"] = _from_horizontal(axis)
    if wrist is not None and middle_tip is not None:
        long_axis = _angle_of_vector_deg(wrist, middle_tip)
        metrics["wrist_to_middle_tip_angle_2d"] = long_axis
        metrics["wrist_to_middle_tip_from_horizontal"] = _from_horizontal(long_axis)
    avg = _avg_pip(angles)
    metrics["avg_pip_flexion_2d"] = None if avg is None else round(avg, 2)
    metrics["thumb_ip_flexion_2d"] = angles.get(THUMB_KEY)
    return metrics

@dataclass
class StaticWristState:
    completed_reps: int = 0
    stable_frames: int = 0
    release_frames: int = 0
    cooldown_frames_left: int = 0
    armed: bool = False
    phase: str = "waiting_release"
    best_score_current_rep: float = 0.0
    last_rep: dict = field(default_factory=dict)

class Hand21StaticWristAlignmentEvaluator:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or {}
        self.state = StaticWristState()
        self.min_tracking_quality = float(self.cfg.get("min_tracking_quality", 0.75))
        self.open_range = self.cfg.get("open_range", [165.0, 180.0])
        self.min_open_fingers = int(self.cfg.get("min_open_fingers", 4))
        self.require_thumb_open = bool(self.cfg.get("require_thumb_open", True))
        self.thumb_open_range = self.cfg.get("thumb_open_range", [155.0, 180.0])
        self.avg_pip_range = self.cfg.get("avg_pip_range", [168.0, 180.0])
        self.axis_target_range = self.cfg.get("axis_target_range", [0.0, 32.0])
        self.release_axis_range = self.cfg.get("release_axis_range", [40.0, 90.0])
        self.release_requires_fingers_open = bool(self.cfg.get("release_requires_fingers_open", True))
        self.min_stable_frames = int(self.cfg.get("min_stable_frames", 8))
        self.release_frames_required = int(self.cfg.get("release_frames", 6))
        self.cooldown_frames = int(self.cfg.get("cooldown_frames", 8))
        self.require_release_before_count = bool(self.cfg.get("require_release_before_count", True))
        print(f"[HAND21_STATIC] Evaluator activo: {EVALUATOR_VERSION}")

    def _score(self, axis, avg_pip, open_count, thumb_open):
        score = 0.0
        score += 45.0 if _in_range(axis, self.axis_target_range) else 0.0
        score += 30.0 if _in_range(avg_pip, self.avg_pip_range) else 0.0
        score += min(20.0, 20.0 * (open_count / max(1, self.min_open_fingers)))
        score += 5.0 if (thumb_open or not self.require_thumb_open) else 0.0
        return round(score, 2)

    def evaluate(self, hand_result=None, frame_index: int = 0, timestamp_s: float = 0.0, **kwargs):
        if hand_result is None:
            self.state.stable_frames = 0; self.state.release_frames = 0
            return self._result(False, 0.0, "sin mano", "Muestra la mano a la camara.")

        q = float(getattr(hand_result, "tracking_quality", 0.0))
        angles = getattr(hand_result, "angles", {}) or {}
        metrics = compute_static_wrist_metrics(hand_result)
        axis = metrics.get("wrist_to_middle_tip_from_horizontal")
        avg_pip = metrics.get("avg_pip_flexion_2d")
        thumb_value = metrics.get("thumb_ip_flexion_2d")
        open_count = _open_count(angles, self.open_range)
        thumb_open = _in_range(thumb_value, self.thumb_open_range)
        fingers_ok = open_count >= self.min_open_fingers and _in_range(avg_pip, self.avg_pip_range) and (thumb_open or not self.require_thumb_open)
        axis_target_ok = _in_range(axis, self.axis_target_range)
        axis_release_ok = _in_range(axis, self.release_axis_range)
        target_ok = q >= self.min_tracking_quality and fingers_ok and axis_target_ok
        release_ok = q >= self.min_tracking_quality and axis_release_ok and (fingers_ok if self.release_requires_fingers_open else True)
        score = self._score(axis, avg_pip, open_count, thumb_open)
        rep_completed = False
        hand_state = "buscando"
        feedback = "Extiende dedos y alinea la mano."

        if q < self.min_tracking_quality:
            self.state.stable_frames = 0; self.state.release_frames = 0; self.state.armed = False
            hand_state = "tracking_lost"; feedback = "Acerca la mano a la camara."
        elif not fingers_ok:
            # Cerrar dedos invalida y NO arma repetición.
            self.state.stable_frames = 0; self.state.release_frames = 0; self.state.armed = False; self.state.phase = "waiting_release"
            hand_state = "dedos_invalidos"
            feedback = f"No cierres los dedos. Extiende dedos: {open_count}/4."
        elif self.state.phase == "waiting_release" and self.require_release_before_count:
            self.state.stable_frames = 0
            if release_ok:
                self.state.release_frames += 1
                hand_state = "release_orientacion"
                feedback = f"Inicio por orientacion: {self.state.release_frames}/{self.release_frames_required}."
                if self.state.release_frames >= self.release_frames_required:
                    self.state.armed = True; self.state.phase = "waiting_target"
                    feedback = "Ahora vuelve a la postura objetivo."
            else:
                self.state.release_frames = 0
                hand_state = "esperando_release_orientacion"
                feedback = "Cambia la orientacion de la mano sin cerrar dedos."
        elif self.state.cooldown_frames_left > 0:
            self.state.cooldown_frames_left -= 1
            hand_state = "cooldown"
            feedback = "Sal de la postura objetivo cambiando orientacion."
            if self.state.cooldown_frames_left <= 0:
                self.state.phase = "waiting_release"; self.state.armed = False
        elif target_ok and (self.state.armed or not self.require_release_before_count):
            self.state.stable_frames += 1; self.state.release_frames = 0
            self.state.best_score_current_rep = max(self.state.best_score_current_rep, score)
            hand_state = "postura_correcta"
            feedback = f"Manten la mano correcta: {self.state.stable_frames}/{self.min_stable_frames}."
            if self.state.stable_frames >= self.min_stable_frames:
                rep_completed = True
                self.state.completed_reps += 1
                self.state.last_rep = {"rep_idx": self.state.completed_reps, "best_score": round(max(score, self.state.best_score_current_rep), 2), "axis_from_horizontal": axis, "avg_pip_flexion_2d": avg_pip, "open_count": open_count, "tracking_quality": round(q, 4)}
                self.state.cooldown_frames_left = self.cooldown_frames
                self.state.phase = "cooldown"; self.state.armed = False; self.state.stable_frames = 0; self.state.release_frames = 0; self.state.best_score_current_rep = 0.0
                feedback = "Repeticion valida: orientacion correcta con dedos extendidos."
        else:
            self.state.stable_frames = 0
            if self.state.armed: self.state.phase = "waiting_target"
            feedback = f"Alinea la mano al objetivo. Angulo={axis}." if not axis_target_ok else "Manten la postura objetivo."

        return {"ok": True, "phase": self.state.phase, "rep_completed": rep_completed, "completed_reps": self.state.completed_reps, "score": score, "best_score_current_rep": round(max(score, self.state.best_score_current_rep), 2), "last_rep": dict(self.state.last_rep), "feedback": feedback, "hand_state": hand_state, "tracking_quality": round(q, 4), "open_count": open_count, "closed_count": 0, "avg_pip_flexion": avg_pip, "avg_pip_flexion_2d": avg_pip, "thumb_value": None if thumb_value is None else round(float(thumb_value), 2), "thumb_open": thumb_open, "thumb_closed": False, "axis_from_horizontal": axis, "wrist_to_middle_tip_from_horizontal": axis, "stable_frames": self.state.stable_frames, "min_stable_frames": self.min_stable_frames, "release_frames": self.state.release_frames, "release_frames_required": self.release_frames_required, "armed": self.state.armed}

    def _result(self, rep_completed, score, hand_state, feedback):
        return {"ok": False, "phase": self.state.phase, "rep_completed": rep_completed, "completed_reps": self.state.completed_reps, "score": score, "best_score_current_rep": round(self.state.best_score_current_rep, 2), "last_rep": dict(self.state.last_rep), "feedback": feedback, "hand_state": hand_state}
