"""
rehab_core/pose33_trajectory_evaluator.py

Evaluador de trayectoria para MediaPipe Pose 33.

Logica clinica: start -> target -> return
usando angulos calculados por rehab_core.angles_33.

v4:
  - Histeresis por frames: evita falsos positivos por ruido de un solo frame.
  - Suavizado temporal (EMA) del angulo principal/confirmacion.
  - Auto-calibracion del neutro del paciente: deriva start/target/safe del
    angulo en reposo, adaptado al cuerpo de cada persona.
  - Modo neck_primary_only / body_region == CUELLO: NO penaliza ni evalua
    tronco, hombro, codo, etc. Solo cuenta el angulo de cuello.
  - Duracion minima de rep sigue activa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _in_range(value: Optional[float], range_pair: Optional[list | tuple]) -> bool:
    if value is None or range_pair is None or len(range_pair) != 2:
        return False
    lo, hi = float(range_pair[0]), float(range_pair[1])
    return lo <= float(value) <= hi


def _range_center(range_pair: list | tuple) -> float:
    return (float(range_pair[0]) + float(range_pair[1])) / 2.0


def _distance_to_range(value: float, range_pair: list | tuple) -> float:
    lo, hi = float(range_pair[0]), float(range_pair[1])
    if lo <= value <= hi:
        return 0.0
    return min(abs(value - lo), abs(value - hi))


def _score_from_distance(value: Optional[float], target_range: Optional[list | tuple], tolerance: float = 45.0) -> float:
    if value is None or target_range is None:
        return 0.0
    dist = _distance_to_range(float(value), target_range)
    return max(0.0, min(100.0, 100.0 * (1.0 - dist / max(tolerance, 1e-6))))


@dataclass
class Pose33EvalState:
    phase: str = "waiting_start"
    started: bool = False
    reached_target: bool = False
    completed_reps: int = 0
    best_score_current_rep: float = 0.0
    last_rep: dict = field(default_factory=dict)
    rep_start_timestamp_s: float = 0.0
    # Histeresis: contadores de frames consecutivos
    frames_in_start: int = 0
    frames_in_target: int = 0
    frames_returning_start: int = 0
    # Suavizado temporal (EMA) del angulo principal / de confirmacion
    ema_primary: Optional[float] = None
    ema_confirmation: Optional[float] = None
    # Auto-calibracion del neutro del paciente
    calibrated: bool = False
    neutral_value: Optional[float] = None
    calib_samples: list = field(default_factory=list)
    dyn_start_range: Optional[list] = None
    dyn_target_range: Optional[list] = None
    dyn_safe_range: Optional[list] = None


class Pose33TrajectoryEvaluator:
    """Evaluador de trayectoria para MediaPipe 33."""

    # Histeresis por defecto (frames consecutivos requeridos)
    DEFAULT_MIN_FRAMES_IN_TARGET = 3   # frames para confirmar llegada a target
    DEFAULT_MIN_FRAMES_IN_START  = 2   # frames para confirmar posicion inicial
    DEFAULT_MIN_FRAMES_RETURN    = 2   # frames para confirmar retorno al inicio

    def __init__(self, exercise_config: dict):
        self.cfg = exercise_config or {}
        self.state = Pose33EvalState()

        self.primary_angle = self.cfg.get("primary_angle")
        self.confirmation_angle = self.cfg.get("confirmation_angle")
        self.start_ranges = self.cfg.get("start_ranges", {}) or {}
        self.target_ranges = self.cfg.get("target_ranges", {}) or {}
        self.safe_ranges = self.cfg.get("safe_ranges", {}) or {}
        self.confirmation_target_ranges = self.cfg.get("confirmation_target_ranges", {}) or {}
        self.required_landmarks = self.cfg.get("required_landmarks", []) or []

        # Estrategia de validacion / region corporal.
        # En modo "neck_primary_only" (o body_region == CUELLO) NO se penalizan
        # ni evaluan tronco, hombro, codo, etc: solo cuenta el angulo de cuello.
        self.body_region = str(self.cfg.get("body_region", "")).upper()
        self.validation_strategy = str(self.cfg.get("validation_strategy", "")).lower()
        self.compensation_rules = self.cfg.get("compensation_rules", []) or []
        self.neck_primary_only = (
            self.body_region == "CUELLO"
            or self.validation_strategy == "neck_primary_only"
        )

        self.min_tracking_confidence = float(self.cfg.get("min_tracking_confidence", 0.35))
        self.valid_rep_score = float(self.cfg.get("valid_rep_score", 70.0))
        self.feedback = self.cfg.get("feedback", {}) or {}
        self.gate_ranges = self.cfg.get("gate_ranges", {}) or {}
        self.gate_feedback = str(self.cfg.get("gate_feedback", "Ajusta la posicion para validar el ejercicio."))

        rep_duration = float(self.cfg.get("rep_duration", 10.0))
        self.min_rep_duration = float(
            self.cfg.get("min_rep_duration", max(3.0, rep_duration * 0.40))
        )

        # Histeresis configurables por ejercicio
        self.min_frames_in_target = int(
            self.cfg.get("min_frames_in_target", self.DEFAULT_MIN_FRAMES_IN_TARGET)
        )
        self.min_frames_in_start = int(
            self.cfg.get("min_frames_in_start", self.DEFAULT_MIN_FRAMES_IN_START)
        )
        self.min_frames_return = int(
            self.cfg.get("min_frames_return", self.DEFAULT_MIN_FRAMES_RETURN)
        )

        # Suavizado temporal (EMA). 0 = sin suavizado; 0.6-0.7 = bastante suave.
        # new = alpha * anterior + (1 - alpha) * crudo
        self.angle_smoothing = float(self.cfg.get("angle_smoothing", 0.0))
        self.angle_smoothing = max(0.0, min(0.95, self.angle_smoothing))

        # Auto-calibracion: adapta el neutro y el objetivo al cuerpo del paciente.
        self.auto_calibrate = bool(self.cfg.get("auto_calibrate", False))
        self.calib_frames = int(self.cfg.get("calib_frames", 15))
        self.calib_max_jitter = float(self.cfg.get("calib_max_jitter", 7.0))
        self.target_delta = float(self.cfg.get("target_delta", 25.0))
        self.return_tolerance = float(self.cfg.get("return_tolerance", 8.0))
        self.safe_margin = float(self.cfg.get("safe_margin", 60.0))
        self.calib_timeout_frames = int(self.cfg.get("calib_timeout_frames", max(30, self.calib_frames * 3)))
        # "increasing": el angulo crece desde el neutro hacia el objetivo.
        # "decreasing": el angulo decrece desde el neutro hacia el objetivo.
        self.direction = str(self.cfg.get("direction", "increasing")).lower()

        if not self.primary_angle:
            raise ValueError("Pose33TrajectoryEvaluator requiere primary_angle en config")

    # ------------------------------------------------------------------
    # Suavizado y auto-calibracion
    # ------------------------------------------------------------------

    def _smooth(self, raw: Optional[float], which: str) -> Optional[float]:
        """Aplica EMA al valor crudo. `which` es 'primary' o 'confirmation'."""
        if raw is None:
            return None
        if self.angle_smoothing <= 0.0:
            return raw
        prev = getattr(self.state, f"ema_{which}")
        if prev is None:
            new = float(raw)
        else:
            a = self.angle_smoothing
            new = a * float(prev) + (1.0 - a) * float(raw)
        setattr(self.state, f"ema_{which}", new)
        return round(new, 2)

    def _build_dynamic_ranges(self, neutral: float) -> None:
        """Construye start/target/safe relativos al neutro calibrado del paciente.
        Garantiza un ancho minimo de 15° para el rango target y 10° para el start."""
        tol = max(self.return_tolerance, 5.0)
        delta = self.target_delta
        margin = max(self.safe_margin, 15.0)
        if self.direction == "decreasing":
            self.state.dyn_start_range = [neutral - tol, neutral + tol]
            self.state.dyn_target_range = [neutral - delta - margin, neutral - delta]
            self.state.dyn_safe_range = [neutral - delta - margin - 20.0, neutral + max(tol, 20.0) + 20.0]
        else:  # increasing
            self.state.dyn_start_range = [neutral - tol, neutral + tol]
            self.state.dyn_target_range = [neutral + delta, neutral + delta + margin]
            self.state.dyn_safe_range = [neutral - max(tol, 20.0) - 20.0, neutral + delta + margin + 20.0]

    def _update_calibration(self, value: Optional[float]) -> bool:
        """
        Acumula muestras del neutro mientras el paciente esta quieto.
        Devuelve True cuando la calibracion queda completada en este frame.
        Si se alcanza calib_timeout_frames, fuerza la calibracion con la mediana
        actual aunque el jitter sea alto, para no bloquear el ejercicio.
        """
        if value is None:
            return False
        samples = self.state.calib_samples
        samples.append(float(value))
        if len(samples) > self.calib_frames:
            samples.pop(0)
        if len(samples) >= self.calib_frames:
            spread = max(samples) - min(samples)
            if spread <= self.calib_max_jitter:
                ordered = sorted(samples)
                neutral = ordered[len(ordered) // 2]  # mediana
                self.state.neutral_value = round(float(neutral), 2)
                self._build_dynamic_ranges(self.state.neutral_value)
                self.state.calibrated = True
                return True
        if len(samples) >= self.calib_timeout_frames:
            ordered = sorted(samples)
            neutral = ordered[len(ordered) // 2]
            self.state.neutral_value = round(float(neutral), 2)
            self._build_dynamic_ranges(self.state.neutral_value)
            self.state.calibrated = True
            return True
        return False

    def reset_current_rep(self) -> None:
        self.state.phase = "waiting_start"
        self.state.started = False
        self.state.reached_target = False
        self.state.best_score_current_rep = 0.0
        self.state.rep_start_timestamp_s = 0.0
        self.state.frames_in_start = 0
        self.state.frames_in_target = 0
        self.state.frames_returning_start = 0

    def _feedback(self, key: str, fallback: str) -> str:
        return str(self.feedback.get(key, fallback))

    def _tracking_ok(self, pose33_result: Any = None, tracking_quality: Optional[dict] = None) -> bool:
        if tracking_quality and isinstance(tracking_quality, dict):
            return bool(tracking_quality.get("ok", False))
        if pose33_result is not None and hasattr(pose33_result, "required_visible"):
            return pose33_result.required_visible(self.required_landmarks, self.min_tracking_confidence)
        return True

    def _gates_ok(self, angles: dict) -> tuple[bool, dict]:
        details = {}
        for key, range_pair in self.gate_ranges.items():
            value = angles.get(key)
            ok = _in_range(value, range_pair)
            details[key] = {"value": value, "range": range_pair, "ok": ok}
            if not ok:
                return False, details
        return True, details

    def _confirmation_in_target(self, confirmation_value: Optional[float], primary_target_range: Optional[list]) -> bool:
        if self.confirmation_angle is None or confirmation_value is None:
            return True
        explicit_range = self.confirmation_target_ranges.get(self.confirmation_angle)
        if explicit_range:
            return _in_range(confirmation_value, explicit_range)
        if primary_target_range and len(primary_target_range) == 2:
            lo = float(primary_target_range[0]) - 20.0
            hi = float(primary_target_range[1]) + 20.0
            return lo <= float(confirmation_value) <= hi
        return True

    def _compensation_flags(self, angles: dict) -> dict:
        flags = {}

        # Cuello: no se evaluan compensaciones de cuerpo (solo cuenta el cuello).
        if self.neck_primary_only or not self.compensation_rules:
            return flags

        trunk = angles.get("trunk_lean_2d")
        trunk_safe = self.safe_ranges.get("trunk_lean_2d")
        if trunk is not None and trunk_safe is not None:
            flags["trunk_lean"] = not _in_range(trunk, trunk_safe)

        for elbow_key in ("right_elbow_flexion_3d", "right_elbow_flexion_2d",
                          "left_elbow_flexion_3d", "left_elbow_flexion_2d"):
            if elbow_key in angles and elbow_key in self.safe_ranges:
                flags["elbow_bend"] = not _in_range(angles.get(elbow_key), self.safe_ranges.get(elbow_key))
                break

        shoulder_align = angles.get("shoulder_alignment_2d")
        shoulder_safe = self.safe_ranges.get("shoulder_alignment_2d")
        if shoulder_align is not None and shoulder_safe is not None:
            flags["shoulder_hike"] = not _in_range(shoulder_align, shoulder_safe)

        return flags

    def _compensation_penalty(self, angles: dict) -> float:
        penalty = 0.0

        # Cuello: nunca se penaliza por tronco/hombro/codo.
        if self.neck_primary_only or not self.compensation_rules:
            return 0.0

        trunk = angles.get("trunk_lean_2d")
        trunk_safe = self.safe_ranges.get("trunk_lean_2d")
        if trunk is not None and trunk_safe is not None:
            dist = _distance_to_range(float(trunk), trunk_safe)
            penalty += min(20.0, dist * 0.80)

        for elbow_key in ("right_elbow_flexion_3d", "right_elbow_flexion_2d",
                          "left_elbow_flexion_3d", "left_elbow_flexion_2d"):
            if elbow_key in angles and elbow_key in self.safe_ranges:
                dist = _distance_to_range(float(angles[elbow_key]), self.safe_ranges[elbow_key])
                penalty += min(12.0, dist * 0.50)
                break

        shoulder_align = angles.get("shoulder_alignment_2d")
        shoulder_safe = self.safe_ranges.get("shoulder_alignment_2d")
        if shoulder_align is not None and shoulder_safe is not None:
            dist = _distance_to_range(float(shoulder_align), shoulder_safe)
            penalty += min(12.0, dist * 0.60)

        return min(40.0, penalty)

    def evaluate(self, angles: dict, pose33_result: Any = None, tracking_quality: Optional[dict] = None,
                 frame_index: int = 0, timestamp_s: float = 0.0) -> dict:
        """
        Evalua un frame con histeresis para evitar falsos positivos por ruido.
        """
        angles = angles or {}
        # Valor crudo + suavizado temporal (EMA) para reducir el ruido frame a frame.
        primary_value = self._smooth(angles.get(self.primary_angle), "primary")
        confirmation_value = (
            self._smooth(angles.get(self.confirmation_angle), "confirmation")
            if self.confirmation_angle else None
        )

        tracking_ok = self._tracking_ok(pose33_result, tracking_quality)
        if not tracking_ok:
            # Resetear contadores de histeresis si se pierde el tracking
            self.state.frames_in_start = 0
            self.state.frames_in_target = 0
            self.state.frames_returning_start = 0
            return {
                "ok": False,
                "phase": self.state.phase,
                "rep_completed": False,
                "completed_reps": self.state.completed_reps,
                "score": 0.0,
                "primary_angle": self.primary_angle,
                "primary_value": primary_value,
                "feedback": self._feedback("lost_tracking", "Ajusta la camara para ver los puntos requeridos."),
                "compensations": {},
            }

        # --- Auto-calibracion del neutro del paciente -----------------------
        if self.auto_calibrate and not self.state.calibrated:
            just_done = self._update_calibration(primary_value)
            if not just_done:
                return {
                    "ok": True,
                    "phase": "calibrating",
                    "rep_completed": False,
                    "completed_reps": self.state.completed_reps,
                    "score": 0.0,
                    "best_score_current_rep": 0.0,
                    "primary_angle": self.primary_angle,
                    "primary_value": primary_value,
                    "calibrating": True,
                    "calib_progress": len(self.state.calib_samples),
                    "calib_frames": self.calib_frames,
                    "feedback": self._feedback(
                        "calibrating",
                        "Mantente quieto en posicion neutra para calibrar...",
                    ),
                    "compensations": {},
                }

        if self.auto_calibrate and self.state.calibrated:
            start_range = self.state.dyn_start_range
            target_range = self.state.dyn_target_range
            safe_range = self.state.dyn_safe_range
        else:
            start_range = self.start_ranges.get(self.primary_angle)
            target_range = self.target_ranges.get(self.primary_angle)
            safe_range = self.safe_ranges.get(self.primary_angle)

        in_start = _in_range(primary_value, start_range)
        in_target_primary = _in_range(primary_value, target_range)
        in_target = in_target_primary and self._confirmation_in_target(confirmation_value, target_range)
        in_safe = _in_range(primary_value, safe_range) if safe_range else True

        gates_ok, gate_details = self._gates_ok(angles)
        if not gates_ok:
            self.state.frames_in_start = 0
            self.state.frames_in_target = 0
            self.state.frames_returning_start = 0
            return {
                "ok": True,
                "phase": self.state.phase,
                "rep_completed": False,
                "completed_reps": self.state.completed_reps,
                "score": 0.0,
                "best_score_current_rep": round(self.state.best_score_current_rep, 2),
                "last_rep": dict(self.state.last_rep),
                "primary_angle": self.primary_angle,
                "primary_value": primary_value,
                "confirmation_angle": self.confirmation_angle,
                "confirmation_value": confirmation_value,
                "calibrated": self.state.calibrated,
                "neutral_value": self.state.neutral_value,
                "in_start": in_start,
                "in_target": False,
                "in_target_primary_only": False,
                "in_safe": in_safe,
                "gate_ok": False,
                "gate_details": gate_details,
                "feedback_key": "gate_failed",
                "feedback": self.gate_feedback,
                "compensations": {},
            }


        target_score = _score_from_distance(primary_value, target_range)
        self.state.best_score_current_rep = max(self.state.best_score_current_rep, target_score)

        compensations = self._compensation_flags(angles)
        has_compensation = any(compensations.values()) if compensations else False

        rep_completed = False
        too_fast = False
        feedback_key = "waiting_start"
        feedback = self._feedback("waiting_start", "Coloca la extremidad en posicion inicial.")

        if not in_safe:
            self.state.frames_in_start = 0
            self.state.frames_in_target = 0
            self.state.frames_returning_start = 0
            feedback = self._feedback("rep_invalid", "Movimiento fuera del rango seguro.")

        elif self.state.phase == "waiting_start":
            if in_start:
                self.state.frames_in_start += 1
                if self.state.frames_in_start >= self.min_frames_in_start:
                    self.state.started = True
                    self.state.phase = "going_target"
                    self.state.frames_in_start = 0
                    self.state.rep_start_timestamp_s = timestamp_s
                    feedback_key = "go_to_target"
                    feedback = self._feedback("go_to_target", "Realiza el movimiento hacia el objetivo.")
                else:
                    feedback_key = "waiting_start"
                    feedback = self._feedback("waiting_start", "Coloca la extremidad en posicion inicial.")
            else:
                self.state.frames_in_start = 0
                feedback_key = "waiting_start"
                feedback = self._feedback("waiting_start", "Coloca la extremidad en posicion inicial.")

        elif self.state.phase == "going_target":
            if in_target:
                self.state.frames_in_target += 1
                if self.state.frames_in_target >= self.min_frames_in_target:
                    self.state.reached_target = True
                    self.state.phase = "returning"
                    self.state.frames_in_target = 0
                    feedback_key = "return_start"
                    feedback = self._feedback("return_start", "Vuelve lentamente a la posicion inicial.")
                else:
                    feedback_key = "go_to_target"
                    feedback = self._feedback("go_to_target", "Manten la posicion objetivo.")
            else:
                self.state.frames_in_target = 0
                feedback_key = "go_to_target"
                feedback = self._feedback("go_to_target", "Continua hacia el rango objetivo.")

        elif self.state.phase == "returning":
            if in_start and self.state.reached_target:
                self.state.frames_returning_start += 1
                if self.state.frames_returning_start >= self.min_frames_return:
                    rep_duration_actual = timestamp_s - self.state.rep_start_timestamp_s
                    too_fast = rep_duration_actual < self.min_rep_duration

                    rep_completed = True
                    self.state.completed_reps += 1

                    penalty = self._compensation_penalty(angles) if has_compensation else 0.0
                    final_score = max(0.0, self.state.best_score_current_rep - penalty)

                    if too_fast:
                        speed_penalty = min(30.0, (self.min_rep_duration - rep_duration_actual) * 6.0)
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
                        "compensations": compensations,
                        "compensation_penalty": round(penalty, 2),
                    }
                    self.reset_current_rep()
                    if too_fast:
                        feedback_key = "rep_too_fast"
                        feedback = self._feedback(
                            "rep_too_fast",
                            f"Movimiento muy rapido ({rep_duration_actual:.1f}s). Realiza el movimiento con mas control."
                        )
                    else:
                        feedback_key = "rep_valid"
                        feedback = self._feedback("rep_valid", "Repeticion valida.")
                else:
                    feedback_key = "return_start"
                    feedback = self._feedback("return_start", "Manten la posicion inicial.")
            else:
                self.state.frames_returning_start = 0
                feedback_key = "return_start"
                feedback = self._feedback("return_start", "Vuelve a la posicion inicial con control.")

        if has_compensation and not too_fast and not rep_completed:
            if compensations.get("trunk_lean"):
                feedback = self._feedback("trunk_lean", "Manten el tronco mas estable.")
            elif compensations.get("elbow_bend"):
                feedback = self._feedback("elbow_bend", "Evita doblar demasiado el codo.")
            elif compensations.get("shoulder_hike"):
                feedback = self._feedback("shoulder_hike", "Evita elevar el hombro como compensacion.")

        return {
            "ok": True,
            "phase": self.state.phase,
            "rep_completed": rep_completed,
            "completed_reps": self.state.completed_reps,
            "score": round(target_score, 2),
            "best_score_current_rep": round(self.state.best_score_current_rep, 2),
            "last_rep": dict(self.state.last_rep),
            "primary_angle": self.primary_angle,
            "primary_value": primary_value,
            "confirmation_angle": self.confirmation_angle,
            "confirmation_value": confirmation_value,
            "confirmation_active": self.confirmation_angle is not None,
            "calibrated": self.state.calibrated,
            "neutral_value": self.state.neutral_value,
            "in_start": in_start,
            "in_target": in_target,
            "in_target_primary_only": in_target_primary,
            "in_safe": in_safe,
            "too_fast": too_fast,
            "min_rep_duration": self.min_rep_duration,
            "frames_in_start": self.state.frames_in_start,
            "frames_in_target": self.state.frames_in_target,
            "feedback_key": feedback_key,
            "feedback": feedback,
            "compensations": compensations,
        }
