from __future__ import annotations


class RepErrorDetector:
    """
    Detecta errores de movimiento observando las transiciones de fase del FSM.

    Lógica:
    - Cuando el usuario SALE de la fase inicial (waiting_start / waiting_release)
      y ENTRA en una fase de movimiento (going_target / waiting_target),
      se marca un intento activo.
    - Si el intento termina VOLVIENDO a la fase inicial sin un rep_completed,
      se cuenta 1 error (movimiento abortado / dirección incorrecta).
    - Si el rep se completa pero con score < valid_rep_score,
      también se cuenta 1 error (movimiento incompleto / fuera de rango).

    Estados del FSM según tracking_type:
      pose33:   waiting_start -> going_target -> returning -> waiting_start
      hand21:   waiting_release -> waiting_target -> cooldown -> waiting_release
    """

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.error_count = 0
        self._attempt_active = False
        self._prev_phase: str | None = None
        self._valid_rep_score = float(self.cfg.get("valid_rep_score", 70.0))
        tracking = str(self.cfg.get("tracking_type", "pose33")).lower()
        if tracking == "hand21":
            self._start_phases = {"waiting_release"}
            self._moving_phases = {"waiting_target", "cooldown"}
        else:
            self._start_phases = {"waiting_start", "calibrating"}
            self._moving_phases = {"going_target", "returning"}

    def evaluate(self, eval_result: dict) -> dict:
        """
        Analiza el resultado del evaluador y actualiza el contador de errores.

        Args:
            eval_result: diccionario devuelto por evaluate() del evaluador.
                Debe contener: 'phase', 'rep_completed', 'score'.

        Returns:
            dict con: error_state, error_count, error_just_counted, error_type,
                      display_status, attempt_started
        """
        phase = str(eval_result.get("phase", ""))
        rep_completed = eval_result.get("rep_completed", False)
        score = float(eval_result.get("score", 0.0))

        error_just_counted = False
        attempt_started = False
        prev_state = self._get_display_state()

        # 1. Detectar inicio de intento de movimiento
        if not self._attempt_active and phase in self._moving_phases:
            self._attempt_active = True
            attempt_started = True

        # 2. Repetición completada — usar best_score del last_rep, no score instantáneo
        if rep_completed:
            self._attempt_active = False
            last_rep = eval_result.get("last_rep") or {}
            rep_score = float(last_rep.get("best_score", score))
            if rep_score < self._valid_rep_score:
                self.error_count += 1
                error_just_counted = True

        # 3. Intento abortado: estaba en movimiento y volvió a inicio sin completar
        if self._attempt_active and phase in self._start_phases and not rep_completed:
            if self._prev_phase and self._prev_phase not in self._start_phases:
                self.error_count += 1
                error_just_counted = True
                self._attempt_active = False

        self._prev_phase = phase

        if phase in self._moving_phases:
            display_status = "correcto"
        elif phase in self._start_phases or not phase:
            display_status = "neutro"
        else:
            display_status = "neutro"

        return {
            "error_state": "incorrect" if error_just_counted else "correct" if phase in self._moving_phases else "neutral",
            "error_count": self.error_count,
            "error_just_counted": error_just_counted,
            "error_type": "aborted" if (error_just_counted and not rep_completed) else "low_score" if (error_just_counted) else None,
            "display_status": display_status,
            "attempt_started": attempt_started,
        }

    def _get_display_state(self) -> str:
        if self.error_count > 0:
            return "has_errors"
        return "clean"

    def reset(self) -> None:
        self.error_count = 0
        self._attempt_active = False
        self._prev_phase = None
