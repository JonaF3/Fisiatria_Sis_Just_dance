from __future__ import annotations


# Mapa: evaluation_mode -> (start_phases, moving_phases)
# Cada evaluador tiene su propio ciclo de fases.
_EVAL_PHASES = {
    "trajectory": ({"waiting_start", "calibrating"}, {"going_target", "returning"}),
    "static_wrist_alignment": ({"waiting_release"}, {"waiting_target"}),
    "open_close": ({"waiting_open"}, {"going_closed", "returning_open"}),
    "finger_adduction": ({"waiting_spread"}, {"going_together", "returning_spread"}),
}


class RepErrorDetector:
    """
    Detecta errores de movimiento observando las transiciones de fase del FSM.

    Lógica:
    - Cuando el usuario SALE de la fase inicial y ENTRA en una fase de movimiento,
      se marca un intento activo.
    - Si el intento termina VOLVIENDO a la fase inicial sin rep_completed,
      se cuenta 1 error (movimiento abortado / dirección incorrecta).
    - NO cuenta errores por score bajo en rep_completed; eso lo maneja el pipeline.

    Las fases se seleccionan según evaluation_mode del config:
      trajectory:       waiting_start -> going_target -> returning
      static_wrist:     waiting_release -> waiting_target
      open_close:       waiting_open -> going_closed -> returning_open
      finger_adduction: waiting_spread -> going_together -> returning_spread
    """

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.error_count = 0
        self._attempt_active = False
        self._prev_phase: str | None = None
        eval_mode = str(self.cfg.get("evaluation_mode", "trajectory")).lower()
        if eval_mode not in _EVAL_PHASES:
            eval_mode = "trajectory"
        self._start_phases, self._moving_phases = _EVAL_PHASES[eval_mode]

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

        error_just_counted = False
        attempt_started = False

        # 1. Detectar inicio de intento de movimiento
        if not self._attempt_active and phase in self._moving_phases:
            self._attempt_active = True
            attempt_started = True

        # 2. Repetición completada — solo resetear intento, NO contar error
        if rep_completed:
            self._attempt_active = False

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
            "error_type": "aborted" if error_just_counted else None,
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
