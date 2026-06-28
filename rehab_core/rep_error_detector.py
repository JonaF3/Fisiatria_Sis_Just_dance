from __future__ import annotations


class RepErrorDetector:
    """
    Detector de errores de movimiento en tiempo real.

    Maquina de estados independiente del contador de repeticiones:

        NEUTRAL ──(movimiento correcto)──> CORRECT
        NEUTRAL ──(error detectado)──────> INCORRECT  (+1 error)
        CORRECT ──(error detectado)──────> INCORRECT  (+1 error)
        CORRECT ──(vuelve a posicion neutra)─> NEUTRAL
        INCORRECT ──(vuelve a neutro)─────> NEUTRAL
        INCORRECT ──(movimiento correcto)─> CORRECT

    Un error se cuenta UNA SOLA VEZ por evento. Para volver a contar otro error,
    el usuario debe regresar a NEUTRAL primero.
    """

    HYSTERESIS_ERROR = 5
    HYSTERESIS_OK = 3
    HYSTERESIS_NEUTRAL = 5
    WRONG_DIR_DELTA = 8.0

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.state = "neutral"
        self.error_count = 0
        self._prev_angle: float | None = None
        self._cons_error = 0
        self._cons_ok = 0
        self._cons_neutral = 0
        self._total_frames_in_error_state = 0
        self.direction = str(self.cfg.get("direction", "increasing")).lower()

    def _is_error_frame(self, phase: str, primary_value: float | None,
                        in_start: bool, in_target: bool, in_safe: bool,
                        compensations: dict) -> tuple[bool, str]:
        if primary_value is None:
            return False, ""

        if not in_safe:
            return True, "unsafe"

        if self._prev_angle is not None and phase not in ("waiting_start", "calibrating"):
            delta = primary_value - self._prev_angle
            if self.direction == "increasing" and delta < -self.WRONG_DIR_DELTA:
                return True, "wrong_direction"
            if self.direction == "decreasing" and delta > self.WRONG_DIR_DELTA:
                return True, "wrong_direction"

        if isinstance(compensations, dict):
            active = [k for k, v in compensations.items() if v]
            if active:
                return True, f"compensation_{active[0]}"

        return False, ""

    def evaluate(self, phase: str, primary_value: float | None,
                 in_start: bool, in_target: bool, in_safe: bool,
                 compensations: dict, frame_index: int = 0) -> dict:
        is_error, err_type = self._is_error_frame(
            phase, primary_value, in_start, in_target, in_safe, compensations
        )

        self._prev_angle = primary_value

        if is_error:
            self._cons_error += 1
            self._cons_ok = 0
            self._cons_neutral = 0
        else:
            if phase in ("waiting_start", "calibrating") or in_start:
                self._cons_neutral += 1
                self._cons_ok = 0
            else:
                self._cons_ok += 1
                self._cons_neutral = 0
            self._cons_error = 0

        error_just_counted = False
        prev_state = self.state

        if self.state == "neutral":
            if self._cons_error >= self.HYSTERESIS_ERROR:
                self.state = "incorrect"
                self.error_count += 1
                error_just_counted = True
            elif self._cons_ok >= self.HYSTERESIS_OK:
                self.state = "correct"

        elif self.state == "incorrect":
            if self._cons_neutral >= self.HYSTERESIS_NEUTRAL:
                self.state = "neutral"
            elif self._cons_ok >= self.HYSTERESIS_OK:
                self.state = "correct"

        elif self.state == "correct":
            if self._cons_error >= self.HYSTERESIS_ERROR:
                self.state = "incorrect"
                self.error_count += 1
                error_just_counted = True
            elif self._cons_neutral >= self.HYSTERESIS_NEUTRAL:
                self.state = "neutral"

        if self.state == "incorrect":
            self._total_frames_in_error_state += 1

        return {
            "error_state": self.state,
            "error_count": self.error_count,
            "error_just_counted": error_just_counted,
            "error_type": err_type if is_error else None,
            "display_status": "correcto" if self.state == "correct" else (
                "incorrecto" if self.state == "incorrect" else "neutro"
            ),
        }
