"""
rehab_core/feedback.py

Mensajes de retroalimentación para rehabilitación física.
"""

from __future__ import annotations

from typing import List, Optional


DEFAULT_MESSAGES = {
    "excellent": "Excelente control del movimiento.",
    "good": "Movimiento correcto.",
    "not_enough_range": "Intenta alcanzar un poco más de rango, sin forzar.",
    "too_much_range": "Evita exceder el rango seguro configurado.",
    "lost_tracking": "Ajusta la cámara para que se vea mejor la zona del cuerpo.",
    "missing_angle": "No se pudo medir correctamente la articulación principal.",
    "unsafe": "Movimiento fuera del rango seguro configurado.",
    "waiting_start": "Colócate en la posición inicial.",
    "go_to_target": "Realiza el movimiento hacia el rango objetivo.",
    "return_start": "Regresa lentamente a la posición inicial.",
    "rep_valid": "Repetición válida.",
    "rep_invalid": "Repetición no válida. Intenta nuevamente con control.",
}


def _feedback_from_config(config: Optional[dict]) -> dict:
    if not isinstance(config, dict):
        return {}

    feedback = config.get("feedback", {})
    return feedback if isinstance(feedback, dict) else {}


def get_message(code: str, config: Optional[dict] = None, fallback: str = "") -> str:
    custom = _feedback_from_config(config)

    if code in custom:
        return str(custom[code])

    if code in DEFAULT_MESSAGES:
        return DEFAULT_MESSAGES[code]

    return fallback or "Continúa con el ejercicio."


def build_feedback(result: dict, config: Optional[dict] = None) -> str:
    if not isinstance(result, dict):
        return get_message("missing_angle", config)

    if not result.get("tracking_ok", True):
        return get_message("lost_tracking", config)

    if result.get("missing_primary_angle"):
        return get_message("missing_angle", config)

    if result.get("unsafe"):
        return get_message("unsafe", config)

    if result.get("rep_completed"):
        last_rep = result.get("last_rep") or {}
        if last_rep.get("valid", False):
            return get_message("rep_valid", config)
        return get_message("rep_invalid", config)

    phase = result.get("phase")

    if phase == "waiting_start":
        return get_message("waiting_start", config)

    if phase == "moving_to_target":
        return get_message("go_to_target", config)

    if phase == "returning":
        return get_message("return_start", config)

    score = float(result.get("score", 0.0))

    if score >= 90:
        return get_message("excellent", config)

    if score >= 70:
        return get_message("good", config)

    return get_message("not_enough_range", config)


def build_feedback_list(result: dict, config: Optional[dict] = None) -> List[str]:
    messages = [build_feedback(result, config)]

    if not isinstance(result, dict):
        return messages

    if result.get("unsafe"):
        messages.append(get_message("too_much_range", config))

    if not result.get("tracking_ok", True):
        messages.append(get_message("lost_tracking", config))

    if result.get("missing_primary_angle"):
        messages.append(get_message("missing_angle", config))

    return list(dict.fromkeys(messages))