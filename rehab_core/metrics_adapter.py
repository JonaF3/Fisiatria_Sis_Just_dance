"""
rehab_core/metrics_adapter.py

Convierte resultados clínicos en estructuras guardables.
"""

from __future__ import annotations

from typing import Optional


def flatten_rehab_result(result: Optional[dict]) -> dict:
    if not isinstance(result, dict):
        return {}

    angles = result.get("angles", {}) or {}

    flat = {
        "rehab_exercise_id": result.get("exercise_id", ""),
        "rehab_exercise_name": result.get("exercise_name", ""),
        "rehab_score": result.get("score", 0.0),
        "rehab_feedback": result.get("feedback", ""),
        "rehab_phase": result.get("phase", ""),
        "rehab_tracking_ok": result.get("tracking_ok", False),
        "rehab_tracking_quality": result.get("tracking_quality", 0.0),
        "rehab_unsafe": result.get("unsafe", False),
        "rehab_rep_completed": result.get("rep_completed", False),
        "rehab_completed_reps": result.get("completed_reps", 0),
        "rehab_valid_reps": result.get("valid_reps", 0),
        "rehab_invalid_reps": result.get("invalid_reps", 0),
    }

    for name, value in angles.items():
        flat[f"angle_{name}"] = value

    return flat


def build_rehab_session_summary(
    evaluator_summary: Optional[dict],
    state_summary: Optional[dict] = None,
) -> dict:
    summary = {}

    if isinstance(evaluator_summary, dict):
        summary.update(evaluator_summary)

    if isinstance(state_summary, dict):
        summary.update({
            "target_repetitions": state_summary.get("target_repetitions"),
            "finished": state_summary.get("finished"),
            "state_rep_results": state_summary.get("rep_results", []),
        })

    return summary


def build_rep_result(
    rep_idx: int,
    rating: str,
    similarity: float,
    rehab_result: Optional[dict],
) -> dict:
    result = {
        "rep_idx": rep_idx,
        "status": rating,
        "similarity": float(similarity),
    }

    if isinstance(rehab_result, dict):
        result.update({
            "rehab_score": float(rehab_result.get("score", similarity)),
            "rehab_feedback": rehab_result.get("feedback", ""),
            "rehab_phase": rehab_result.get("phase", ""),
            "valid_reps": rehab_result.get("valid_reps"),
            "invalid_reps": rehab_result.get("invalid_reps"),
            "angles": rehab_result.get("angles", {}),
            "unsafe": rehab_result.get("unsafe", False),
        })

    return result