from __future__ import annotations

from typing import Optional

import cv2

from .angles import calculate_angles, filter_angles
from .evaluator import RehabExerciseEvaluator
from .metrics_adapter import build_rep_result
from .session_state import RehabSessionState


class RehabControllerBridge:
    """
    Puente entre JustDanceController y la capa clínica de rehabilitación.

    El controller principal sigue manejando:
    - cámara
    - video
    - pygame
    - HUD
    - inferencia
    - audio

    Este bridge maneja:
    - cálculo/evaluación clínica
    - feedback
    - estado de sesión
    - resultados por repetición
    """

    def __init__(
        self,
        exercise_config: dict,
        exercise_id: str = "unknown",
        target_repetitions: int = 0,
    ):
        self.exercise_config = exercise_config or {}

        if "id" not in self.exercise_config:
            self.exercise_config["id"] = exercise_id or "unknown"

        self.evaluator = RehabExerciseEvaluator(self.exercise_config)

        self.state = RehabSessionState(
            exercise_id=self.exercise_config.get("id", exercise_id or "unknown"),
            exercise_name=self.exercise_config.get("name", "Ejercicio"),
            target_repetitions=int(target_repetitions or 0),
        )

        self.latest_result: Optional[dict] = None
        self.latest_feedback: str = ""

    def start(self) -> None:
        self.state.start()

    def evaluate(
        self,
        keypoints=None,
        angles: Optional[dict] = None,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> Optional[dict]:
        result = self.evaluator.evaluate_frame(
            keypoints=keypoints,
            angles=angles,
            update_repetition=True,
        )

        self.latest_result = result
        self.latest_feedback = result.get("feedback", "")

        self.state.update_from_result(
            result=result,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )

        return result

    def calculate_angles_for_exercise(self, keypoints) -> dict:
        active = (
            self.exercise_config.get("active_angles")
            or self.exercise_config.get("active_joints")
            or None
        )

        min_conf = float(
            self.exercise_config.get("min_tracking_confidence", 0.25)
        )

        return calculate_angles(
            keypoints=keypoints,
            active_angles=active,
            min_confidence=min_conf,
            include_aliases=True,
        )

    def filter_angles_for_exercise(self, angles: dict) -> dict:
        active = (
            self.exercise_config.get("active_angles")
            or self.exercise_config.get("active_joints")
            or None
        )

        return filter_angles(angles or {}, active)

    def adjust_score(self, current_score: float) -> float:
        if not isinstance(self.latest_result, dict):
            return float(current_score)

        rehab_score = float(self.latest_result.get("score", 0.0))
        return max(float(current_score), rehab_score)

    def build_rep_result(
        self,
        rep_idx: int,
        rating: str,
        similarity: float,
    ) -> dict:
        return build_rep_result(
            rep_idx=rep_idx,
            rating=rating,
            similarity=similarity,
            rehab_result=self.latest_result,
        )

    def draw_feedback_overlay(self, frame):
        if not self.latest_feedback:
            return frame

        try:
            h, w, _ = frame.shape
            text = str(self.latest_feedback)[:75]

            cv2.rectangle(
                frame,
                (20, h - 72),
                (w - 20, h - 25),
                (10, 20, 35),
                -1,
            )

            cv2.rectangle(
                frame,
                (20, h - 72),
                (w - 20, h - 25),
                (0, 215, 255),
                2,
            )

            cv2.putText(
                frame,
                text,
                (35, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        except Exception:
            pass

        return frame

    def get_summary(self) -> dict:
        evaluator_summary = self.evaluator.get_summary()
        state_summary = self.state.get_summary()

        summary = dict(evaluator_summary)
        summary["state"] = state_summary

        return summary

    def finish(self) -> dict:
        self.state.finish()
        return self.get_summary()