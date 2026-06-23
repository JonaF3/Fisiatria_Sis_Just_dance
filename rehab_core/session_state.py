"""
rehab_core/session_state.py

Estado de sesión clínica separado del controller principal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RehabFrameState:
    frame_index: int = 0
    timestamp_s: float = 0.0
    angles: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    feedback: str = ""
    phase: str = ""
    tracking_ok: bool = False


@dataclass
class RehabSessionState:
    exercise_id: str = "unknown"
    exercise_name: str = "Ejercicio"
    target_repetitions: int = 0

    current_rep: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    completed_reps: int = 0

    latest_frame: Optional[RehabFrameState] = None
    latest_result: Optional[dict] = None
    rep_results: List[dict] = field(default_factory=list)

    started: bool = False
    finished: bool = False
    _start_wall_time: Optional[float] = field(default=None, repr=False)
    _end_wall_time: Optional[float] = field(default=None, repr=False)

    def start(self) -> None:
        self.started = True
        self.finished = False
        self._start_wall_time = time.time()
        self._end_wall_time = None

    def finish(self) -> None:
        self.finished = True
        self._end_wall_time = time.time()

    def update_from_result(
        self,
        result: dict,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> None:
        if not isinstance(result, dict):
            return

        self.latest_result = result

        self.latest_frame = RehabFrameState(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            angles=result.get("angles", {}) or {},
            score=float(result.get("score", 0.0)),
            feedback=str(result.get("feedback", "")),
            phase=str(result.get("phase", "")),
            tracking_ok=bool(result.get("tracking_ok", False)),
        )

        self.valid_reps = int(result.get("valid_reps", self.valid_reps) or 0)
        self.invalid_reps = int(result.get("invalid_reps", self.invalid_reps) or 0)
        self.completed_reps = int(result.get("completed_reps", self.completed_reps) or 0)
        self.current_rep = self.completed_reps

        if result.get("rep_completed") and result.get("last_rep"):
            self.rep_results.append(result["last_rep"])

    def get_latest_feedback(self) -> str:
        if self.latest_frame:
            return self.latest_frame.feedback
        return ""

    def get_latest_score(self) -> float:
        if self.latest_frame:
            return self.latest_frame.score
        return 0.0

    def get_summary(self) -> dict:
        total = self.completed_reps
        valid = self.valid_reps
        valid_percentage = round((valid / total) * 100.0, 1) if total else 0.0

        # Duración total de la sesión
        if self._start_wall_time is not None:
            end = self._end_wall_time if self._end_wall_time is not None else time.time()
            session_duration_s = round(end - self._start_wall_time, 1)
        else:
            session_duration_s = None

        return {
            "exercise_id": self.exercise_id,
            "exercise_name": self.exercise_name,
            "target_repetitions": self.target_repetitions,
            "completed_reps": self.completed_reps,
            "valid_reps": self.valid_reps,
            "invalid_reps": self.invalid_reps,
            "valid_percentage": valid_percentage,
            "session_duration_s": session_duration_s,
            "rep_results": list(self.rep_results),
            "finished": self.finished,
        }