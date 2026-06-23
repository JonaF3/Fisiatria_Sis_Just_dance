from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .angles import (
    angle_in_range,
    calculate_angles,
    canonical_angle_name,
    distance_to_range,
    filter_angles,
)
from .feedback import build_feedback, build_feedback_list


@dataclass
class RepetitionState:
    phase: str = "waiting_start"
    completed_reps: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    entered_start: bool = False
    entered_target: bool = False
    best_score_current_rep: float = 0.0
    rep_history: List[dict] = field(default_factory=list)


class RehabExerciseEvaluator:
    def __init__(self, exercise_config: dict):
        self.config = exercise_config or {}
        self.state = RepetitionState()
        self.min_tracking_confidence = float(
            self.config.get("min_tracking_confidence", 0.25)
        )
        self.valid_rep_score = float(
            self.config.get("valid_rep_score", 70.0)
        )

    def get_active_angles(self) -> List[str]:
        active = (
            self.config.get("active_angles")
            or self.config.get("active_joints")
            or self.get_primary_angles()
        )
        return [canonical_angle_name(x) for x in active]

    def get_primary_angles(self) -> List[str]:
        if "primary_angles" in self.config:
            return [
                canonical_angle_name(x)
                for x in self.config.get("primary_angles", [])
            ]

        if "primary_angle" in self.config:
            return [canonical_angle_name(self.config["primary_angle"])]

        active = self.config.get("active_joints", [])
        if active:
            return [canonical_angle_name(active[0])]

        return []

    def _range_for_angle(self, range_key: str, angle_name: str):
        canonical = canonical_angle_name(angle_name)
        ranges = self.config.get(range_key)

        if isinstance(ranges, dict):
            if canonical in ranges:
                return ranges[canonical]

            for key, value in ranges.items():
                if canonical_angle_name(key) == canonical:
                    return value

        fallback_map = {
            "target_ranges": "target_angle_range",
            "safe_ranges": "safe_angle_range",
            "start_ranges": "start_angle_range",
            "return_ranges": "return_angle_range",
        }

        fallback_key = fallback_map.get(range_key)
        if fallback_key and fallback_key in self.config:
            return self.config.get(fallback_key)

        return None

    def evaluate_frame(
        self,
        keypoints=None,
        angles: Optional[Dict[str, float]] = None,
        update_repetition: bool = True,
    ) -> dict:
        active_angles = self.get_active_angles()
        primary_angles = self.get_primary_angles()

        if angles is None:
            angles = calculate_angles(
                keypoints=keypoints,
                active_angles=active_angles,
                min_confidence=self.min_tracking_confidence,
            )
        else:
            angles = filter_angles(angles, active_angles)

        tracking_ok = bool(angles)

        result = {
            "exercise_id": self.config.get("id") or self.config.get("name", "unknown"),
            "exercise_name": self.config.get("name", "Ejercicio"),
            "active_angles": active_angles,
            "primary_angles": primary_angles,
            "angles": angles,
            "tracking_ok": tracking_ok,
            "tracking_quality": 100.0 if tracking_ok else 0.0,
            "score": 0.0,
            "unsafe": False,
            "not_enough_range": False,
            "missing_primary_angle": False,
            "phase": self.state.phase,
            "rep_completed": False,
            "completed_reps": self.state.completed_reps,
            "valid_reps": self.state.valid_reps,
            "invalid_reps": self.state.invalid_reps,
            "last_rep": None,
        }

        if not tracking_ok:
            result["feedback"] = build_feedback(result, self.config)
            result["feedback_list"] = build_feedback_list(result, self.config)
            return result

        primary_scores = []
        unsafe_flags = []
        not_enough_flags = []

        for angle_name in primary_angles:
            canonical = canonical_angle_name(angle_name)
            value = angles.get(canonical)

            if value is None:
                result["missing_primary_angle"] = True
                continue

            target_range = self._range_for_angle("target_ranges", canonical)
            safe_range = self._range_for_angle("safe_ranges", canonical)

            if safe_range is not None and not angle_in_range(value, safe_range):
                unsafe_flags.append(True)

            if target_range is not None:
                score = self._score_angle_against_range(value, target_range)
                primary_scores.append(score)

                if not angle_in_range(value, target_range):
                    not_enough_flags.append(True)
            else:
                primary_scores.append(50.0)

        if primary_scores:
            result["score"] = round(float(np.mean(primary_scores)), 1)
        else:
            result["score"] = 0.0
            result["missing_primary_angle"] = True

        result["unsafe"] = any(unsafe_flags)
        result["not_enough_range"] = any(not_enough_flags)

        if update_repetition:
            rep_info = self._update_repetition_state(angles, result)
            result.update(rep_info)

        result["feedback"] = build_feedback(result, self.config)
        result["feedback_list"] = build_feedback_list(result, self.config)
        return result

    def _score_angle_against_range(self, angle: float, target_range) -> float:
        if angle_in_range(angle, target_range):
            return 100.0

        dist = distance_to_range(angle, target_range)
        score = max(0.0, 100.0 - dist * 2.0)
        return round(score, 1)

    def _all_primary_in_range(self, angles: Dict[str, float], range_key: str) -> bool:
        primary = self.get_primary_angles()
        if not primary:
            return False

        ok_count = 0
        for angle_name in primary:
            canonical = canonical_angle_name(angle_name)
            value = angles.get(canonical)
            configured_range = self._range_for_angle(range_key, canonical)

            if value is None or configured_range is None:
                continue

            if angle_in_range(value, configured_range):
                ok_count += 1

        return ok_count > 0 and ok_count == len(primary)

    def _any_primary_in_target(self, angles: Dict[str, float]) -> bool:
        for angle_name in self.get_primary_angles():
            canonical = canonical_angle_name(angle_name)
            value = angles.get(canonical)
            target_range = self._range_for_angle("target_ranges", canonical)

            if value is not None and target_range is not None:
                if angle_in_range(value, target_range):
                    return True

        return False

    def _update_repetition_state(self, angles: Dict[str, float], result: dict) -> dict:
        info = {
            "phase": self.state.phase,
            "rep_completed": False,
            "completed_reps": self.state.completed_reps,
            "valid_reps": self.state.valid_reps,
            "invalid_reps": self.state.invalid_reps,
            "last_rep": None,
        }

        score = float(result.get("score", 0.0))
        self.state.best_score_current_rep = max(
            self.state.best_score_current_rep,
            score,
        )

        has_start_range = any(
            self._range_for_angle("start_ranges", angle_name) is not None
            for angle_name in self.get_primary_angles()
        )

        in_start = (
            self._all_primary_in_range(angles, "start_ranges")
            if has_start_range
            else False
        )
        in_target = self._any_primary_in_target(angles)

        if self.state.phase == "waiting_start":
            if not has_start_range:
                self.state.phase = "moving_to_target"
            elif in_start:
                self.state.entered_start = True
                self.state.phase = "moving_to_target"

        elif self.state.phase == "moving_to_target":
            if in_target:
                self.state.entered_target = True
                self.state.phase = "returning"

        elif self.state.phase == "returning":
            if has_start_range:
                returned = in_start
            else:
                returned = not in_target and self.state.entered_target

            if returned:
                valid = (
                    self.state.entered_target
                    and self.state.best_score_current_rep >= self.valid_rep_score
                    and not result.get("unsafe", False)
                )

                self.state.completed_reps += 1

                if valid:
                    self.state.valid_reps += 1
                else:
                    self.state.invalid_reps += 1

                rep_record = {
                    "rep_idx": self.state.completed_reps,
                    "valid": bool(valid),
                    "best_score": round(self.state.best_score_current_rep, 1),
                    "unsafe": bool(result.get("unsafe", False)),
                }

                self.state.rep_history.append(rep_record)

                info["rep_completed"] = True
                info["last_rep"] = rep_record

                self.state.phase = "waiting_start"
                self.state.entered_start = False
                self.state.entered_target = False
                self.state.best_score_current_rep = 0.0

        info["phase"] = self.state.phase
        info["completed_reps"] = self.state.completed_reps
        info["valid_reps"] = self.state.valid_reps
        info["invalid_reps"] = self.state.invalid_reps
        return info

    def reset(self) -> None:
        self.state = RepetitionState()

    def get_summary(self) -> dict:
        total = self.state.completed_reps
        valid = self.state.valid_reps
        valid_percentage = round((valid / total) * 100.0, 1) if total else 0.0

        return {
            "exercise_id": self.config.get("id") or self.config.get("name", "unknown"),
            "exercise_name": self.config.get("name", "Ejercicio"),
            "completed_reps": total,
            "valid_reps": valid,
            "invalid_reps": self.state.invalid_reps,
            "valid_percentage": valid_percentage,
            "rep_history": list(self.state.rep_history),
        }
