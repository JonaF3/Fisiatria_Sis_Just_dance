"""
rehab_core/angles.py

Cálculo centralizado de ángulos corporales para rehabilitación física.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from keypoint import KEYPOINT_DICT
except Exception:
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }


ANGLE_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "left_elbow": (
        KEYPOINT_DICT["left_shoulder"],
        KEYPOINT_DICT["left_elbow"],
        KEYPOINT_DICT["left_wrist"],
    ),
    "right_elbow": (
        KEYPOINT_DICT["right_shoulder"],
        KEYPOINT_DICT["right_elbow"],
        KEYPOINT_DICT["right_wrist"],
    ),

    "left_shoulder": (
        KEYPOINT_DICT["left_elbow"],
        KEYPOINT_DICT["left_shoulder"],
        KEYPOINT_DICT["left_hip"],
    ),
    "right_shoulder": (
        KEYPOINT_DICT["right_elbow"],
        KEYPOINT_DICT["right_shoulder"],
        KEYPOINT_DICT["right_hip"],
    ),

    "left_hip": (
        KEYPOINT_DICT["left_shoulder"],
        KEYPOINT_DICT["left_hip"],
        KEYPOINT_DICT["left_knee"],
    ),
    "right_hip": (
        KEYPOINT_DICT["right_shoulder"],
        KEYPOINT_DICT["right_hip"],
        KEYPOINT_DICT["right_knee"],
    ),

    "left_knee": (
        KEYPOINT_DICT["left_hip"],
        KEYPOINT_DICT["left_knee"],
        KEYPOINT_DICT["left_ankle"],
    ),
    "right_knee": (
        KEYPOINT_DICT["right_hip"],
        KEYPOINT_DICT["right_knee"],
        KEYPOINT_DICT["right_ankle"],
    ),

    "left_trunk": (
        KEYPOINT_DICT["left_shoulder"],
        KEYPOINT_DICT["left_hip"],
        KEYPOINT_DICT["left_knee"],
    ),
    "right_trunk": (
        KEYPOINT_DICT["right_shoulder"],
        KEYPOINT_DICT["right_hip"],
        KEYPOINT_DICT["right_knee"],
    ),

    "head": (
        KEYPOINT_DICT["left_ear"],
        KEYPOINT_DICT["nose"],
        KEYPOINT_DICT["right_ear"],
    ),
}


ANGLE_ALIASES = {
    "left_arm": "left_elbow",
    "right_arm": "right_elbow",
    "left_leg": "left_knee",
    "right_leg": "right_knee",
    "left_thigh": "left_hip",
    "right_thigh": "right_hip",
}


def canonical_angle_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return ANGLE_ALIASES.get(name, name)


def normalize_keypoints(keypoints) -> Optional[np.ndarray]:
    if keypoints is None:
        return None

    arr = np.array(keypoints, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None

    if arr.shape[0] < 17 or arr.shape[1] < 3:
        return None

    return arr[:17, :3]


def keypoint_confidence_ok(
    keypoints: np.ndarray,
    indices: Tuple[int, int, int],
    min_confidence: float = 0.25,
) -> bool:
    if keypoints is None:
        return False

    for idx in indices:
        if idx < 0 or idx >= len(keypoints):
            return False
        if float(keypoints[idx][2]) < min_confidence:
            return False

    return True


def calculate_angle_from_points(p1, p2, p3) -> float:
    a = np.array(p1[:2], dtype=np.float32)
    b = np.array(p2[:2], dtype=np.float32)
    c = np.array(p3[:2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return float("nan")

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))

    return float(math.degrees(math.acos(cosine)))


def calculate_angle(
    keypoints,
    angle_name: str,
    min_confidence: float = 0.25,
) -> Optional[float]:
    kp = normalize_keypoints(keypoints)

    if kp is None:
        return None

    canonical = canonical_angle_name(angle_name)

    if canonical not in ANGLE_TRIPLETS:
        return None

    triplet = ANGLE_TRIPLETS[canonical]

    if not keypoint_confidence_ok(kp, triplet, min_confidence):
        return None

    angle = calculate_angle_from_points(
        kp[triplet[0]],
        kp[triplet[1]],
        kp[triplet[2]],
    )

    if math.isnan(angle):
        return None

    return round(angle, 2)


def calculate_angles(
    keypoints,
    active_angles: Optional[Iterable[str]] = None,
    min_confidence: float = 0.25,
    include_aliases: bool = True,
) -> Dict[str, float]:
    if active_angles is None:
        names = list(ANGLE_TRIPLETS.keys())
    else:
        names = [canonical_angle_name(name) for name in active_angles]

    result: Dict[str, float] = {}

    for name in names:
        angle = calculate_angle(
            keypoints=keypoints,
            angle_name=name,
            min_confidence=min_confidence,
        )

        if angle is not None:
            result[name] = angle

    if include_aliases and active_angles is not None:
        for old_name in active_angles:
            canonical = canonical_angle_name(old_name)
            if old_name != canonical and canonical in result:
                result[old_name] = result[canonical]

    return result


def filter_angles(
    angles: Dict[str, float],
    active_angles: Optional[Iterable[str]],
) -> Dict[str, float]:
    if not active_angles:
        return dict(angles or {})

    wanted = set()

    for name in active_angles:
        wanted.add(name)
        wanted.add(canonical_angle_name(name))

    return {
        name: value
        for name, value in (angles or {}).items()
        if name in wanted or canonical_angle_name(name) in wanted
    }


def angle_in_range(angle: Optional[float], angle_range) -> bool:
    if angle is None:
        return False

    if not angle_range or len(angle_range) != 2:
        return False

    a = float(angle_range[0])
    b = float(angle_range[1])

    low = min(a, b)
    high = max(a, b)

    return low <= float(angle) <= high


def distance_to_range(angle: Optional[float], angle_range) -> float:
    if angle is None:
        return float("inf")

    if not angle_range or len(angle_range) != 2:
        return float("inf")

    a = float(angle_range[0])
    b = float(angle_range[1])

    low = min(a, b)
    high = max(a, b)

    if low <= angle <= high:
        return 0.0

    if angle < low:
        return low - angle

    return angle - high


def get_required_keypoint_indices(angle_names: Iterable[str]) -> List[int]:
    indices = set()

    for name in angle_names:
        canonical = canonical_angle_name(name)
        triplet = ANGLE_TRIPLETS.get(canonical)

        if triplet:
            indices.update(triplet)

    return sorted(indices)