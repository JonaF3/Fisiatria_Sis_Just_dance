"""
pose_backends/mediapipe_hand21_backend.py

Backend base para MediaPipe Hand Landmarker con 21 landmarks por mano.
No toca el juego principal.

Uso básico:
    from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend

    backend = MediaPipeHand21Backend("model/hand_landmarker.task")
    result = backend.detect_bgr(frame_bgr)
    print(result.detected, result.num_hands)
    print(result.hands[0].handedness_label)
    backend.close()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

NAME_TO_INDEX = {name: idx for idx, name in enumerate(HAND_LANDMARK_NAMES)}


@dataclass
class Hand21Point:
    name: str
    index: int
    x: float
    y: float
    z: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass
class Hand21SingleResult:
    handedness_label: str
    handedness_score: float
    landmarks: Dict[str, Hand21Point]
    world_landmarks: Dict[str, Hand21Point]
    tracking_quality: float
    angles: Dict[str, float]

    def required_visible(self, required_landmarks: list[str]) -> bool:
        return all(name in self.landmarks for name in required_landmarks)

    def as_dict(self) -> dict:
        return {
            "handedness_label": self.handedness_label,
            "handedness_score": self.handedness_score,
            "tracking_quality": self.tracking_quality,
            "landmarks": {k: v.as_dict() for k, v in self.landmarks.items()},
            "world_landmarks": {k: v.as_dict() for k, v in self.world_landmarks.items()},
            "angles": dict(self.angles),
        }


@dataclass
class Hand21Result:
    detected: bool
    hands: List[Hand21SingleResult]
    timestamp_ms: int
    raw_result: Optional[Any] = None

    @property
    def num_hands(self) -> int:
        return len(self.hands)

    def first_hand(self) -> Optional[Hand21SingleResult]:
        return self.hands[0] if self.hands else None

    def as_dict(self, include_raw: bool = False) -> dict:
        data = {
            "detected": self.detected,
            "num_hands": self.num_hands,
            "timestamp_ms": self.timestamp_ms,
            "hands": [h.as_dict() for h in self.hands],
        }
        if include_raw:
            data["raw_result"] = self.raw_result
        return data


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if den < 1e-6:
        return None
    cosang = float(np.dot(v1, v2) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return round(float(math.degrees(math.acos(cosang))), 2)


def angle_2d(points: dict[str, Hand21Point], a: str, b: str, c: str) -> Optional[float]:
    if a not in points or b not in points or c not in points:
        return None
    pa, pb, pc = points[a], points[b], points[c]
    v1 = np.array([pa.x - pb.x, pa.y - pb.y], dtype=np.float32)
    v2 = np.array([pc.x - pb.x, pc.y - pb.y], dtype=np.float32)
    return _angle_between(v1, v2)


def angle_3d(points: dict[str, Hand21Point], a: str, b: str, c: str) -> Optional[float]:
    if a not in points or b not in points or c not in points:
        return None
    pa, pb, pc = points[a], points[b], points[c]
    v1 = np.array([pa.x - pb.x, pa.y - pb.y, pa.z - pb.z], dtype=np.float32)
    v2 = np.array([pc.x - pb.x, pc.y - pb.y, pc.z - pb.z], dtype=np.float32)
    return _angle_between(v1, v2)


HAND_ANGLE_DEFINITIONS_2D = {
    "thumb_mcp_flexion_2d": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
    "thumb_ip_flexion_2d": ("thumb_mcp", "thumb_ip", "thumb_tip"),

    "index_mcp_flexion_2d": ("wrist", "index_mcp", "index_pip"),
    "index_pip_flexion_2d": ("index_mcp", "index_pip", "index_dip"),
    "index_dip_flexion_2d": ("index_pip", "index_dip", "index_tip"),

    "middle_mcp_flexion_2d": ("wrist", "middle_mcp", "middle_pip"),
    "middle_pip_flexion_2d": ("middle_mcp", "middle_pip", "middle_dip"),
    "middle_dip_flexion_2d": ("middle_pip", "middle_dip", "middle_tip"),

    "ring_mcp_flexion_2d": ("wrist", "ring_mcp", "ring_pip"),
    "ring_pip_flexion_2d": ("ring_mcp", "ring_pip", "ring_dip"),
    "ring_dip_flexion_2d": ("ring_pip", "ring_dip", "ring_tip"),

    "pinky_mcp_flexion_2d": ("wrist", "pinky_mcp", "pinky_pip"),
    "pinky_pip_flexion_2d": ("pinky_mcp", "pinky_pip", "pinky_dip"),
    "pinky_dip_flexion_2d": ("pinky_pip", "pinky_dip", "pinky_tip"),
}

HAND_ANGLE_DEFINITIONS_3D = {
    name.replace("_2d", "_3d"): triplet
    for name, triplet in HAND_ANGLE_DEFINITIONS_2D.items()
}


def compute_hand_roll(points: dict[str, Hand21Point]) -> Optional[float]:
    required = ["wrist", "middle_mcp", "index_mcp", "pinky_mcp"]
    if not all(name in points for name in required):
        return None
    
    p_wrist = points["wrist"]
    p_middle_mcp = points["middle_mcp"]
    p_index_mcp = points["index_mcp"]
    p_pinky_mcp = points["pinky_mcp"]
    
    # Vectores en 3D
    u = np.array([p_middle_mcp.x - p_wrist.x, p_middle_mcp.y - p_wrist.y, p_middle_mcp.z - p_wrist.z], dtype=np.float32)
    v = np.array([p_index_mcp.x - p_pinky_mcp.x, p_index_mcp.y - p_pinky_mcp.y, p_index_mcp.z - p_pinky_mcp.z], dtype=np.float32)
    
    norm_u = float(np.linalg.norm(u))
    if norm_u < 1e-6:
        return None
    u_unit = u / norm_u
    
    # Ortogonalizar v respecto a u
    w = v - np.dot(v, u_unit) * u_unit
    norm_w = float(np.linalg.norm(w))
    if norm_w < 1e-6:
        return None
    w_unit = w / norm_w
    
    # Determinar vector de referencia r perpendicular a u
    if abs(u_unit[2]) >= 0.8:
        r_ref = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        r_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
    r = r_ref - np.dot(r_ref, u_unit) * u_unit
    norm_r = float(np.linalg.norm(r))
    if norm_r < 1e-6:
        return None
    r_unit = r / norm_r
    
    # Vector perpendicular a u y r_unit
    r_perp = np.cross(u_unit, r_unit)
    
    # Proyectar w_unit en el plano perpendicular a u
    x = float(np.dot(w_unit, r_unit))
    y = float(np.dot(w_unit, r_perp))
    
    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)
    return round(float(angle_deg % 360.0), 2)


def compute_hand_roll_2d(points: dict[str, Hand21Point]) -> Optional[float]:
    """2D-only hand roll from image landmarks.
    
    Uses the angle of the index_mcp→pinky_mcp line projected in the image plane.
    This line rotates visibly when the wrist pronates/supinates,
    even with a closed fist where z-tracking (world_landmarks) is poor.
    """
    required = ["index_mcp", "pinky_mcp"]
    if not all(name in points for name in required):
        return None
    dx = points["pinky_mcp"].x - points["index_mcp"].x
    dy = points["pinky_mcp"].y - points["index_mcp"].y
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None
    return round(float(math.degrees(math.atan2(dy, dx)) % 360.0), 2)


def compute_hand21_angles(landmarks: dict[str, Hand21Point], world_landmarks: Optional[dict[str, Hand21Point]] = None) -> dict:
    angles = {}
    for name, triplet in HAND_ANGLE_DEFINITIONS_2D.items():
        val = angle_2d(landmarks, *triplet)
        if val is not None:
            angles[name] = val

    source_3d = world_landmarks if world_landmarks else landmarks
    for name, triplet in HAND_ANGLE_DEFINITIONS_3D.items():
        val = angle_3d(source_3d, *triplet)
        if val is not None:
            angles[name] = val

    roll = compute_hand_roll(source_3d)
    if roll is not None:
        angles["hand_rotation_roll"] = roll

    # 2D fallback: purely from image landmarks, robust even with fist and poor z-tracking
    roll_2d = compute_hand_roll_2d(landmarks)
    if roll_2d is not None:
        angles["hand_rotation_2d"] = roll_2d

    return angles


class MediaPipeHand21Backend:
    """Backend MediaPipe Hand Landmarker con landmarks completos de mano."""

    def __init__(
        self,
        model_path: str = "model/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.35,
        min_hand_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        timestamp_step_ms: int = 33,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo MediaPipe Hand: {self.model_path}. "
                "Coloca hand_landmarker.task en la carpeta model/."
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.vision = vision
        self.timestamp_step_ms = int(timestamp_step_ms)
        self._timestamp_ms = 0

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print(f"[INFO] MediaPipeHand21Backend activo: {self.model_path}")

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def _convert_landmarks(self, landmarks) -> Dict[str, Hand21Point]:
        converted = {}
        if not landmarks:
            return converted
        for idx, lm in enumerate(landmarks):
            if idx >= len(HAND_LANDMARK_NAMES):
                continue
            name = HAND_LANDMARK_NAMES[idx]
            converted[name] = Hand21Point(
                name=name,
                index=idx,
                x=float(getattr(lm, "x", 0.0)),
                y=float(getattr(lm, "y", 0.0)),
                z=float(getattr(lm, "z", 0.0)),
            )
        return converted

    def _handedness(self, raw, idx: int) -> tuple[str, float]:
        try:
            if raw.handedness and idx < len(raw.handedness) and raw.handedness[idx]:
                cat = raw.handedness[idx][0]
                return str(cat.category_name), float(cat.score)
        except Exception:
            pass
        return "unknown", 0.0

    def _tracking_quality(self, landmarks: dict[str, Hand21Point], handedness_score: float) -> float:
        # HandLandmarker no expone visibility por punto como Pose; usamos presencia de 21 puntos + handedness.
        landmark_ratio = len(landmarks) / 21.0
        return round(float(max(0.0, min(1.0, (landmark_ratio * 0.7) + (handedness_score * 0.3)))), 4)

    def detect_rgb(self, rgb_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Hand21Result:
        if rgb_frame is None:
            return Hand21Result(False, [], self._timestamp_ms, None)

        image = np.ascontiguousarray(rgb_frame.astype(np.uint8))
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image)

        if timestamp_ms is None:
            self._timestamp_ms += self.timestamp_step_ms
            timestamp_ms = self._timestamp_ms
        else:
            timestamp_ms = int(timestamp_ms)
            self._timestamp_ms = max(self._timestamp_ms, timestamp_ms)

        raw = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        detected = bool(raw and raw.hand_landmarks)
        if not detected:
            return Hand21Result(False, [], timestamp_ms, raw)

        hands = []
        for idx, hand_landmarks in enumerate(raw.hand_landmarks):
            landmarks = self._convert_landmarks(hand_landmarks)
            world_landmarks = {}
            if getattr(raw, "hand_world_landmarks", None):
                if raw.hand_world_landmarks and idx < len(raw.hand_world_landmarks):
                    world_landmarks = self._convert_landmarks(raw.hand_world_landmarks[idx])

            label, score = self._handedness(raw, idx)
            angles = compute_hand21_angles(landmarks, world_landmarks if world_landmarks else None)
            quality = self._tracking_quality(landmarks, score)
            hands.append(Hand21SingleResult(label, score, landmarks, world_landmarks, quality, angles))

        return Hand21Result(True, hands, timestamp_ms, raw)

    def detect_bgr(self, bgr_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Hand21Result:
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(rgb, timestamp_ms=timestamp_ms)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
