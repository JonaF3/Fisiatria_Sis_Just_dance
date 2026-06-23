"""
pose_backends/mediapipe33_backend.py

Backend base para usar MediaPipe Pose Landmarker con los 33 landmarks completos.
Este archivo NO convierte landmarks a formato MoveNet de 17 puntos.

Uso básico:
    from pose_backends.mediapipe33_backend import MediaPipe33Backend

    backend = MediaPipe33Backend("model/pose_landmarker_lite.task")
    result = backend.detect_bgr(frame_bgr)
    print(result.tracking_quality)
    print(result.image_landmarks["right_shoulder"])
    backend.close()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NAME_TO_INDEX = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}


@dataclass
class Landmark33:
    """Landmark individual de MediaPipe Pose 33."""
    name: str
    index: int
    x: float
    y: float
    z: float
    visibility: float
    presence: float

    @property
    def confidence(self) -> float:
        return max(0.0, min(1.0, float(self.visibility) * float(self.presence)))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
            "presence": self.presence,
            "confidence": self.confidence,
        }


@dataclass
class Pose33Result:
    """Resultado completo del backend MediaPipe 33."""
    detected: bool
    image_landmarks: Dict[str, Landmark33]
    world_landmarks: Dict[str, Landmark33]
    tracking_quality: float
    timestamp_ms: int
    raw_result: Optional[Any] = None

    def required_visible(self, required_landmarks: list[str], min_confidence: float = 0.35) -> bool:
        """Devuelve True si todos los landmarks requeridos tienen confianza suficiente."""
        for name in required_landmarks:
            point = self.image_landmarks.get(name)
            if point is None or point.confidence < min_confidence:
                return False
        return True

    def as_dict(self, include_raw: bool = False) -> dict:
        data = {
            "detected": self.detected,
            "tracking_quality": self.tracking_quality,
            "timestamp_ms": self.timestamp_ms,
            "image_landmarks": {k: v.as_dict() for k, v in self.image_landmarks.items()},
            "world_landmarks": {k: v.as_dict() for k, v in self.world_landmarks.items()},
        }
        if include_raw:
            data["raw_result"] = self.raw_result
        return data


class MediaPipe33Backend:
    """
    Backend MediaPipe Pose Landmarker usando 33 landmarks completos.

    Modos:
      - Por simplicidad y estabilidad inicial, se usa RunningMode.VIDEO.
      - Cada llamada incrementa timestamp_ms automáticamente.
    """

    def __init__(
        self,
        model_path: str = "model/pose_landmarker_lite.task",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.35,
        min_pose_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        timestamp_step_ms: int = 33,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo MediaPipe: {self.model_path}. "
                "Coloca pose_landmarker_lite.task en la carpeta model/."
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.vision = vision
        self.timestamp_step_ms = int(timestamp_step_ms)
        self._timestamp_ms = 0

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=int(num_poses),
            min_pose_detection_confidence=float(min_pose_detection_confidence),
            min_pose_presence_confidence=float(min_pose_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print(f"[INFO] MediaPipe33Backend activo: {self.model_path}")

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def _convert_landmarks(self, landmarks) -> Dict[str, Landmark33]:
        converted: Dict[str, Landmark33] = {}
        if not landmarks:
            return converted

        for idx, lm in enumerate(landmarks):
            if idx >= len(LANDMARK_NAMES):
                continue
            name = LANDMARK_NAMES[idx]
            converted[name] = Landmark33(
                name=name,
                index=idx,
                x=float(getattr(lm, "x", 0.0)),
                y=float(getattr(lm, "y", 0.0)),
                z=float(getattr(lm, "z", 0.0)),
                visibility=float(getattr(lm, "visibility", 1.0)),
                presence=float(getattr(lm, "presence", 1.0)),
            )
        return converted

    def _tracking_quality(self, image_landmarks: Dict[str, Landmark33]) -> float:
        """Promedio de confianza de landmarks corporales principales."""
        important = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]
        values = [image_landmarks[name].confidence for name in important if name in image_landmarks]
        if not values:
            return 0.0
        return round(float(np.mean(values)), 4)

    def detect_rgb(self, rgb_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose33Result:
        """Detecta pose desde un frame RGB."""
        if rgb_frame is None:
            return Pose33Result(False, {}, {}, 0.0, self._timestamp_ms, None)

        image = np.ascontiguousarray(rgb_frame.astype(np.uint8))
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image)

        if timestamp_ms is None:
            self._timestamp_ms += self.timestamp_step_ms
            timestamp_ms = self._timestamp_ms
        else:
            timestamp_ms = int(timestamp_ms)
            self._timestamp_ms = max(self._timestamp_ms, timestamp_ms)

        raw = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        detected = bool(raw and raw.pose_landmarks)

        if not detected:
            return Pose33Result(False, {}, {}, 0.0, timestamp_ms, raw)

        image_landmarks = self._convert_landmarks(raw.pose_landmarks[0])
        world_landmarks = {}
        if getattr(raw, "pose_world_landmarks", None):
            if raw.pose_world_landmarks:
                world_landmarks = self._convert_landmarks(raw.pose_world_landmarks[0])

        quality = self._tracking_quality(image_landmarks)
        return Pose33Result(True, image_landmarks, world_landmarks, quality, timestamp_ms, raw)

    def detect_bgr(self, bgr_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose33Result:
        """Detecta pose desde un frame BGR de OpenCV."""
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(rgb, timestamp_ms=timestamp_ms)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
