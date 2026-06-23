
"""
test_mediapipe33_camera.py
Prueba directa de MediaPipe Pose Landmarker usando los 33 landmarks completos.

Uso:
    python test_mediapipe33_camera.py

Requisitos:
    pip install mediapipe opencv-python numpy
    model/pose_landmarker_lite.task

Controles:
    Q o ESC : salir
    M       : activar/desactivar espejo
    S       : mostrar/ocultar esqueleto
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("model/pose_landmarker_lite.task")
CAMERA_INDEX = 0

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

CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_thumb"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_thumb"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
]

@dataclass
class LandmarkPoint:
    name: str
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def to_named_landmarks(landmarks) -> dict[str, LandmarkPoint]:
    result = {}
    for idx, lm in enumerate(landmarks):
        if idx >= len(LANDMARK_NAMES):
            continue
        name = LANDMARK_NAMES[idx]
        result[name] = LandmarkPoint(
            name=name,
            x=float(getattr(lm, "x", 0.0)),
            y=float(getattr(lm, "y", 0.0)),
            z=float(getattr(lm, "z", 0.0)),
            visibility=float(getattr(lm, "visibility", 1.0)),
            presence=float(getattr(lm, "presence", 1.0)),
        )
    return result


def valid_point(points: dict[str, LandmarkPoint], name: str, min_conf: float = 0.35):
    p = points.get(name)
    if p is None:
        return None
    if p.visibility < min_conf or p.presence < min_conf:
        return None
    return p


def angle_2d(points: dict[str, LandmarkPoint], a: str, b: str, c: str) -> float | None:
    pa = valid_point(points, a)
    pb = valid_point(points, b)
    pc = valid_point(points, c)
    if pa is None or pb is None or pc is None:
        return None

    v1 = np.array([pa.x - pb.x, pa.y - pb.y], dtype=np.float32)
    v2 = np.array([pc.x - pb.x, pc.y - pb.y], dtype=np.float32)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-6:
        return None
    cosang = float(np.dot(v1, v2) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return round(float(math.degrees(math.acos(cosang))), 1)


def angle_3d(points: dict[str, LandmarkPoint], a: str, b: str, c: str) -> float | None:
    pa = valid_point(points, a)
    pb = valid_point(points, b)
    pc = valid_point(points, c)
    if pa is None or pb is None or pc is None:
        return None

    v1 = np.array([pa.x - pb.x, pa.y - pb.y, pa.z - pb.z], dtype=np.float32)
    v2 = np.array([pc.x - pb.x, pc.y - pb.y, pc.z - pb.z], dtype=np.float32)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-6:
        return None
    cosang = float(np.dot(v1, v2) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return round(float(math.degrees(math.acos(cosang))), 1)


def compute_angles_33(image_points: dict[str, LandmarkPoint], world_points: dict[str, LandmarkPoint] | None = None):
    angles = {
        "right_shoulder_flexion_2d": angle_2d(image_points, "right_elbow", "right_shoulder", "right_hip"),
        "left_shoulder_flexion_2d": angle_2d(image_points, "left_elbow", "left_shoulder", "left_hip"),
        "right_elbow_flexion_2d": angle_2d(image_points, "right_shoulder", "right_elbow", "right_wrist"),
        "left_elbow_flexion_2d": angle_2d(image_points, "left_shoulder", "left_elbow", "left_wrist"),
        "right_knee_flexion_2d": angle_2d(image_points, "right_hip", "right_knee", "right_ankle"),
        "left_knee_flexion_2d": angle_2d(image_points, "left_hip", "left_knee", "left_ankle"),
    }

    if world_points:
        angles.update({
            "right_shoulder_flexion_3d": angle_3d(world_points, "right_elbow", "right_shoulder", "right_hip"),
            "left_shoulder_flexion_3d": angle_3d(world_points, "left_elbow", "left_shoulder", "left_hip"),
            "right_elbow_flexion_3d": angle_3d(world_points, "right_shoulder", "right_elbow", "right_wrist"),
            "left_elbow_flexion_3d": angle_3d(world_points, "left_shoulder", "left_elbow", "left_wrist"),
        })

    return {k: v for k, v in angles.items() if v is not None}


def trunk_lean_2d(points: dict[str, LandmarkPoint]) -> float | None:
    left_shoulder = valid_point(points, "left_shoulder")
    right_shoulder = valid_point(points, "right_shoulder")
    left_hip = valid_point(points, "left_hip")
    right_hip = valid_point(points, "right_hip")
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return None

    shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
    hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
    v = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])
    den = np.linalg.norm(v) * np.linalg.norm(vertical)
    if den < 1e-6:
        return None
    cosang = float(np.dot(v, vertical) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return round(float(math.degrees(math.acos(cosang))), 1)


def point_to_pixel(p: LandmarkPoint, w: int, h: int):
    return int(p.x * w), int(p.y * h)


def draw_skeleton(frame, points: dict[str, LandmarkPoint]):
    h, w = frame.shape[:2]
    for a, b in CONNECTIONS:
        pa = valid_point(points, a, min_conf=0.25)
        pb = valid_point(points, b, min_conf=0.25)
        if pa and pb:
            cv2.line(frame, point_to_pixel(pa, w, h), point_to_pixel(pb, w, h), (40, 200, 120), 2, cv2.LINE_AA)
    for name, p in points.items():
        if p.visibility >= 0.25 and p.presence >= 0.25:
            cv2.circle(frame, point_to_pixel(p, w, h), 3, (0, 180, 180), -1, cv2.LINE_AA)


def draw_panel(frame, fps, image_points, angles, lean, detected, mirror):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 142), (12, 20, 40), -1)
    cv2.line(frame, (0, 142), (w, 142), (0, 180, 180), 2)
    cv2.putText(frame, "MediaPipe 33 landmarks - prueba directa", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 245, 250), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Mirror: {'ON' if mirror else 'OFF'} | Detectado: {'SI' if detected else 'NO'} | Landmarks: {len(image_points)} / 33", (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 180), 1, cv2.LINE_AA)

    selected = [
        "right_shoulder_flexion_2d", "right_shoulder_flexion_3d",
        "right_elbow_flexion_2d", "right_elbow_flexion_3d",
        "left_shoulder_flexion_2d", "left_shoulder_flexion_3d",
    ]
    text_items = []
    for key in selected:
        if key in angles:
            short = key.replace("_flexion", "").replace("_shoulder", "_sh").replace("right_", "R_").replace("left_", "L_")
            text_items.append(f"{short}: {angles[key]:.1f}")
    if lean is not None:
        text_items.append(f"trunk_lean_2d: {lean:.1f}")

    line1 = " | ".join(text_items[:3]) if text_items else "Sin ángulos suficientes"
    line2 = " | ".join(text_items[3:]) if len(text_items) > 3 else ""
    cv2.putText(frame, line1[:150], (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150, 170, 185), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2[:150], (16, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150, 170, 185), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q/ESC salir | M mirror | S skeleton", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 160, 40), 1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] No existe el modelo: {MODEL_PATH}")
        return

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mirror = True
    show_skeleton = True
    last_t = time.time()
    fps_smooth = 0.0
    frame_idx = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000 / 30))
            frame_idx += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            detected = bool(result.pose_landmarks)
            image_points = {}
            world_points = {}
            angles = {}
            lean = None

            if detected:
                image_points = to_named_landmarks(result.pose_landmarks[0])
                if result.pose_world_landmarks:
                    world_points = to_named_landmarks(result.pose_world_landmarks[0])
                angles = compute_angles_33(image_points, world_points if world_points else None)
                lean = trunk_lean_2d(image_points)
                if show_skeleton:
                    draw_skeleton(frame, image_points)

            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst_fps if fps_smooth <= 0 else fps_smooth * 0.9 + inst_fps * 0.1

            draw_panel(frame, fps_smooth, image_points, angles, lean, detected, mirror)
            cv2.imshow("MediaPipe 33 Landmarks - Prueba", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            if key in (ord("s"), ord("S")):
                show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
