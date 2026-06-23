
"""
test_hand21_camera.py

Prueba inicial de MediaPipe Hand Landmarker con 21 puntos por mano.
No toca el juego principal.

Uso:
    python test_hand21_camera.py

Requisitos:
    model/hand_landmarker.task

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
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

MODEL_PATH = Path("model/hand_landmarker.task")
CAMERA_INDEX = 0

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

CONNECTIONS = [
    ("wrist", "thumb_cmc"), ("thumb_cmc", "thumb_mcp"), ("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"), ("index_mcp", "index_pip"), ("index_pip", "index_dip"), ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"), ("middle_mcp", "middle_pip"), ("middle_pip", "middle_dip"), ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"), ("ring_mcp", "ring_pip"), ("ring_pip", "ring_dip"), ("ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp"), ("pinky_mcp", "pinky_pip"), ("pinky_pip", "pinky_dip"), ("pinky_dip", "pinky_tip"),
    ("index_mcp", "middle_mcp"), ("middle_mcp", "ring_mcp"), ("ring_mcp", "pinky_mcp"),
]

@dataclass
class HandPoint:
    name: str
    index: int
    x: float
    y: float
    z: float


def to_named_hand_landmarks(landmarks) -> dict[str, HandPoint]:
    out = {}
    for idx, lm in enumerate(landmarks):
        if idx >= len(HAND_LANDMARK_NAMES):
            continue
        name = HAND_LANDMARK_NAMES[idx]
        out[name] = HandPoint(name=name, index=idx, x=float(lm.x), y=float(lm.y), z=float(lm.z))
    return out


def angle_2d(points: dict[str, HandPoint], a: str, b: str, c: str):
    if a not in points or b not in points or c not in points:
        return None
    pa, pb, pc = points[a], points[b], points[c]
    v1 = np.array([pa.x - pb.x, pa.y - pb.y], dtype=np.float32)
    v2 = np.array([pc.x - pb.x, pc.y - pb.y], dtype=np.float32)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-6:
        return None
    cosang = float(np.dot(v1, v2) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return round(float(math.degrees(math.acos(cosang))), 1)


def compute_hand_angles(points: dict[str, HandPoint]) -> dict:
    # Ángulos iniciales útiles para flexión/extensión de dedos.
    candidates = {
        "thumb_mcp_flexion_2d": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
        "thumb_ip_flexion_2d": ("thumb_mcp", "thumb_ip", "thumb_tip"),
        "index_pip_flexion_2d": ("index_mcp", "index_pip", "index_dip"),
        "index_dip_flexion_2d": ("index_pip", "index_dip", "index_tip"),
        "middle_pip_flexion_2d": ("middle_mcp", "middle_pip", "middle_dip"),
        "middle_dip_flexion_2d": ("middle_pip", "middle_dip", "middle_tip"),
        "ring_pip_flexion_2d": ("ring_mcp", "ring_pip", "ring_dip"),
        "pinky_pip_flexion_2d": ("pinky_mcp", "pinky_pip", "pinky_dip"),
        "wrist_index_angle_2d": ("wrist", "index_mcp", "index_pip"),
    }
    out = {}
    for name, triplet in candidates.items():
        val = angle_2d(points, *triplet)
        if val is not None:
            out[name] = val
    return out


def px(p: HandPoint, w: int, h: int):
    return int(p.x * w), int(p.y * h)


def draw_hand(frame, points: dict[str, HandPoint]):
    h, w = frame.shape[:2]
    for a, b in CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, px(points[a], w, h), px(points[b], w, h), (40, 200, 120), 2, cv2.LINE_AA)
    for p in points.values():
        cv2.circle(frame, px(p, w, h), 4, (0, 180, 180), -1, cv2.LINE_AA)


def handedness_label(handedness_item):
    try:
        if handedness_item and len(handedness_item) > 0:
            cat = handedness_item[0]
            return f"{cat.category_name} ({cat.score:.2f})"
    except Exception:
        pass
    return "unknown"


def draw_panel(frame, fps, detected_count, label, angles, mirror):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 150), (12, 20, 40), -1)
    cv2.line(frame, (0, 150), (w, 150), (0, 180, 180), 2)
    cv2.putText(frame, "MediaPipe Hand21 - prueba inicial", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235,245,250), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Manos detectadas: {detected_count} | Handedness: {label} | Mirror: {'ON' if mirror else 'OFF'}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,180,180), 1, cv2.LINE_AA)

    keys = ["index_pip_flexion_2d", "middle_pip_flexion_2d", "ring_pip_flexion_2d", "pinky_pip_flexion_2d", "thumb_ip_flexion_2d"]
    vals = [f"{k.replace('_flexion_2d','')}: {angles[k]:.1f}" for k in keys if k in angles]
    line1 = " | ".join(vals[:3]) if vals else "Sin ángulos suficientes"
    line2 = " | ".join(vals[3:]) if len(vals) > 3 else ""
    cv2.putText(frame, line1[:150], (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150,170,185), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2[:150], (16, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150,170,185), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q/ESC salir | M mirror", (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,160,40), 1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] No existe {MODEL_PATH}")
        return

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mirror = True
    fps_smooth = 0.0
    last_t = time.time()
    frame_idx = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000 / 30))
            frame_idx += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            detected_count = len(result.hand_landmarks) if result and result.hand_landmarks else 0
            label = "none"
            angles = {}

            if detected_count > 0:
                # Tomamos la primera mano para métricas, pero dibujamos todas.
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    points = to_named_hand_landmarks(hand_landmarks)
                    draw_hand(frame, points)
                    if i == 0:
                        angles = compute_hand_angles(points)
                        if result.handedness and i < len(result.handedness):
                            label = handedness_label(result.handedness[i])

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            draw_panel(frame, fps_smooth, detected_count, label, angles, mirror)
            cv2.imshow("MediaPipe Hand21 - prueba", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('m'), ord('M')):
                mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
