
"""
test_mediapipe33_pipeline.py

Paso 3: prueba conjunta del backend real MediaPipe33Backend + rehab_core.angles_33.
No convierte a MoveNet 17. Usa los 33 landmarks completos.

Uso:
    python test_mediapipe33_pipeline.py

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from pose_backends.mediapipe33_backend import MediaPipe33Backend
from rehab_core.angles_33 import compute_angles_33, tracking_quality_for

MODEL_PATH = "model/pose_landmarker_lite.task"
CAMERA_INDEX = 0

REQUIRED_SHOULDER_RIGHT = [
    "right_shoulder",
    "right_elbow",
    "right_hip",
]

DRAW_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def point_px(point, w, h):
    return int(point.x * w), int(point.y * h)


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    for a, b in DRAW_CONNECTIONS:
        pa = landmarks.get(a)
        pb = landmarks.get(b)
        if pa and pb and pa.confidence >= 0.25 and pb.confidence >= 0.25:
            cv2.line(frame, point_px(pa, w, h), point_px(pb, w, h), (40, 200, 120), 2, cv2.LINE_AA)
    for name, point in landmarks.items():
        if point.confidence >= 0.25:
            cv2.circle(frame, point_px(point, w, h), 3, (0, 180, 180), -1, cv2.LINE_AA)


def draw_panel(frame, fps, result, angles, quality_report, mirror):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 154), (12, 20, 40), -1)
    cv2.line(frame, (0, 154), (w, 154), (0, 180, 180), 2)

    cv2.putText(
        frame,
        "Paso 3 - MediaPipe33Backend + angles_33",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 245, 250),
        2,
        cv2.LINE_AA,
    )

    status = "SI" if result.detected else "NO"
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Detectado: {status} | Tracking: {result.tracking_quality:.2f} | Mirror: {'ON' if mirror else 'OFF'}",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 180, 180),
        1,
        cv2.LINE_AA,
    )

    ok = quality_report.get("ok", False)
    q_col = (40, 200, 120) if ok else (40, 160, 255)
    cv2.putText(
        frame,
        f"Required hombro derecho: {'OK' if ok else 'INCOMPLETO'} | mean_conf: {quality_report.get('mean_confidence', 0):.2f}",
        (16, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        q_col,
        1,
        cv2.LINE_AA,
    )

    keys = [
        "right_shoulder_flexion_2d",
        "right_shoulder_flexion_3d",
        "right_elbow_flexion_2d",
        "trunk_lean_2d",
    ]
    txt = []
    for key in keys:
        if key in angles:
            short = key.replace("right_", "R_").replace("_flexion", "").replace("shoulder", "sh").replace("elbow", "elbow")
            txt.append(f"{short}: {angles[key]:.1f}")
    line = " | ".join(txt) if txt else "Sin angulos suficientes"
    cv2.putText(frame, line[:150], (16, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 170, 185), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q/ESC salir | M mirror", (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 160, 40), 1, cv2.LINE_AA)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] No existe {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
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

    with MediaPipe33Backend(MODEL_PATH) as backend:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            result = backend.detect_bgr(frame)
            angles = {}
            quality = {"ok": False, "mean_confidence": 0.0, "details": {}}

            if result.detected:
                angles = compute_angles_33(
                    result.image_landmarks,
                    result.world_landmarks,
                    min_confidence=0.35,
                    include_3d=True,
                )
                quality = tracking_quality_for(
                    result.image_landmarks,
                    REQUIRED_SHOULDER_RIGHT,
                    min_confidence=0.35,
                )
                draw_pose(frame, result.image_landmarks)

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            draw_panel(frame, fps_smooth, result, angles, quality, mirror)
            cv2.imshow("Paso 3 - MediaPipe33 + angles_33", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
