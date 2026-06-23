
"""
test_neck_angles_camera.py

Diagnostico de cuello Pose33 SIN contar repeticiones.
Sirve para encontrar qué métrica realmente cambia con:
  1) cuello neutro
  2) cuello atras real
  3) movimiento falso de brazo/tronco

Uso:
    python test_neck_angles_camera.py

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
    L       -> logging on/off en consola
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from pose_backends.mediapipe33_backend import MediaPipe33Backend
from rehab_core.angles_33 import compute_angles_33, tracking_quality_for

MODEL_PATH = "model/pose_landmarker_lite.task"
CAMERA_INDEX = 0
MIN_CONF = 0.15

METRICS = [
    "neck_sagittal_2d",
    "neck_deviation_2d",
    "neck_lateral_signed_2d",
    "neck_extension_side_2d",
    "head_pitch_side_2d",
    "trunk_lean_2d",
    "shoulder_alignment_2d",
]

REQUIRED = ["nose", "right_ear", "right_shoulder"]
OPTIONAL = ["left_ear", "left_shoulder", "right_hip", "left_hip"]

CONNECTIONS = [
    ("nose", "right_ear"),
    ("right_ear", "right_shoulder"),
    ("left_ear", "left_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
]


def point_px(point, w, h):
    return int(point.x * w), int(point.y * h)


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    for a, b in CONNECTIONS:
        pa = landmarks.get(a)
        pb = landmarks.get(b)
        if pa is not None and pb is not None and pa.confidence >= MIN_CONF and pb.confidence >= MIN_CONF:
            cv2.line(frame, point_px(pa, w, h), point_px(pb, w, h), (40, 220, 120), 2, cv2.LINE_AA)
    for name in REQUIRED + OPTIONAL:
        p = landmarks.get(name)
        if p is not None and p.confidence >= MIN_CONF:
            color = (0, 220, 220) if name in REQUIRED else (150, 170, 185)
            cv2.circle(frame, point_px(p, w, h), 5 if name in REQUIRED else 3, color, -1, cv2.LINE_AA)
            cv2.putText(frame, name.replace("right_", "r_").replace("left_", "l_"),
                        (point_px(p, w, h)[0] + 5, point_px(p, w, h)[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_panel(frame, fps, result, angles, q, mirror, logging):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 218), (12, 20, 40), -1)
    cv2.line(frame, (0, 218), (w, 218), (0, 180, 180), 2)

    detected = "SI" if result.detected else "NO"
    cv2.putText(frame, "Diagnostico cuello Pose33 - NO cuenta reps", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235,245,250), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Detectado: {detected} | Mirror: {'ON' if mirror else 'OFF'} | Log: {'ON' if logging else 'OFF'}",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,180,180), 1, cv2.LINE_AA)

    y = 88
    for metric in METRICS:
        val = angles.get(metric)
        txt = "--" if val is None else f"{val:.1f}"
        cv2.putText(frame, f"{metric}: {txt}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (235,245,250) if metric.startswith(("neck", "head")) else (150,170,185), 1, cv2.LINE_AA)
        y += 20

    details = q.get("details", {}) if isinstance(q, dict) else {}
    parts = []
    for name in REQUIRED + OPTIONAL:
        item = details.get(name, {})
        conf = item.get("confidence", 0.0)
        ok = "+" if item.get("visible", False) else "-"
        short = name.replace("right_", "r").replace("left_", "l")
        parts.append(f"{short}:{conf:.2f}{ok}")
    cv2.putText(frame, " | ".join(parts)[:150], (16, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q/ESC salir | M mirror | L log", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,160,40), 1, cv2.LINE_AA)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] No existe {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mirror = True
    logging = True
    frame_idx = 0
    fps_smooth = 0.0
    last_t = time.time()

    print("[INFO] Prueba 1: quedate neutro 3 segundos")
    print("[INFO] Prueba 2: lleva cabeza atras sin mover tronco/brazo")
    print("[INFO] Prueba 3: mueve brazo/tronco sin mover cabeza")

    with MediaPipe33Backend(MODEL_PATH) as backend:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            result = backend.detect_bgr(frame)
            angles = {}
            q = {"details": {}}
            if result.detected:
                angles = compute_angles_33(
                    result.image_landmarks,
                    result.world_landmarks,
                    min_confidence=MIN_CONF,
                    include_3d=True,
                )
                q = tracking_quality_for(result.image_landmarks, REQUIRED + OPTIONAL, min_confidence=MIN_CONF)
                draw_pose(frame, result.image_landmarks)

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            if logging and frame_idx % 15 == 0:
                vals = " | ".join(f"{m}={('--' if angles.get(m) is None else f'{angles.get(m):.1f}')}" for m in METRICS)
                details = q.get("details", {}) if isinstance(q, dict) else {}
                confs = " | ".join(f"{n}:{details.get(n, {}).get('confidence', 0.0):.2f}{'+' if details.get(n, {}).get('visible', False) else '-'}" for n in REQUIRED + OPTIONAL)
                print(f"[NECKDBG f={frame_idx}] {vals} || {confs}")

            draw_panel(frame, fps_smooth, result, angles, q, mirror, logging)
            cv2.imshow("Diagnostico cuello Pose33", frame)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('m'), ord('M')):
                mirror = not mirror
            if key in (ord('l'), ord('L')):
                logging = not logging
                print(f"[INFO] Logging: {'ON' if logging else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
