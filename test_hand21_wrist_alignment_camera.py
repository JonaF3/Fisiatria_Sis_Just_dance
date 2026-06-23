
"""
test_hand21_wrist_alignment_camera.py

Diagnostico Hand21 para ejercicios de mano/muneca SIN contar repeticiones.
Muestra metricas utiles para decidir como validar ejercicios tipo:
  - dedos arriba / mano recta
  - flexion-extension de muneca
  - postura de mano extendida

Uso:
    python test_hand21_wrist_alignment_camera.py

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
    L       -> logging on/off
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend

MODEL_PATH = "model/hand_landmarker.task"
CAMERA_INDEX = 0
MIN_QUALITY = 0.35

CONNECTIONS = [
    ("wrist", "thumb_cmc"), ("thumb_cmc", "thumb_mcp"), ("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"), ("index_mcp", "index_pip"), ("index_pip", "index_dip"), ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"), ("middle_mcp", "middle_pip"), ("middle_pip", "middle_dip"), ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"), ("ring_mcp", "ring_pip"), ("ring_pip", "ring_dip"), ("ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp"), ("pinky_mcp", "pinky_pip"), ("pinky_pip", "pinky_dip"), ("pinky_dip", "pinky_tip"),
]

FINGER_PIP_KEYS = [
    "index_pip_flexion_2d",
    "middle_pip_flexion_2d",
    "ring_pip_flexion_2d",
    "pinky_pip_flexion_2d",
]


def px(point, w, h):
    return int(point.x * w), int(point.y * h)


def angle_of_vector_deg(a, b):
    dx = float(b.x - a.x)
    dy = float(b.y - a.y)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    # 0 = horizontal derecha, 90 = vertical abajo, -90 = vertical arriba
    return round(float(math.degrees(math.atan2(dy, dx))), 2)


def abs_horizontal_angle(angle):
    if angle is None:
        return None
    # distancia angular contra horizontal en grados, 0 horizontal, 90 vertical
    a = abs(float(angle))
    if a > 90:
        a = 180 - a
    return round(a, 2)


def compute_hand_metrics(hand):
    lm = hand.landmarks
    angles = getattr(hand, "angles", {}) or {}
    metrics = {}

    wrist = lm.get("wrist")
    middle_mcp = lm.get("middle_mcp")
    middle_tip = lm.get("middle_tip")
    index_mcp = lm.get("index_mcp")
    pinky_mcp = lm.get("pinky_mcp")

    if wrist is not None and middle_mcp is not None:
        axis = angle_of_vector_deg(wrist, middle_mcp)
        metrics["hand_axis_angle_2d"] = axis
        metrics["hand_axis_from_horizontal"] = abs_horizontal_angle(axis)

    if wrist is not None and middle_tip is not None:
        long_axis = angle_of_vector_deg(wrist, middle_tip)
        metrics["wrist_to_middle_tip_angle_2d"] = long_axis
        metrics["wrist_to_middle_tip_from_horizontal"] = abs_horizontal_angle(long_axis)

    if index_mcp is not None and pinky_mcp is not None:
        palm_line = angle_of_vector_deg(index_mcp, pinky_mcp)
        metrics["palm_mcp_line_angle_2d"] = palm_line
        metrics["palm_mcp_line_from_horizontal"] = abs_horizontal_angle(palm_line)

    open_count = 0
    pip_vals = []
    for key in FINGER_PIP_KEYS:
        v = angles.get(key)
        if v is not None:
            pip_vals.append(float(v))
            if 145.0 <= float(v) <= 180.0:
                open_count += 1
    metrics["open_count_4"] = open_count
    metrics["avg_pip_flexion_2d"] = round(sum(pip_vals) / len(pip_vals), 2) if pip_vals else None
    metrics["thumb_ip_flexion_2d"] = angles.get("thumb_ip_flexion_2d")
    return metrics


def choose_hand(result):
    if not result.detected or not result.hands:
        return None
    return max(result.hands, key=lambda h: h.tracking_quality)


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    lm = hand.landmarks
    for a, b in CONNECTIONS:
        if a in lm and b in lm:
            cv2.line(frame, px(lm[a], w, h), px(lm[b], w, h), (40, 220, 120), 2, cv2.LINE_AA)
    for name, point in lm.items():
        cv2.circle(frame, px(point, w, h), 4, (0, 220, 220), -1, cv2.LINE_AA)
        if name in ("wrist", "middle_mcp", "middle_tip", "index_mcp", "pinky_mcp"):
            cv2.putText(frame, name, (px(point, w, h)[0] + 4, px(point, w, h)[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 245, 250), 1, cv2.LINE_AA)


def draw_panel(frame, fps, hand, metrics, mirror, logging):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 220), (12, 20, 40), -1)
    cv2.line(frame, (0, 220), (w, 220), (180, 120, 255), 2)
    detected = "SI" if hand is not None else "NO"
    quality = 0.0 if hand is None else hand.tracking_quality
    label = "--" if hand is None else getattr(hand, "handedness_label", "--")
    cv2.putText(frame, "Diagnostico Hand21 muneca/mano - NO cuenta reps", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235,245,250), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Detectado: {detected} | Calidad: {quality:.2f} | Mano: {label} | Mirror: {'ON' if mirror else 'OFF'} | Log: {'ON' if logging else 'OFF'}",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,120,255), 1, cv2.LINE_AA)

    rows = [
        "hand_axis_angle_2d",
        "hand_axis_from_horizontal",
        "wrist_to_middle_tip_angle_2d",
        "wrist_to_middle_tip_from_horizontal",
        "palm_mcp_line_angle_2d",
        "palm_mcp_line_from_horizontal",
        "open_count_4",
        "avg_pip_flexion_2d",
        "thumb_ip_flexion_2d",
    ]
    y = 88
    for key in rows:
        val = metrics.get(key)
        txt = "--" if val is None else str(val)
        cv2.putText(frame, f"{key}: {txt}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235,245,250), 1, cv2.LINE_AA)
        y += 18
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

    print("[INFO] Prueba A: mano en posicion inicial del video")
    print("[INFO] Prueba B: movimiento objetivo")
    print("[INFO] Prueba C: falso positivo: mueve brazo sin cambiar mano")

    with MediaPipeHand21Backend(MODEL_PATH, num_hands=1) as backend:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            result = backend.detect_bgr(frame)
            hand = choose_hand(result)
            metrics = {}
            if hand is not None:
                metrics = compute_hand_metrics(hand)
                draw_hand(frame, hand)

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            if logging and frame_idx % 15 == 0:
                keys = [
                    "hand_axis_angle_2d", "hand_axis_from_horizontal",
                    "wrist_to_middle_tip_angle_2d", "wrist_to_middle_tip_from_horizontal",
                    "open_count_4", "avg_pip_flexion_2d", "thumb_ip_flexion_2d",
                ]
                vals = " | ".join(f"{k}={metrics.get(k, '--')}" for k in keys)
                q = 0.0 if hand is None else hand.tracking_quality
                print(f"[HANDDBG f={frame_idx}] q={q:.2f} | {vals}")

            draw_panel(frame, fps_smooth, hand, metrics, mirror, logging)
            cv2.imshow("Diagnostico Hand21 muneca/mano", frame)

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
