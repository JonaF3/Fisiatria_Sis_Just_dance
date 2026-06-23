
"""
test_hand21_backend_camera.py

Paso 3 de mano: prueba conjunta usando el backend real MediaPipeHand21Backend.

Flujo:
    cámara OpenCV
    -> MediaPipeHand21Backend
    -> hand.angles
    -> visualización/debug

Uso:
    python test_hand21_backend_camera.py

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
    H       -> cambiar mano principal: auto / Left / Right
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend

MODEL_PATH = "model/hand_landmarker.task"
CAMERA_INDEX = 0

CONNECTIONS = [
    ("wrist", "thumb_cmc"), ("thumb_cmc", "thumb_mcp"), ("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"), ("index_mcp", "index_pip"), ("index_pip", "index_dip"), ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"), ("middle_mcp", "middle_pip"), ("middle_pip", "middle_dip"), ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"), ("ring_mcp", "ring_pip"), ("ring_pip", "ring_dip"), ("ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp"), ("pinky_mcp", "pinky_pip"), ("pinky_pip", "pinky_dip"), ("pinky_dip", "pinky_tip"),
    ("index_mcp", "middle_mcp"), ("middle_mcp", "ring_mcp"), ("ring_mcp", "pinky_mcp"),
]

HAND_MODE_SEQUENCE = ["auto", "Left", "Right"]


def px(point, w, h):
    return int(point.x * w), int(point.y * h)


def draw_hand(frame, hand, is_selected=False):
    h, w = frame.shape[:2]
    landmarks = hand.landmarks
    line_color = (40, 220, 120) if is_selected else (90, 120, 110)
    dot_color = (0, 220, 220) if is_selected else (120, 150, 150)
    thickness = 3 if is_selected else 1

    for a, b in CONNECTIONS:
        if a in landmarks and b in landmarks:
            cv2.line(frame, px(landmarks[a], w, h), px(landmarks[b], w, h), line_color, thickness, cv2.LINE_AA)

    for point in landmarks.values():
        cv2.circle(frame, px(point, w, h), 4 if is_selected else 3, dot_color, -1, cv2.LINE_AA)

    wrist = landmarks.get("wrist")
    if wrist is not None:
        label = f"{hand.handedness_label} {hand.handedness_score:.2f}"
        cv2.putText(frame, label, px(wrist, w, h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 80), 1, cv2.LINE_AA)


def choose_hand(result, mode="auto"):
    if not result.detected or not result.hands:
        return None
    if mode == "auto":
        return max(result.hands, key=lambda h: h.tracking_quality)
    candidates = [h for h in result.hands if h.handedness_label == mode]
    if candidates:
        return max(candidates, key=lambda h: h.tracking_quality)
    return max(result.hands, key=lambda h: h.tracking_quality)


def finger_average_flexion(angles: dict):
    keys = [
        "index_pip_flexion_2d",
        "middle_pip_flexion_2d",
        "ring_pip_flexion_2d",
        "pinky_pip_flexion_2d",
    ]
    vals = [float(angles[k]) for k in keys if k in angles]
    if not vals:
        return None
    return sum(vals) / len(vals)


def hand_state_from_angles(angles: dict):
    avg = finger_average_flexion(angles)
    if avg is None:
        return "sin datos", None
    # En esta geometría: mano abierta ≈ 160-180, puño/cerrada baja bastante.
    if avg >= 145:
        return "mano abierta", avg
    if avg <= 95:
        return "mano cerrada", avg
    return "transición", avg


def draw_panel(frame, fps, result, selected_hand, mode, mirror):
    h, w = frame.shape[:2]
    bg = (12, 20, 40)
    teal = (0, 180, 180)
    text = (235, 245, 250)
    muted = (150, 170, 185)
    amber = (255, 170, 50)

    cv2.rectangle(frame, (0, 0), (w, 172), bg, -1)
    cv2.line(frame, (0, 172), (w, 172), teal, 2)

    cv2.putText(frame, "Paso 3 Hand21 - Backend real", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Manos: {result.num_hands} | Modo mano: {mode} | Mirror: {'ON' if mirror else 'OFF'}",
        (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, teal, 1, cv2.LINE_AA
    )

    if selected_hand is None:
        cv2.putText(frame, "No hay mano seleccionada", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, amber, 1, cv2.LINE_AA)
        return

    angles = selected_hand.angles
    state, avg = hand_state_from_angles(angles)
    avg_txt = "--" if avg is None else f"{avg:.1f}"
    cv2.putText(
        frame,
        f"Seleccionada: {selected_hand.handedness_label} ({selected_hand.handedness_score:.2f}) | quality: {selected_hand.tracking_quality:.2f} | estado: {state} | avg dedos: {avg_txt}",
        (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.46, muted, 1, cv2.LINE_AA
    )

    keys1 = ["index_pip_flexion_2d", "middle_pip_flexion_2d", "ring_pip_flexion_2d", "pinky_pip_flexion_2d"]
    values1 = []
    for key in keys1:
        if key in angles:
            short = key.replace("_flexion_2d", "").replace("index", "idx").replace("middle", "mid")
            values1.append(f"{short}: {angles[key]:.1f}")
    line1 = " | ".join(values1) if values1 else "Sin ángulos PIP"
    cv2.putText(frame, line1[:150], (16, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, muted, 1, cv2.LINE_AA)

    keys2 = ["thumb_ip_flexion_2d", "index_mcp_flexion_2d", "middle_mcp_flexion_2d"]
    values2 = []
    for key in keys2:
        if key in angles:
            short = key.replace("_flexion_2d", "").replace("thumb", "th").replace("index", "idx").replace("middle", "mid")
            values2.append(f"{short}: {angles[key]:.1f}")
    line2 = " | ".join(values2) if values2 else ""
    cv2.putText(frame, line2[:150], (16, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.44, muted, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q/ESC salir | M mirror | H mano auto/Left/Right", (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, amber, 1, cv2.LINE_AA)


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
        print("[ERROR] No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mirror = True
    mode_idx = 0
    fps_smooth = 0.0
    last_t = time.time()

    with MediaPipeHand21Backend(MODEL_PATH, num_hands=2) as backend:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            result = backend.detect_bgr(frame)
            mode = HAND_MODE_SEQUENCE[mode_idx]
            selected = choose_hand(result, mode)

            if result.detected:
                for hand in result.hands:
                    draw_hand(frame, hand, is_selected=(hand is selected))

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            draw_panel(frame, fps_smooth, result, selected, mode, mirror)
            cv2.imshow("Paso 3 Hand21 - Backend", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            if key in (ord("h"), ord("H")):
                mode_idx = (mode_idx + 1) % len(HAND_MODE_SEQUENCE)
                print(f"[INFO] Modo mano: {HAND_MODE_SEQUENCE[mode_idx]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
