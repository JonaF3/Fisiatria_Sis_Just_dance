"""
test_hand21_evaluator_camera.py

Prueba del evaluador Hand21 estricto:
    mano abierta estable -> mano cerrada estable -> mano abierta estable = 1 repetición

Uso:
    python test_hand21_evaluator_camera.py

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
    R       -> reset evaluador
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend
from rehab_core.hand21_evaluator import Hand21OpenCloseEvaluator

MODEL_PATH = "model/hand_landmarker.task"
CAMERA_INDEX = 0

CONNECTIONS = [
    ("wrist", "thumb_cmc"), ("thumb_cmc", "thumb_mcp"), ("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"), ("index_mcp", "index_pip"), ("index_pip", "index_dip"), ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"), ("middle_mcp", "middle_pip"), ("middle_pip", "middle_dip"), ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"), ("ring_mcp", "ring_pip"), ("ring_pip", "ring_dip"), ("ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp"), ("pinky_mcp", "pinky_pip"), ("pinky_pip", "pinky_dip"), ("pinky_dip", "pinky_tip"),
]

CONFIG = {
            
    "open_range": [145.0, 180.0],
    "closed_range": [0.0, 105.0],
    "min_open_fingers": 4,
    "min_closed_fingers": 3,
    "min_stable_frames": 5,
    "cooldown_frames": 10,
    "target_repetitions": 5,
    "stop_at_target": True,



}


def px(point, w, h):
    return int(point.x * w), int(point.y * h)


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    landmarks = hand.landmarks
    for a, b in CONNECTIONS:
        if a in landmarks and b in landmarks:
            cv2.line(frame, px(landmarks[a], w, h), px(landmarks[b], w, h), (40, 220, 120), 2, cv2.LINE_AA)
    for point in landmarks.values():
        cv2.circle(frame, px(point, w, h), 4, (0, 220, 220), -1, cv2.LINE_AA)


def choose_hand(result):
    if not result.detected or not result.hands:
        return None
    return max(result.hands, key=lambda h: h.tracking_quality)


def phase_label(phase):
    return {
        "waiting_open": "ESPERANDO MANO ABIERTA",
        "going_closed": "CIERRA LA MANO",
        "returning_open": "ABRE LA MANO",
        "done": "COMPLETADO",
    }.get(str(phase), str(phase).upper())


def draw_panel(frame, fps, result, eval_result, mirror, target_reps=5):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 190), (12, 20, 40), -1)
    cv2.line(frame, (0, 190), (w, 190), (0, 180, 180), 2)

    cv2.putText(frame, "Hand21 Evaluator v2 - estricto", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235,245,250), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Manos: {result.num_hands} | Mirror: {'ON' if mirror else 'OFF'}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,180,180), 1, cv2.LINE_AA)

    phase = phase_label(eval_result.get("phase", ""))
    reps = eval_result.get("completed_reps", 0)
    cv2.putText(frame, f"FASE: {phase} | REPS: {reps}/{target_reps}", (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (40,220,120), 2, cv2.LINE_AA)

    avg = eval_result.get("avg_pip_flexion")
    avg_txt = "--" if avg is None else f"{avg:.1f}"
    state = eval_result.get("hand_state", "--")
    quality = eval_result.get("tracking_quality", 0.0)
    label = eval_result.get("handedness_label", "unknown")
    cv2.putText(frame, f"{label} | quality: {quality:.2f} | estado: {state} | avg dedos: {avg_txt}", (16, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150,170,185), 1, cv2.LINE_AA)

    open_count = eval_result.get("open_count", 0)
    closed_count = eval_result.get("closed_count", 0)
    open_streak = eval_result.get("open_streak", 0)
    closed_streak = eval_result.get("closed_streak", 0)
    min_stable = eval_result.get("min_stable_frames", 5)
    cv2.putText(frame, f"abiertos: {open_count}/4 streak {open_streak}/{min_stable} | cerrados: {closed_count}/4 streak {closed_streak}/{min_stable}", (16, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,170,185), 1, cv2.LINE_AA)

    feedback = str(eval_result.get("feedback", ""))[:120]
    cv2.putText(frame, feedback, (16, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,220,120), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q/ESC salir | M mirror | R reset", (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,160,40), 1, cv2.LINE_AA)


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

    evaluator = Hand21OpenCloseEvaluator(CONFIG)
    mirror = True
    fps_smooth = 0.0
    last_t = time.time()
    frame_idx = 0
    start_time = time.time()

    with MediaPipeHand21Backend(MODEL_PATH, num_hands=1) as backend:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            result = backend.detect_bgr(frame)
            hand = choose_hand(result)
            eval_result = {
                "ok": False,
                "phase": evaluator.state.phase,
                "completed_reps": evaluator.state.completed_reps,
                "feedback": "Muestra una mano a la cámara.",
                "tracking_quality": 0.0,
                "hand_state": "sin mano",
                "open_count": 0,
                "closed_count": 0,
                "open_streak": 0,
                "closed_streak": 0,
                "min_stable_frames": CONFIG["min_stable_frames"],
            }

            if hand is not None:
                draw_hand(frame, hand)
                eval_result = evaluator.evaluate(
                    hand_result=hand,
                    frame_index=frame_idx,
                    timestamp_s=time.time() - start_time,
                )
                if eval_result.get("rep_completed"):
                    print(f"[HAND21] Rep completada: {eval_result.get('completed_reps')} | last_rep={eval_result.get('last_rep')}")

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            draw_panel(frame, fps_smooth, result, eval_result, mirror, target_reps=CONFIG["target_repetitions"])
            cv2.imshow("Hand21 Evaluator v2", frame)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            if key in (ord("r"), ord("R")):
                evaluator = Hand21OpenCloseEvaluator(CONFIG)
                print("[INFO] Evaluador Hand21 reiniciado")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
