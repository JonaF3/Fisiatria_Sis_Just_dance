
"""
test_pose33_evaluator_camera.py

Paso 8: prueba en cámara del evaluador Pose33.

Objetivo:
    Validar fuera del juego que el flujo completo funciona:
        MediaPipe33Backend -> angles_33 -> Pose33TrajectoryEvaluator

Uso:
    python test_pose33_evaluator_camera.py

Controles:
    Q / ESC -> salir
    M       -> mirror on/off
    R       -> reset evaluador

Qué deberías ver:
    - fase: waiting_start / going_target / returning
    - repeticiones completadas
    - ángulo primario right_shoulder_flexion_3d
    - feedback clínico
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
from pose_backends.mediapipe33_backend import MediaPipe33Backend
from rehab_core.angles_33 import compute_angles_33, tracking_quality_for
from rehab_core.pose33_trajectory_evaluator import Pose33TrajectoryEvaluator

MODEL_PATH = "model/pose_landmarker_lite.task"
CAMERA_INDEX = 0
EXERCISE_KEY = "Shoulder Flexion and Extension"

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
    for _, point in landmarks.items():
        if point.confidence >= 0.25:
            cv2.circle(frame, point_px(point, w, h), 3, (0, 180, 180), -1, cv2.LINE_AA)


def phase_label(phase: str) -> str:
    return {
        "waiting_start": "ESPERANDO INICIO",
        "going_target": "SUBE AL OBJETIVO",
        "returning": "REGRESA AL INICIO",
    }.get(phase, str(phase).upper())


def draw_panel(frame, fps, result, angles, eval_result, reps_target, mirror):
    h, w = frame.shape[:2]
    bg = (12, 20, 40)
    teal = (0, 180, 180)
    green = (40, 200, 120)
    amber = (255, 160, 40)
    text = (235, 245, 250)
    muted = (150, 170, 185)

    cv2.rectangle(frame, (0, 0), (w, 182), bg, -1)
    cv2.line(frame, (0, 182), (w, 182), teal, 2)

    cv2.putText(frame, "Paso 8 - Evaluador Pose33 en camara", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, text, 2, cv2.LINE_AA)

    detected = "SI" if result.detected else "NO"
    cv2.putText(frame, f"FPS: {fps:.1f} | Detectado: {detected} | Tracking: {result.tracking_quality:.2f} | Mirror: {'ON' if mirror else 'OFF'}",
                (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, teal, 1, cv2.LINE_AA)

    phase = phase_label(eval_result.get("phase", ""))
    completed = eval_result.get("completed_reps", 0)
    cv2.putText(frame, f"FASE: {phase} | REPS: {completed}/{reps_target}",
                (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.62, green, 2, cv2.LINE_AA)

    primary_name = eval_result.get("primary_angle", "primary")
    primary_value = eval_result.get("primary_value")
    confirmation_value = eval_result.get("confirmation_value")
    pv = "--" if primary_value is None else f"{primary_value:.1f}"
    cv = "--" if confirmation_value is None else f"{confirmation_value:.1f}"
    cv2.putText(frame, f"{primary_name}: {pv} | confirmacion 2D: {cv}",
                (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1, cv2.LINE_AA)

    feedback = str(eval_result.get("feedback", ""))[:120]
    color = amber if not eval_result.get("ok", True) else text
    cv2.putText(frame, feedback, (16, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    flags = eval_result.get("compensations", {}) or {}
    active_flags = [k for k, v in flags.items() if v]
    flag_text = "Compensaciones: " + (", ".join(active_flags) if active_flags else "OK")
    cv2.putText(frame, flag_text[:120], (16, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.43, muted, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q/ESC salir | M mirror | R reset", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 160, 40), 1, cv2.LINE_AA)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] No existe {MODEL_PATH}")
        return

    cfg = REHAB_EXERCISE_CONFIGS.get(EXERCISE_KEY)
    if not cfg:
        print(f"[ERROR] No existe '{EXERCISE_KEY}' en just_dance_rehab_config.py")
        return

    reps_target = 5
    evaluator = Pose33TrajectoryEvaluator(cfg)

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
    fps_smooth = 0.0
    last_t = time.time()
    frame_idx = 0
    start_time = time.time()

    with MediaPipe33Backend(MODEL_PATH) as backend:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            timestamp_s = time.time() - start_time
            result = backend.detect_bgr(frame)
            angles = {}
            eval_result = {
                "ok": False,
                "phase": evaluator.state.phase,
                "completed_reps": evaluator.state.completed_reps,
                "feedback": "Ajusta encuadre para detectar pose.",
                "compensations": {},
            }

            if result.detected:
                angles = compute_angles_33(
                    result.image_landmarks,
                    result.world_landmarks,
                    min_confidence=cfg.get("min_tracking_confidence", 0.35),
                    include_3d=True,
                )
                tracking_quality = tracking_quality_for(
                    result.image_landmarks,
                    cfg.get("required_landmarks", []),
                    min_confidence=cfg.get("min_tracking_confidence", 0.35),
                )
                eval_result = evaluator.evaluate(
                    angles=angles,
                    pose33_result=result,
                    tracking_quality=tracking_quality,
                    frame_index=frame_idx,
                    timestamp_s=timestamp_s,
                )
                draw_pose(frame, result.image_landmarks)

                if eval_result.get("rep_completed"):
                    print(f"[POSE33] Rep completada: {eval_result.get('completed_reps')} | last_rep={eval_result.get('last_rep')}")

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst if fps_smooth <= 0 else fps_smooth * 0.9 + inst * 0.1

            draw_panel(frame, fps_smooth, result, angles, eval_result, reps_target, mirror)
            cv2.imshow("Paso 8 - Pose33 Evaluator Camera", frame)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            if key in (ord("r"), ord("R")):
                evaluator = Pose33TrajectoryEvaluator(cfg)
                print("[INFO] Evaluador reiniciado")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
