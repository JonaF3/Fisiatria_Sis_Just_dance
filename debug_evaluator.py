"""
Debug: print EVERY frame's angle, in_start, in_target, phase
"""
import sys, cv2
from pathlib import Path
from pose_backends.mediapipe33_backend import MediaPipe33Backend
from rehab_core.angles_33 import compute_angles_33, tracking_quality_for
from rehab_core.pose33_trajectory_evaluator import Pose33TrajectoryEvaluator
from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS

MODEL_PATH = "model/pose_landmarker_lite.task"
video_key = sys.argv[1] if len(sys.argv) > 1 else "tobillo_dentro"
video_path = f"rehab_references/videos/{video_key}.mov"
if not Path(video_path).exists():
    video_path = video_path.replace(".mov", ".mp4")

cfg = REHAB_EXERCISE_CONFIGS[video_key]
primary = cfg["primary_angle"]
min_conf = cfg.get("min_tracking_confidence", 0.35)
req_lm = cfg.get("required_landmarks", [])
start_r = cfg["start_ranges"][primary]
target_r = cfg["target_ranges"][primary]

print(f"Exercise: {video_key} | primary: {primary}")
print(f"start_range: {start_r} | target_range: {target_r}")
print(f"direction: {cfg.get('direction')} | auto_calibrate: {cfg.get('auto_calibrate')}")
print()

evaluator = Pose33TrajectoryEvaluator(cfg)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open {video_path}")
    exit()

with MediaPipe33Backend(MODEL_PATH) as backend:
    for idx in range(200):
        ret, frame = cap.read()
        if not ret: break

        result = backend.detect_bgr(frame)
        if not result.detected:
            continue

        angles = compute_angles_33(
            result.image_landmarks, result.world_landmarks,
            min_confidence=min_conf, include_3d=True,
        )
        tracking_quality = tracking_quality_for(
            result.image_landmarks, req_lm,
            min_confidence=min_conf,
        )
        eval_result = evaluator.evaluate(
            angles=angles, pose33_result=result,
            tracking_quality=tracking_quality,
            frame_index=idx, timestamp_s=idx / 30.0,
        )

        primary_val = angles.get(primary, None)
        pv = f"{primary_val:.3f}" if primary_val is not None else "None"
        phase = eval_result.get("phase", "?")
        reps = eval_result.get("completed_reps", 0)
        in_st = eval_result.get("in_start", "?")
        in_tg = eval_result.get("in_target", "?")
        in_tgp = eval_result.get("in_target_primary_only", "?")
        fis = eval_result.get("frames_in_start", "?")
        fit = eval_result.get("frames_in_target", "?")
        cal = eval_result.get("calibrated", "?")
        smoothed = eval_result.get("primary_value", None)
        sv = f"{smoothed:.3f}" if smoothed is not None else "None"
        feedback = eval_result.get("feedback", "")[:40]

        print(f"{idx:>4} | raw={pv:>8} | sm={sv:>8} | {phase:>15} | st={in_st} | tg={in_tg} | tg1={in_tgp} | fis={fis} | fit={fit} | cal={cal} | {feedback}")

cap.release()
