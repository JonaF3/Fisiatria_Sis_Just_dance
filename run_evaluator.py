"""
Run full Pose33TrajectoryEvaluator on a rehab video.
Shows phase transitions, rep counting, and angle values.
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

evaluator = Pose33TrajectoryEvaluator(cfg)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open {video_path}")
    exit()

print(f"Exercise: {video_key}")
print(f"Primary: {primary}")
print(f"Direction: {cfg.get('direction')}")
print(f"auto_calibrate: {cfg.get('auto_calibrate')}")
print(f"target_delta: {cfg.get('target_delta')}")
print(f"{'Frame':>5} | {'Phase':>18} | {'Primary':>8} | {'Reps':>4} | {'Feedback'}")

with MediaPipe33Backend(MODEL_PATH) as backend:
    for idx in range(150):
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
        pv = f"{primary_val:.1f}" if primary_val is not None else "None"
        phase = eval_result.get("phase", "?")
        reps = eval_result.get("completed_reps", 0)
        feedback = str(eval_result.get("feedback", ""))[:60]
        start_range = evaluator.state.dyn_start_range
        target_range = evaluator.state.dyn_target_range

        if idx % 10 == 0 or eval_result.get("rep_completed"):
            marker = " *** REP ***" if eval_result.get("rep_completed") else ""
            print(f"{idx:>5} | {phase:>18} | {pv:>8} | {reps:>4} | {feedback[:50]}{marker}")

cap.release()

state = evaluator.state
print(f"\nFinal: {state.completed_reps} reps completed")
print(f"Phase: {state.phase}")
print(f"Calibrated: {state.calibrated}")
print(f"Neutral: {state.neutral_value}")
if state.dyn_start_range:
    print(f"Dyn start: {[f'{v:.1f}' for v in state.dyn_start_range]}")
if state.dyn_target_range:
    print(f"Dyn target: {[f'{v:.1f}' for v in state.dyn_target_range]}")
