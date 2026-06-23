"""
extract_poses_hand21.py

Extractor de trayectoria para ejercicios de mano usando MediaPipe Hand21.
No toca el juego principal.

Ejemplo:
    python extract_poses_hand21.py --video "rehab_references/videos/pueba mano.mp4" --exercise_key "pueba mano" --frame_skip 1

Salida:
    rehab_references/trajectories/<exercise_key>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend
from rehab_core.hand21_evaluator import average_pip_flexion, PIP_KEYS


def moving_average(values, window=5):
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    out = []
    for idx in range(len(arr)):
        start = max(0, idx - window // 2)
        end = min(len(arr), idx + window // 2 + 1)
        out.append(float(np.mean(arr[start:end])))
    return out


def choose_hand(result, preferred_handedness="auto"):
    if not result.detected or not result.hands:
        return None
    if preferred_handedness != "auto":
        candidates = [h for h in result.hands if h.handedness_label == preferred_handedness]
        if candidates:
            return max(candidates, key=lambda h: h.tracking_quality)
    return max(result.hands, key=lambda h: h.tracking_quality)


def percentile_range(values, low_pct, high_pct, margin=5.0, lo=0.0, hi=180.0):
    arr = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(arr, low_pct)) - margin
    high = float(np.percentile(arr, high_pct)) + margin
    return [round(max(lo, min(hi, low)), 2), round(max(lo, min(hi, high)), 2)]


def compact_landmarks(hand):
    return {name: point.as_dict() for name, point in hand.landmarks.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--exercise_key", required=True)
    parser.add_argument("--output", default="rehab_references/trajectories")
    parser.add_argument("--model", default="model/hand_landmarker.task")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--preferred_handedness", default="auto", choices=["auto", "Left", "Right"])
    parser.add_argument("--smooth_window", type=int, default=5)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"[ERROR] No existe video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] No se pudo abrir video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps > 0 else 0.0

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f} | Frames: {total_frames} | Duracion: {duration_s:.2f}s")

    samples = []
    skipped_no_hand = 0
    frame_idx = 0

    with MediaPipeHand21Backend(args.model, num_hands=2) as backend:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % max(1, args.frame_skip) == 0:
                result = backend.detect_bgr(frame)
                hand = choose_hand(result, args.preferred_handedness)
                if hand is None:
                    skipped_no_hand += 1
                else:
                    avg = average_pip_flexion(hand.angles)
                    if avg is not None:
                        samples.append({
                            "frame_index": frame_idx,
                            "time_s": round(frame_idx / fps, 4),
                            "progress_raw": round(frame_idx / max(total_frames - 1, 1), 5),
                            "handedness_label": hand.handedness_label,
                            "handedness_score": round(float(hand.handedness_score), 4),
                            "tracking_quality": round(float(hand.tracking_quality), 4),
                            "avg_pip_flexion_2d": round(float(avg), 2),
                            "angles": hand.angles,
                            "landmarks": compact_landmarks(hand),
                        })

            frame_idx += 1

    cap.release()

    if len(samples) < 5:
        raise SystemExit(f"[ERROR] Muy pocas muestras validas: {len(samples)} | skipped_no_hand={skipped_no_hand}")

    values = [float(s["avg_pip_flexion_2d"]) for s in samples]
    smoothed = moving_average(values, window=max(1, args.smooth_window))
    for sample, val in zip(samples, smoothed):
        sample["avg_pip_flexion_2d"] = round(float(val), 2)

    arr = np.asarray(smoothed, dtype=np.float32)
    stats = {
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "start": round(float(arr[0]), 2),
        "end": round(float(arr[-1]), 2),
    }

    # En esta métrica: abierto = valores altos, cerrado = valores bajos.
    open_range = percentile_range(smoothed, 80, 100, margin=5.0)
    closed_range = percentile_range(smoothed, 0, 20, margin=5.0)
    safe_range = [0.0, 180.0]

    idxs = np.linspace(0, len(samples) - 1, min(31, len(samples))).astype(int)
    trajectory = []
    for out_idx, src_idx in enumerate(idxs):
        sample = samples[int(src_idx)]
        trajectory.append({
            "progress": round(out_idx / max(len(idxs) - 1, 1), 5),
            "source_frame_index": sample["frame_index"],
            "source_time_s": sample["time_s"],
            "avg_pip_flexion_2d": sample["avg_pip_flexion_2d"],
            "angles": sample["angles"],
            "tracking_quality": sample["tracking_quality"],
            "handedness_label": sample["handedness_label"],
            "landmarks": sample["landmarks"],
        })

    output = {
        "schema_version": "rehab_trajectory_hand21_v1",
        "exercise_id": args.exercise_key,
        "name": args.exercise_key,
        "tracking_type": "hand21",
        "landmark_schema": "mediapipe_hand_21",
        "reference_type": "single_rep_open_close",
        "source_video": str(video_path).replace("\\", "/"),
        "metadata": {
            "fps": round(float(fps), 3),
            "total_frames": total_frames,
            "duration_s": round(float(duration_s), 3),
            "frame_skip": args.frame_skip,
            "valid_samples": len(samples),
            "skipped_no_hand": skipped_no_hand,
            "avg_pip_stats": stats,
        },
        "primary_metric": "avg_pip_flexion_2d",
        "active_angles": PIP_KEYS,
        "required_landmarks": [
            "index_mcp", "index_pip", "index_dip",
            "middle_mcp", "middle_pip", "middle_dip",
            "ring_mcp", "ring_pip", "ring_dip",
            "pinky_mcp", "pinky_pip", "pinky_dip",
        ],
        "ranges": {
            "open_range": open_range,
            "closed_range": closed_range,
            "safe_range": safe_range,
        },
        "suggested_config_patch": {
            "tracking_type": "hand21",
            "landmark_schema": "mediapipe_hand_21",
            "evaluation_mode": "open_close",
            "primary_metric": "avg_pip_flexion_2d",
            "open_range": open_range,
            "closed_range": closed_range,
            "min_open_fingers": 4,
            "min_closed_fingers": 3,
            "min_stable_frames": 5,
            "cooldown_frames": 10,
        },
        "trajectory": trajectory,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.exercise_key}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Trayectoria guardada: {out_path}")
    print(f"[OK] Samples validas: {len(samples)} | skipped_no_hand={skipped_no_hand}")
    print(f"[OK] Avg PIP stats: {stats}")
    print(f"[OK] Open range sugerido: {open_range}")
    print(f"[OK] Closed range sugerido: {closed_range}")


if __name__ == "__main__":
    main()
