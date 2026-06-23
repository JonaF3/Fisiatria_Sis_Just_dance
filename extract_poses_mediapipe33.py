
"""
extract_poses_mediapipe33.py

Extractor de trayectoria usando MediaPipe Pose Landmarker con 33 landmarks completos.
NO convierte a MoveNet 17.

Uso recomendado para tu video:
    python extract_poses_mediapipe33.py --video "rehab_references/videos/Shoulder Flexion and Extension.mp4" --song_key "Shoulder Flexion and Extension" --primary_angle right_shoulder_flexion_3d --confirmation_angle right_shoulder_flexion_2d --direction increasing --frame_skip 2

Salida:
    rehab_references/trajectories/<song_key>.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from pose_backends.mediapipe33_backend import MediaPipe33Backend
from rehab_core.angles_33 import compute_angles_33, tracking_quality_for


DEFAULT_ACTIVE_ANGLES = [
    "right_shoulder_flexion_3d",
    "right_shoulder_flexion_2d",
    "right_elbow_flexion_3d",
    "right_elbow_flexion_2d",
    "trunk_lean_2d",
    "shoulder_alignment_2d",
]

DEFAULT_REQUIRED_LANDMARKS = [
    "right_shoulder",
    "right_elbow",
    "right_hip",
]


def moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    out = []
    for idx in range(len(arr)):
        start = max(0, idx - window // 2)
        end = min(len(arr), idx + window // 2 + 1)
        out.append(float(np.mean(arr[start:end])))
    return out


def clamp_range(values: list[float], lo: float = 0.0, hi: float = 180.0) -> list[float]:
    return [round(max(lo, min(hi, float(values[0]))), 2), round(max(lo, min(hi, float(values[1]))), 2)]


def infer_direction(values: list[float]) -> str:
    first = float(values[0])
    mn = float(np.min(values))
    mx = float(np.max(values))
    return "decreasing" if abs(first - mn) > abs(first - mx) else "increasing"


def build_ranges(values: list[float], direction: str, margin: float) -> tuple[list[float], list[float], list[float], dict]:
    arr = np.asarray(values, dtype=np.float32)
    edge = max(3, int(len(arr) * 0.15))

    # Inicio: mediana del inicio y final, porque video es una repetición completa: abajo -> arriba -> abajo.
    start_center = float(np.median(list(arr[:edge]) + list(arr[-edge:])))

    if direction == "increasing":
        target_center = float(np.max(arr))
    else:
        target_center = float(np.min(arr))

    start_range = clamp_range([start_center - margin, start_center + margin])
    target_range = clamp_range([target_center - margin, target_center + margin])
    safe_range = clamp_range([float(np.min(arr)) - margin * 2.0, float(np.max(arr)) + margin * 2.0])

    stats = {
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "start": round(float(arr[0]), 2),
        "end": round(float(arr[-1]), 2),
        "start_center": round(start_center, 2),
        "target_center": round(target_center, 2),
    }
    return start_range, target_range, safe_range, stats


def compact_landmarks(landmarks: dict, required: list[str], optional: list[str]) -> dict:
    """Guarda solo landmarks útiles para no inflar demasiado el JSON."""
    keep = list(dict.fromkeys(required + optional))
    data = {}
    for name in keep:
        point = landmarks.get(name)
        if point is not None:
            data[name] = point.as_dict()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--song_key", required=True)
    parser.add_argument("--primary_angle", default="right_shoulder_flexion_3d")
    parser.add_argument("--confirmation_angle", default="right_shoulder_flexion_2d")
    parser.add_argument("--direction", default="auto", choices=["auto", "increasing", "decreasing"])
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--model", default="model/pose_landmarker_lite.task")
    parser.add_argument("--output", default="rehab_references/trajectories")
    parser.add_argument("--range_margin", type=float, default=10.0)
    parser.add_argument("--min_confidence", type=float, default=0.35)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument(
        "--required_landmarks",
        nargs="+",
        default=None,
        help="Landmarks requeridos para control de calidad. Si no se especifica, "
             "usa el default: right_shoulder right_elbow right_hip",
    )
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

    samples = []
    skipped_no_pose = 0
    skipped_required = 0
    frame_idx = 0

    required_landmarks = args.required_landmarks if args.required_landmarks else DEFAULT_REQUIRED_LANDMARKS
    optional_landmarks = [
        "right_wrist",
        "left_shoulder",
        "left_hip",
        "left_elbow",
        "left_wrist",
    ]

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f} | Frames: {total_frames} | Duración: {duration_s:.2f}s")
    print(f"[INFO] Primary angle: {args.primary_angle}")
    print(f"[INFO] Confirmation angle: {args.confirmation_angle}")

    with MediaPipe33Backend(args.model) as backend:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.frame_skip == 0:
                result = backend.detect_bgr(frame)
                if not result.detected:
                    skipped_no_pose += 1
                else:
                    quality = tracking_quality_for(
                        result.image_landmarks,
                        required_landmarks,
                        min_confidence=args.min_confidence,
                    )
                    if not quality["ok"]:
                        skipped_required += 1
                    else:
                        angles = compute_angles_33(
                            result.image_landmarks,
                            result.world_landmarks,
                            min_confidence=args.min_confidence,
                            include_3d=True,
                        )
                        if args.primary_angle in angles:
                            samples.append({
                                "frame_index": frame_idx,
                                "time_s": round(frame_idx / fps, 4),
                                "progress_raw": round(frame_idx / max(total_frames - 1, 1), 5),
                                "angles": angles,
                                "tracking_quality": quality,
                                "image_landmarks": compact_landmarks(result.image_landmarks, required_landmarks, optional_landmarks),
                                "world_landmarks": compact_landmarks(result.world_landmarks, required_landmarks, optional_landmarks),
                            })

            frame_idx += 1

    cap.release()

    if len(samples) < 5:
        raise SystemExit(
            f"[ERROR] Muy pocas muestras válidas: {len(samples)}. "
            f"Sin pose: {skipped_no_pose}, required incompleto: {skipped_required}"
        )

    primary_values = [float(s["angles"][args.primary_angle]) for s in samples]
    smooth_primary = moving_average(primary_values, window=args.smooth_window)

    for sample, smoothed in zip(samples, smooth_primary):
        sample["angles"][args.primary_angle] = round(float(smoothed), 2)

    direction = args.direction if args.direction != "auto" else infer_direction(smooth_primary)
    start_range, target_range, safe_range, primary_stats = build_ranges(
        smooth_primary,
        direction=direction,
        margin=args.range_margin,
    )

    # Rangos secundarios útiles.
    secondary_ranges = {}
    for angle_name in DEFAULT_ACTIVE_ANGLES:
        vals = [float(s["angles"][angle_name]) for s in samples if angle_name in s["angles"]]
        if len(vals) >= 5:
            secondary_ranges[angle_name] = {
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
                "mean": round(float(np.mean(vals)), 2),
            }

    # Trayectoria compacta por progreso normalizado.
    idxs = np.linspace(0, len(samples) - 1, min(31, len(samples))).astype(int)
    trajectory = []
    for out_idx, src_idx in enumerate(idxs):
        sample = samples[int(src_idx)]
        trajectory.append({
            "progress": round(out_idx / max(len(idxs) - 1, 1), 5),
            "source_frame_index": sample["frame_index"],
            "source_time_s": sample["time_s"],
            "angles": sample["angles"],
            "tracking_quality": sample["tracking_quality"],
            "image_landmarks": sample["image_landmarks"],
            "world_landmarks": sample["world_landmarks"],
        })

    output = {
        "schema_version": "rehab_trajectory_mediapipe33_v1",
        "exercise_id": args.song_key,
        "name": args.song_key,
        "backend": "mediapipe_pose_landmarker",
        "landmark_schema": "mediapipe_pose_33",
        "tracking_type": "pose33",
        "reference_type": "single_rep_trajectory",
        "source_video": str(video_path).replace("\\", "/"),
        "metadata": {
            "fps": round(float(fps), 3),
            "total_frames": total_frames,
            "duration_s": round(float(duration_s), 3),
            "frame_skip": args.frame_skip,
            "valid_samples": len(samples),
            "skipped_no_pose": skipped_no_pose,
            "skipped_required_landmarks": skipped_required,
            "primary_angle_stats": primary_stats,
            "secondary_angle_stats": secondary_ranges,
        },
        "required_landmarks": required_landmarks,
        "optional_landmarks": optional_landmarks,
        "primary_angle": args.primary_angle,
        "confirmation_angle": args.confirmation_angle,
        "active_angles": DEFAULT_ACTIVE_ANGLES,
        "direction": direction,
        "ranges": {
            "start_ranges": {
                args.primary_angle: start_range,
            },
            "target_ranges": {
                args.primary_angle: target_range,
            },
            "safe_ranges": {
                args.primary_angle: safe_range,
            },
        },
        "validation_strategy": "hybrid_3d_primary_2d_confirmation",
        "trajectory": trajectory,
        "suggested_rehab_config_patch": {
            "tracking_type": "pose33",
            "landmark_schema": "mediapipe_pose_33",
            "evaluation_mode": "trajectory",
            "validation_strategy": "hybrid_3d_primary_2d_confirmation",
            "primary_angle": args.primary_angle,
            "confirmation_angle": args.confirmation_angle,
            "active_angles": DEFAULT_ACTIVE_ANGLES,
            "required_landmarks": required_landmarks,
            "start_ranges": {
                args.primary_angle: start_range,
            },
            "target_ranges": {
                args.primary_angle: target_range,
            },
            "safe_ranges": {
                args.primary_angle: safe_range,
            },
        },
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.song_key}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Trayectoria guardada: {out_path}")
    print(f"[OK] Dirección detectada: {direction}")
    print(f"[OK] Primary angle stats: {primary_stats}")
    print(f"[OK] Start range: {start_range}")
    print(f"[OK] Target range: {target_range}")
    print(f"[OK] Safe range: {safe_range}")
    print(f"[OK] Valid samples: {len(samples)} | skipped_no_pose={skipped_no_pose} | skipped_required={skipped_required}")


if __name__ == "__main__":
    main()
