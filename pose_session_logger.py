"""
pose_session_logger.py
Registra datos de estimacion de pose por sesion para analisis posterior.
Requerimiento del estudio CPU vs GPU.
"""
import json
import math
import os
import time
from collections import deque

import numpy as np

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FAST_MOTION_JOINTS = [7, 8, 9, 10]
CONF_THRESHOLD     = 0.25
SPEED_THRESHOLD    = 0.04
OUTPUT_DIR         = "sessions"


class PoseSessionLogger:

    def __init__(self, song_key, difficulty, fps=30.0, condition="CPU"):
        self.song_key   = song_key
        self.difficulty = difficulty
        self.fps        = fps
        self.condition  = condition
        self.start_time = time.time()

        self._kp_history    = []
        self._angle_history = []
        self._kp_window     = deque(maxlen=5)
        self._lost_counts   = {i: 0 for i in range(17)}
        self._total_frames  = 0
        self._fast_errors   = []
        self._stability_acc = {i: [] for i in range(17)}

    def log_frame(self, video_time_s, keypoints_raw, angles=None):
        kp = np.squeeze(np.array(keypoints_raw))
        if kp.ndim != 2 or kp.shape[0] < 17:
            return

        self._total_frames += 1
        kp17 = kp[:17]

        self._kp_history.append({
            "t":  round(float(video_time_s), 4),
            "kp": [[round(float(v), 5) for v in row] for row in kp17.tolist()],
        })

        if angles:
            self._angle_history.append({
                "t":      round(float(video_time_s), 4),
                "angles": {k: round(float(v), 3) for k, v in angles.items()},
            })

        for i in range(17):
            if kp17[i][2] < CONF_THRESHOLD:
                self._lost_counts[i] += 1

        if len(self._kp_window) > 0:
            prev_kp = self._kp_window[-1]
            for i in range(17):
                if kp17[i][2] >= CONF_THRESHOLD and prev_kp[i][2] >= CONF_THRESHOLD:
                    dy = float(kp17[i][0]) - float(prev_kp[i][0])
                    dx = float(kp17[i][1]) - float(prev_kp[i][1])
                    dist = math.sqrt(dx * dx + dy * dy)
                    self._stability_acc[i].append(dist)

        if len(self._kp_window) > 0:
            prev_kp = self._kp_window[-1]
            for i in FAST_MOTION_JOINTS:
                if prev_kp[i][2] >= CONF_THRESHOLD:
                    dy = float(kp17[i][0]) - float(prev_kp[i][0])
                    dx = float(kp17[i][1]) - float(prev_kp[i][1])
                    speed = math.sqrt(dx * dx + dy * dy)
                    if speed > SPEED_THRESHOLD and kp17[i][2] < CONF_THRESHOLD:
                        self._fast_errors.append({
                            "t":     round(float(video_time_s), 4),
                            "joint": JOINT_NAMES[i],
                            "speed": round(speed, 5),
                            "conf":  round(float(kp17[i][2]), 4),
                        })

        self._kp_window.append(kp17.copy())

    def _calc_temporal_stability(self):
        result = {}
        for i in range(17):
            dists = self._stability_acc[i]
            if not dists:
                result[JOINT_NAMES[i]] = {
                    "mean_disp": 0.0, "std_disp": 0.0, "stable": True
                }
                continue
            mean = float(np.mean(dists))
            std  = float(np.std(dists))
            result[JOINT_NAMES[i]] = {
                "mean_disp": round(mean, 6),
                "std_disp":  round(std,  6),
                "stable":    mean < 0.015,
            }
        return result

    def _calc_keypoint_loss(self):
        result = {}
        for i in range(17):
            lost = self._lost_counts[i]
            pct  = (lost / self._total_frames * 100) if self._total_frames > 0 else 0.0
            result[JOINT_NAMES[i]] = {
                "lost_frames": lost,
                "loss_pct":    round(pct, 2),
            }
        return result

    def _calc_angular_variability(self):
        if len(self._angle_history) < 2:
            return {}
        joint_deltas = {}
        for rec in self._angle_history:
            for j, v in rec["angles"].items():
                joint_deltas.setdefault(j, []).append(v)
        result = {}
        for j, vals in joint_deltas.items():
            if len(vals) < 2:
                continue
            deltas = [abs(vals[k+1] - vals[k]) for k in range(len(vals) - 1)]
            result[j] = {
                "mean_delta": round(float(np.mean(deltas)), 3),
                "std_delta":  round(float(np.std(deltas)),  3),
                "max_delta":  round(float(np.max(deltas)),  3),
            }
        return result

    def _calc_trajectory_consistency(self):
        if len(self._kp_history) < 3:
            return {}
        result = {}
        for i in range(17):
            positions = []
            for rec in self._kp_history:
                kp = rec["kp"]
                if kp[i][2] >= CONF_THRESHOLD:
                    positions.append((kp[i][0], kp[i][1]))
            if len(positions) < 3:
                result[JOINT_NAMES[i]] = {
                    "mean_accel": 0.0, "consistency_score": 1.0
                }
                continue
            accel = []
            for k in range(1, len(positions) - 1):
                dy = positions[k+1][0] - 2*positions[k][0] + positions[k-1][0]
                dx = positions[k+1][1] - 2*positions[k][1] + positions[k-1][1]
                accel.append(math.sqrt(dx * dx + dy * dy))
            mean_a = float(np.mean(accel))
            score  = max(0.0, 1.0 - mean_a / 0.05)
            result[JOINT_NAMES[i]] = {
                "mean_accel":        round(mean_a, 6),
                "consistency_score": round(min(1.0, score), 4),
            }
        return result

    def _calc_fast_motion_errors(self):
        if not self._fast_errors:
            return {"total_errors": 0, "per_joint": {}, "events": []}
        per_joint = {}
        for err in self._fast_errors:
            per_joint[err["joint"]] = per_joint.get(err["joint"], 0) + 1
        return {
            "total_errors": len(self._fast_errors),
            "per_joint":    per_joint,
            "events":       self._fast_errors[:50],
        }

    def save(self, final_score=0, difficulty=None, extra_metadata=None):
        if difficulty:
            self.difficulty = difficulty

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts       = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.song_key}_{self.condition}_{self.difficulty}_{ts}.json"
        path     = os.path.join(OUTPUT_DIR, filename)
        duration = time.time() - self.start_time

        stability   = self._calc_temporal_stability()
        kp_loss     = self._calc_keypoint_loss()
        angular_var = self._calc_angular_variability()
        trajectory  = self._calc_trajectory_consistency()
        fast_errors = self._calc_fast_motion_errors()

        stable_joints = sum(1 for v in stability.values() if v["stable"])
        mean_loss_pct = float(np.mean([v["loss_pct"] for v in kp_loss.values()]))
        mean_consistency = float(np.mean([
            v["consistency_score"] for v in trajectory.values()
        ])) if trajectory else 0.0

        session = {
            "song_key":              self.song_key,
            "condition":             self.condition,
            "difficulty":            self.difficulty,
            "timestamp":             ts,
            "duration_s":            round(duration, 2),
            "fps_camera":            self.fps,
            "total_frames_logged":   self._total_frames,
            "final_score":           final_score,
            "summary": {
                "stable_joints_count":         stable_joints,
                "mean_keypoint_loss_pct":       round(mean_loss_pct, 2),
                "mean_trajectory_consistency":  round(mean_consistency, 4),
                "total_fast_motion_errors":     fast_errors["total_errors"],
            },
            "temporal_stability":    stability,
            "keypoint_loss":         kp_loss,
            "angular_variability":   angular_var,
            "trajectory_consistency": trajectory,
            "fast_motion_errors":    fast_errors,
            "keypoints_per_frame":   self._kp_history,
        }

        if extra_metadata:
            session["extra"] = extra_metadata

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session, f, separators=(",", ":"))
            size_kb = os.path.getsize(path) / 1024
            print(f"[SESSION] Guardado: {path} ({size_kb:.0f} KB)")
            return path
        except Exception as e:
            print(f"[SESSION ERROR] No se pudo guardar: {e}")
            return None