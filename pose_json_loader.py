"""
pose_json_loader.py — Módulo auxiliar para just_dance_controller.py
====================================================================
Carga y consulta el JSON de poses pre-computadas.

Uso en just_dance_controller.py:
    from pose_json_loader import PoseJsonLoader
    self._pose_loader = PoseJsonLoader(song_key)  # en __init__
    data = self._pose_loader.get_at(video_time)   # en el loop
"""

import bisect
import json
import os
from typing import Optional
import numpy as np

from duo_pose_utils import PERSON_IDS, normalize_keypoints


# Ángulos neutros de fallback (persona parada quieta)
NEUTRAL_ANGLES = {
    "left_arm":    170.0, "right_arm":    170.0,
    "left_elbow":  170.0, "right_elbow":  170.0,
    "left_trunk":  175.0, "right_trunk":  175.0,
    "left_knee":   175.0, "right_knee":   175.0,
    "left_ankle":  175.0, "right_ankle":  175.0,
}

ANGLE_ALIASES = {
    "left_thigh": "left_knee",
    "right_thigh": "right_knee",
    "left_leg": "left_ankle",
    "right_leg": "right_ankle",
}

# Keypoints neutros: 17 puntos todos en el centro con score 0
NEUTRAL_KPS = np.zeros((1, 1, 17, 3), dtype=np.float32)


class PoseJsonLoader:
    """
    Carga songs_poses/<song_key>.json y permite búsqueda O(log n)
    del frame más cercano a un timestamp dado.

    Si el JSON no existe o falla al cargar, todas las consultas
    devuelven datos neutros → retrocompatible con el flujo actual.
    """

    def __init__(self, song_key: str, poses_dir: str = "songs_poses"):
        self.song_key   = song_key
        self.available  = False
        self.players    = 1
        self.multi_person = False
        self._timestamps = []   # lista ordenada de timestamps_ms
        self._frames     = []   # lista paralela de dicts

        if not song_key:
            return

        json_path = os.path.join(poses_dir, f"{song_key}.json")
        if not os.path.exists(json_path):
            print(f"[INFO] Sin JSON de poses para '{song_key}'. Modo inferencia en vivo.")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            self.players = int(data.get("players", metadata.get("players", 1)) or 1)
            frames = data.get("frames", [])
            if not frames:
                print(f"[WARN] JSON de poses vacío: {json_path}")
                return

            self._timestamps = [fr["timestamp_ms"] for fr in frames]
            self._frames     = frames
            self.multi_person = any("persons" in fr for fr in frames)
            self.available   = True
            print(f"[INFO] Poses pre-computadas cargadas: {len(frames)} frames ({json_path})")

        except Exception as e:
            print(f"[WARN] Error cargando JSON de poses ({json_path}): {e}. Usando inferencia en vivo.")

    def get_at(self, video_time_seconds: float, person_id: Optional[str] = None) -> dict:
        """
        Devuelve el frame más cercano al timestamp dado.

        Args:
            video_time_seconds: tiempo actual del video en segundos

        Returns:
            dict con claves:
              'angles'       → dict con los 8 ángulos
              'keypoints'    → np.ndarray (1,1,17,3) listo para draw_skeleton / vectores
              'found'        → bool (True si vino del JSON, False si es neutro)
        """
        if not self.available:
            return self._neutral()

        target_ms = video_time_seconds * 1000.0

        # Bisect sobre lista ordenada → O(log n)
        idx = bisect.bisect_left(self._timestamps, target_ms)

        # Clampear al rango válido
        if idx >= len(self._timestamps):
            idx = len(self._timestamps) - 1
        elif idx > 0:
            # Elegir el más cercano entre idx-1 e idx
            if abs(self._timestamps[idx-1] - target_ms) < abs(self._timestamps[idx] - target_ms):
                idx -= 1

        frame = self._frames[idx]
        if "persons" in frame:
            return self._person_from_frame(frame, person_id or "left")

        angles = self._normalize_angles(frame.get("angles", {}))

        # Rellenar ángulos faltantes con neutros (EC-5)
        for k, v in NEUTRAL_ANGLES.items():
            if k not in angles:
                angles[k] = v

        # Reconstruir keypoints como ndarray (1,1,17,3)
        kps_raw = frame.get("keypoints_raw", None)
        if kps_raw is not None:
            keypoints = normalize_keypoints(kps_raw)
            if keypoints is None:
                keypoints = NEUTRAL_KPS.copy()
        else:
            keypoints = NEUTRAL_KPS.copy()

        return {"angles": angles, "keypoints": keypoints, "found": True}

    def get_duo_at(self, video_time_seconds: float) -> dict:
        if not self.available:
            return {pid: self._neutral(person_id=pid) for pid in PERSON_IDS}

        target_ms = video_time_seconds * 1000.0
        idx = bisect.bisect_left(self._timestamps, target_ms)
        if idx >= len(self._timestamps):
            idx = len(self._timestamps) - 1
        elif idx > 0:
            if abs(self._timestamps[idx-1] - target_ms) < abs(self._timestamps[idx] - target_ms):
                idx -= 1

        frame = self._frames[idx]
        if "persons" not in frame:
            single = self.get_at(video_time_seconds)
            return {
                "left": single,
                "right": self._neutral(person_id="right"),
            }
        return {pid: self._person_from_frame(frame, pid) for pid in PERSON_IDS}

    def _person_from_frame(self, frame: dict, person_id: str) -> dict:
        persons = frame.get("persons", [])
        selected = None
        for person in persons:
            if person.get("id") == person_id:
                selected = person
                break
        if selected is None:
            return self._neutral(person_id=person_id)

        angles = self._normalize_angles(selected.get("angles") or {})
        for k, v in NEUTRAL_ANGLES.items():
            if k not in angles:
                angles[k] = v

        keypoints = normalize_keypoints(selected.get("keypoints_raw"))
        if keypoints is None:
            keypoints = NEUTRAL_KPS.copy()

        return {
            "angles": angles,
            "keypoints": keypoints,
            "found": True,
            "person_id": selected.get("id", person_id),
            "visible": bool(selected.get("visible", True)),
            "carried": bool(selected.get("carried", False)),
        }

    def _neutral(self, person_id: Optional[str] = None) -> dict:
        return {
            "angles": NEUTRAL_ANGLES.copy(),
            "keypoints": NEUTRAL_KPS.copy(),
            "found": False,
            "person_id": person_id,
            "visible": False,
            "carried": False,
        }

    def _normalize_angles(self, angles: dict) -> dict:
        normalized = dict(angles or {})
        for old_key, new_key in ANGLE_ALIASES.items():
            if new_key not in normalized and old_key in normalized:
                normalized[new_key] = normalized[old_key]
        return normalized
