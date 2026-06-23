"""Helpers for two-person pose extraction and matching."""

from __future__ import annotations

from itertools import permutations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


PERSON_IDS = ("left", "right")


# Cuánto peso darle a la continuidad de posición X vs. forma de pose.
# 0.0 = comportamiento actual (solo pose shape)
# 0.3 = leve preferencia de continuidad X
# 1.0+ = fuerte ancla espacial (recomendado para evaluacion de rehabilitacion)
X_WEIGHT = 0.6


def _x_jump_cost(new_x: float, last_x: Optional[float], velocity: float = 0.0,
                 max_credible_jump: float = 0.12) -> float:
    """Penaliza asignaciones donde el centro X salta más de lo físicamente creíble respecto a la posición predicha.

    max_credible_jump: fracción del ancho de frame por frame de inferencia.
    """
    if last_x is None:
        return 0.0
    predicted_x = last_x + velocity
    jump = abs(new_x - predicted_x)
    if jump <= max_credible_jump:
        return 0.0
    # Penalización cuadrática: saltos el doble del límite cuestan 4x, no 2x
    return ((jump - max_credible_jump) / max_credible_jump) ** 2



def empty_keypoints() -> np.ndarray:
    return np.zeros((1, 1, 17, 3), dtype=np.float32)


def normalize_keypoints(keypoints) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.shape == (1, 1, 17, 3):
        return arr
    if arr.shape == (17, 3):
        return arr[np.newaxis, np.newaxis, :, :]
    squeezed = np.squeeze(arr)
    if squeezed.shape == (17, 3):
        return squeezed[np.newaxis, np.newaxis, :, :]
    return None


def keypoints_raw(keypoints) -> Optional[List[List[float]]]:
    arr = normalize_keypoints(keypoints)
    if arr is None:
        return None
    return np.squeeze(arr).tolist()


def body_center_x(keypoints, min_conf: float = 0.15) -> Optional[float]:
    arr = normalize_keypoints(keypoints)
    if arr is None:
        return None
    kps = np.squeeze(arr)
    core = [5, 6, 11, 12]
    xs = [float(kps[idx][1]) for idx in core if float(kps[idx][2]) >= min_conf]
    if not xs:
        xs = [float(kp[1]) for kp in kps if float(kp[2]) >= min_conf]
    if not xs:
        return None
    return float(np.mean(xs))


def pose_confidence(keypoints) -> float:
    arr = normalize_keypoints(keypoints)
    if arr is None:
        return 0.0
    return float(np.mean(np.squeeze(arr)[:, 2]))


def _center_pose(kps: np.ndarray, min_conf: float = 0.15) -> Optional[np.ndarray]:
    core = [5, 6, 11, 12]
    valid_yx = [kps[i, :2] for i in core if kps[i, 2] >= min_conf]
    if not valid_yx:
        valid_yx = [kps[i, :2] for i in range(17) if kps[i, 2] >= min_conf]
    if not valid_yx:
        return None
    centroid = np.mean(valid_yx, axis=0)
    centered = kps.copy()
    centered[:, :2] -= centroid
    return centered


def keypoint_distance(kp1: np.ndarray, kp2: np.ndarray,
                      min_conf: float = 0.15) -> float:
    a = np.squeeze(normalize_keypoints(kp1))
    b = np.squeeze(normalize_keypoints(kp2))
    ac = _center_pose(a, min_conf)
    bc = _center_pose(b, min_conf)
    if ac is None or bc is None:
        return float("inf")
    dists = []
    for i in range(17):
        if a[i][2] >= min_conf and b[i][2] >= min_conf:
            d = float(np.linalg.norm(ac[i, :2] - bc[i, :2]))
            dists.append(d)
    if not dists:
        return float("inf")
    return float(np.mean(dists))


class DuoPoseTracker:
    """Asigna poses detectadas a slots P1 (left) y P2 (right).

    La identidad se fija en el PRIMER frame donde se detectan 2 personas
    (izquierda → left/P1, derecha → right/P2) y NUNCA cambia, sin importar
    si los pacientes se cruzan o intercambian posiciones.

    En frames siguientes, los candidatos se re-asignan al slot mas cercano
    en forma de pose respecto al frame anterior — pero el slot en si es fijo.
    """

    def __init__(self, carry_frames: int = 12):
        self.carry_frames = max(0, int(carry_frames))
        self._last: Dict[str, Optional[np.ndarray]] = {pid: None for pid in PERSON_IDS}
        self._last_x: Dict[str, Optional[float]] = {pid: None for pid in PERSON_IDS}
        self._velocity_x: Dict[str, float] = {pid: 0.0 for pid in PERSON_IDS}
        self._missing: Dict[str, int] = {pid: self.carry_frames + 1 for pid in PERSON_IDS}
        # True una vez que se asignaron los slots iniciales
        self._identity_locked: bool = False

    def assign(self, candidates: Iterable[np.ndarray]) -> Dict[str, dict]:
        infos = []
        for candidate in candidates or []:
            keypoints = normalize_keypoints(candidate)
            center = body_center_x(keypoints)
            if keypoints is None or center is None:
                continue
            infos.append({
                "keypoints": keypoints,
                "center_x": center,
                "confidence": pose_confidence(keypoints),
            })

        infos.sort(key=lambda item: item["confidence"], reverse=True)
        infos = infos[:2]

        assigned: Dict[str, dict] = {}
        if len(infos) >= 2:
            assigned = self._assign_two(infos)
        elif len(infos) == 1:
            assigned = self._assign_one(infos[0])

        result: Dict[str, dict] = {}
        for pid in PERSON_IDS:
            if pid in assigned:
                pose = assigned[pid]
                pose.update({"visible": True, "carried": False})
                result[pid] = pose

                # Calcular velocidad antes de actualizar _last_x
                if self._last_x[pid] is not None and pose["center_x"] is not None:
                    vel = pose["center_x"] - self._last_x[pid]
                    # Filtro de suavizado (paso bajo) para la velocidad
                    self._velocity_x[pid] = 0.6 * self._velocity_x[pid] + 0.4 * vel
                else:
                    self._velocity_x[pid] = 0.0

                self._last[pid] = pose["keypoints"]
                self._last_x[pid] = pose["center_x"]
                self._missing[pid] = 0
            else:
                result[pid] = self._carry(pid)
                # Si no está visible, amortiguar la velocidad
                self._velocity_x[pid] *= 0.5
        return result

    def _assign_two(self, infos: List[dict]) -> Dict[str, dict]:
        # ── Primera vez: fijar identidad por posicion X ───────────────
        if not self._identity_locked:
            ordered = sorted(infos, key=lambda item: item["center_x"])
            self._identity_locked = True
            print("[DUO] Identidades fijadas: "
                  f"P1(left) x={ordered[0]['center_x']:.2f} | "
                  f"P2(right) x={ordered[1]['center_x']:.2f}")
            return {"left": ordered[0].copy(), "right": ordered[1].copy()}

        # ── Frames siguientes: asignar cada candidato al slot mas
        #    parecido en forma de pose + posicion X continua — la identidad ya esta bloqueada ─
        best_order = None
        best_cost = float("inf")
        for order in permutations(range(2)):
            # Costo de forma (ya existía)
            shape_cost = sum(
                keypoint_distance(infos[idx]["keypoints"], self._last[pid])
                if self._last[pid] is not None else 0.0
                for pid, idx in zip(PERSON_IDS, order)
            )
            # Costo de posición X (nuevo): penaliza saltos bruscos respecto a predicción
            x_cost = sum(
                _x_jump_cost(infos[idx]["center_x"], self._last_x[pid], self._velocity_x[pid])
                for pid, idx in zip(PERSON_IDS, order)
            )
            cost = shape_cost + X_WEIGHT * x_cost
            if cost < best_cost:
                best_cost = cost
                best_order = order

        return {pid: infos[idx].copy() for pid, idx in zip(PERSON_IDS, best_order)}

    def _assign_one(self, info: dict) -> Dict[str, dict]:
        known_kp = {pid: kp for pid, kp in self._last.items() if kp is not None}
        if known_kp:
            def combined_cost(key):
                shape = keypoint_distance(info["keypoints"], known_kp[key])
                x = _x_jump_cost(info["center_x"], self._last_x[key], self._velocity_x[key])
                return shape + X_WEIGHT * x
            pid = min(known_kp, key=combined_cost)
        else:
            known_x = {pid: x for pid, x in self._last_x.items() if x is not None}
            if known_x:
                pid = min(known_x, key=lambda key: abs(info["center_x"] - (known_x[key] + self._velocity_x[key])))
            else:
                pid = "left" if info["center_x"] <= 0.5 else "right"
        return {pid: info.copy()}

    def _carry(self, pid: str) -> dict:
        self._missing[pid] += 1
        if self._last[pid] is not None and self._missing[pid] <= self.carry_frames:
            return {
                "keypoints": self._last[pid],
                "center_x": self._last_x[pid],
                "confidence": pose_confidence(self._last[pid]),
                "visible": True,
                "carried": True,
            }
        return {
            "keypoints": None,
            "center_x": self._last_x[pid],
            "confidence": 0.0,
            "visible": False,
            "carried": False,
        }
