"""
rehab_core/angles_33.py

Cálculo de ángulos y métricas posturales usando landmarks completos de
MediaPipe Pose 33. Este módulo NO convierte a formato MoveNet 17.

Entrada esperada:
    image_landmarks: dict[str, Landmark33]
    world_landmarks: dict[str, Landmark33] opcional

Compatible con pose_backends.mediapipe33_backend.Landmark33.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Iterable

import numpy as np


# -----------------------------------------------------------------------------
# Utilidades base
# -----------------------------------------------------------------------------


def _confidence(point) -> float:
    if point is None:
        return 0.0
    if hasattr(point, "confidence"):
        return float(point.confidence)
    visibility = float(getattr(point, "visibility", 1.0))
    presence = float(getattr(point, "presence", visibility))
    return max(0.0, min(1.0, visibility * presence))


def is_visible(landmarks: dict, name: str, min_confidence: float = 0.35) -> bool:
    """True si el landmark existe y supera la confianza mínima."""
    point = landmarks.get(name)
    return point is not None and _confidence(point) >= min_confidence


def visible_landmarks(landmarks: dict, names: Iterable[str], min_confidence: float = 0.35) -> bool:
    """True si todos los landmarks requeridos son visibles."""
    return all(is_visible(landmarks, name, min_confidence) for name in names)


def _vec2(point_a, point_b) -> np.ndarray:
    return np.array(
        [
            float(point_a.x) - float(point_b.x),
            float(point_a.y) - float(point_b.y),
        ],
        dtype=np.float32,
    )


def _vec3(point_a, point_b) -> np.ndarray:
    return np.array(
        [
            float(point_a.x) - float(point_b.x),
            float(point_a.y) - float(point_b.y),
            float(point_a.z) - float(point_b.z),
        ],
        dtype=np.float32,
    )


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if den < 1e-6:
        return None
    cosang = float(np.dot(v1, v2) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return round(float(math.degrees(math.acos(cosang))), 2)


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    """Normaliza un vector 3D. Retorna None si la norma es casi cero."""
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return v / n


def angle_2d(landmarks: dict, start: str, middle: str, end: str, min_confidence: float = 0.35) -> Optional[float]:
    """
    Ángulo 2D en `middle` usando coordenadas normalizadas x/y.

    Ejemplo hombro derecho:
        angle_2d(points, "right_elbow", "right_shoulder", "right_hip")
    """
    if not visible_landmarks(landmarks, [start, middle, end], min_confidence):
        return None
    p_start = landmarks[start]
    p_middle = landmarks[middle]
    p_end = landmarks[end]
    return _angle_between(_vec2(p_start, p_middle), _vec2(p_end, p_middle))


def angle_3d(landmarks: dict, start: str, middle: str, end: str, min_confidence: float = 0.35) -> Optional[float]:
    """
    Ángulo 3D en `middle` usando coordenadas x/y/z.

    Puede usarse con image_landmarks o world_landmarks. Para análisis clínico
    de compensación, world_landmarks suele ser más útil si está disponible.
    """
    if not visible_landmarks(landmarks, [start, middle, end], min_confidence):
        return None
    p_start = landmarks[start]
    p_middle = landmarks[middle]
    p_end = landmarks[end]
    return _angle_between(_vec3(p_start, p_middle), _vec3(p_end, p_middle))


def _weighted_point(
    image_landmarks: dict,
    world_landmarks: Optional[dict],
    name: str,
    min_confidence: float = 0.35,
) -> Optional[np.ndarray]:
    """
    Combina image_landmarks (x,y estables) con world_landmarks (z confiable)
    ponderado por visibility. Si world_landmarks tiene buena confianza usa
    su z; si no, usa z de image_landmarks como fallback.
    """
    img_pt = image_landmarks.get(name)
    if img_pt is None or _confidence(img_pt) < min_confidence:
        return None

    x = float(img_pt.x)
    y = float(img_pt.y)

    wld_pt = world_landmarks.get(name) if world_landmarks else None
    if wld_pt is not None and _confidence(wld_pt) >= min_confidence:
        z = float(wld_pt.z)
    else:
        z = float(getattr(img_pt, "z", 0.0))

    return np.array([x, y, z], dtype=np.float32)


def angle_3d_weighted(
    image_landmarks: dict,
    world_landmarks: Optional[dict],
    start: str,
    middle: str,
    end: str,
    min_confidence: float = 0.35,
) -> Optional[float]:
    """
    Ángulo 3D combinando image_landmarks (x,y) y world_landmarks (z).
    Usa world_landmarks para profundidad cuando están disponibles,
    image_landmarks para x,y que son más estables en píxeles.
    """
    p_start = _weighted_point(image_landmarks, world_landmarks, start, min_confidence)
    p_mid = _weighted_point(image_landmarks, world_landmarks, middle, min_confidence)
    p_end = _weighted_point(image_landmarks, world_landmarks, end, min_confidence)

    if p_start is None or p_mid is None or p_end is None:
        return None

    return _angle_between(p_start - p_mid, p_end - p_mid)


# -----------------------------------------------------------------------------
# Ángulos principales MediaPipe 33
# -----------------------------------------------------------------------------


ANGLE_DEFINITIONS_2D = {
    # Hombro: codo -> hombro -> cadera
    "right_shoulder_flexion_2d": ("right_elbow", "right_shoulder", "right_hip"),
    "left_shoulder_flexion_2d": ("left_elbow", "left_shoulder", "left_hip"),

    # Codo: hombro -> codo -> muñeca
    "right_elbow_flexion_2d": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_elbow_flexion_2d": ("left_shoulder", "left_elbow", "left_wrist"),

    # Cadera: hombro -> cadera -> rodilla
    "right_hip_flexion_2d": ("right_shoulder", "right_hip", "right_knee"),
    "left_hip_flexion_2d": ("left_shoulder", "left_hip", "left_knee"),

    # Rodilla: cadera -> rodilla -> tobillo
    "right_knee_flexion_2d": ("right_hip", "right_knee", "right_ankle"),
    "left_knee_flexion_2d": ("left_hip", "left_knee", "left_ankle"),

    # Tobillo aproximado: rodilla -> tobillo -> punta del pie
    "right_ankle_angle_2d": ("right_knee", "right_ankle", "right_foot_index"),
    "left_ankle_angle_2d": ("left_knee", "left_ankle", "left_foot_index"),

    # Mano/muñeca aproximada desde Pose 33, no tan precisa como Hand 21
    "right_wrist_pose_angle_2d": ("right_elbow", "right_wrist", "right_index"),
    "left_wrist_pose_angle_2d": ("left_elbow", "left_wrist", "left_index"),

    # Cuello (aproximado). Para clínica completa usar landmarks craneales.
    # neck_sagittal: inclinación adelante/atrás (cuello atras = cabeza atrás)
    # neck_lateral : inclinación lateral (cuello izq = oreja izq hacia hombro izq)
    "neck_sagittal_2d": ("nose", "right_shoulder", "right_hip"),
    "left_neck_lateral_2d": ("left_ear", "left_shoulder", "left_hip"),
    "right_neck_lateral_2d": ("right_ear", "right_shoulder", "right_hip"),
}

# Para 3D usamos las mismas definiciones, cambiando sufijo.
ANGLE_DEFINITIONS_3D = {
    name.replace("_2d", "_3d"): points
    for name, points in ANGLE_DEFINITIONS_2D.items()
}


# -----------------------------------------------------------------------------
# Ángulos cuerpo-relativos — invariantes al ángulo de cámara
# -----------------------------------------------------------------------------
#
# El sistema de referencia del cuerpo se construye con los 4 landmarks del tronco
# (hombros y caderas) usando world_landmarks de MediaPipe (espacio 3D métrico con
# origen en las caderas). Los ángulos calculados aquí son independientes de la
# posición o ángulo de la cámara: si el paciente levanta el brazo 90°, el resultado
# es ~90° independientemente de si la cámara está de frente, de lado o en diagonal.
#
# Claves generadas (side = "right" | "left"):
#   {side}_shoulder_flexion_body_rel   — elevación en plano sagital (frente/atrás)
#   {side}_shoulder_abduction_body_rel — elevación en plano frontal (lateral)
#   {side}_hip_flexion_body_rel        — flexión de cadera en plano sagital
#   trunk_lean_body_rel                — inclinación real del tronco vs la gravedad


def compute_body_frame(world_landmarks: dict, min_confidence: float = 0.35) -> Optional[dict]:
    """
    Construye el sistema de referencia local del cuerpo a partir de world_landmarks.

    Ejes del frame:
        up      — vector unitario de caderas a hombros (craneal)
        lateral — vector unitario de hombro izquierdo a derecho (ortogonalizado vs up)
        forward — vector unitario hacia el frente del cuerpo (producto cruzado)

    Usa _robust_midpoint_3d para tolerar visibilidad parcial (ej. vista de perfil).
    """
    mid_hip = _robust_midpoint_3d(world_landmarks, "left_hip", "right_hip", min_confidence)
    mid_shoulder = _robust_midpoint_3d(world_landmarks, "left_shoulder", "right_shoulder", min_confidence)
    if mid_hip is None or mid_shoulder is None:
        return None

    up = _normalize(mid_shoulder - mid_hip)
    if up is None:
        return None

    ls = world_landmarks.get("left_shoulder")
    rs = world_landmarks.get("right_shoulder")
    ls_ok = ls is not None and _confidence(ls) >= min_confidence
    rs_ok = rs is not None and _confidence(rs) >= min_confidence

    if ls_ok and rs_ok:
        lateral_raw = np.array([rs.x - ls.x, rs.y - ls.y, rs.z - ls.z], dtype=np.float64)
        lateral = _normalize(lateral_raw - np.dot(lateral_raw, up) * up)
        if lateral is None:
            return None
    else:
        lateral_raw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = _normalize(lateral_raw - np.dot(lateral_raw, up) * up)
        if lateral is None:
            lateral = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = _normalize(np.cross(lateral, up))
    if forward is None:
        return None

    return {
        "up": up,
        "lateral": lateral,
        "forward": forward,
        "mid_hip": mid_hip,
        "mid_shoulder": mid_shoulder,
    }


def _limb_angle_in_plane(
    world_landmarks: dict,
    proximal: str,
    distal: str,
    plane_normal: np.ndarray,
    body_down: np.ndarray,
    min_confidence: float = 0.35,
) -> Optional[float]:
    """
    Proyecta el vector proximal→distal sobre un plano corporal y devuelve el
    ángulo respecto a 'body_down'.

        0°   = segmento alineado con el cuerpo hacia abajo (posición de reposo)
        90°  = segmento horizontal
        180° = segmento apuntando hacia arriba (elevación completa)
    """
    if not visible_landmarks(world_landmarks, [proximal, distal], min_confidence):
        return None

    p = world_landmarks[proximal]
    d = world_landmarks[distal]
    segment = np.array([d.x - p.x, d.y - p.y, d.z - p.z], dtype=np.float64)

    # Proyectar sobre el plano eliminando la componente normal al plano
    seg_proj = segment - np.dot(segment, plane_normal) * plane_normal
    seg_proj_norm = _normalize(seg_proj)
    if seg_proj_norm is None:
        return None

    return _angle_between(seg_proj_norm.astype(np.float32), body_down.astype(np.float32))


def shoulder_flexion_body_rel(
    world_landmarks: dict, side: str = "right", min_confidence: float = 0.35
) -> Optional[float]:
    """
    Flexión de hombro en el plano sagital relativo al cuerpo.

    Mide la elevación del brazo hacia adelante/atrás, independientemente
    del ángulo de cámara.

        0°   ≈ brazo colgando junto al cuerpo
        90°  ≈ brazo horizontal hacia adelante
        180° ≈ brazo elevado verticalmente
    """
    frame = compute_body_frame(world_landmarks, min_confidence)
    if frame is None:
        return None
    # Plano sagital: su normal es el eje lateral del cuerpo
    return _limb_angle_in_plane(
        world_landmarks,
        proximal=f"{side}_shoulder",
        distal=f"{side}_elbow",
        plane_normal=frame["lateral"],
        body_down=-frame["up"],
        min_confidence=min_confidence,
    )


def shoulder_abduction_body_rel(
    world_landmarks: dict, side: str = "right", min_confidence: float = 0.35
) -> Optional[float]:
    """
    Abducción de hombro en el plano frontal relativo al cuerpo.

    Mide la elevación lateral del brazo.

        0°   ≈ brazo junto al cuerpo
        90°  ≈ brazo horizontal hacia el lado
        180° ≈ brazo elevado verticalmente
    """
    frame = compute_body_frame(world_landmarks, min_confidence)
    if frame is None:
        return None
    # Plano frontal: su normal es el eje forward del cuerpo
    return _limb_angle_in_plane(
        world_landmarks,
        proximal=f"{side}_shoulder",
        distal=f"{side}_elbow",
        plane_normal=frame["forward"],
        body_down=-frame["up"],
        min_confidence=min_confidence,
    )


def hip_flexion_body_rel(
    world_landmarks: dict, side: str = "right", min_confidence: float = 0.35
) -> Optional[float]:
    """
    Flexión de cadera en el plano sagital relativo al cuerpo.

        0°   ≈ pierna alineada con el tronco
        90°  ≈ muslo horizontal
        180° ≈ pierna elevada verticalmente
    """
    frame = compute_body_frame(world_landmarks, min_confidence)
    if frame is None:
        return None
    return _limb_angle_in_plane(
        world_landmarks,
        proximal=f"{side}_hip",
        distal=f"{side}_knee",
        plane_normal=frame["lateral"],
        body_down=-frame["up"],
        min_confidence=min_confidence,
    )


def trunk_lean_body_rel(
    world_landmarks: dict, min_confidence: float = 0.35
) -> Optional[float]:
    """
    Inclinación real del tronco respecto a la gravedad usando world_landmarks.

    En el espacio de world_landmarks de MediaPipe el eje Y apunta hacia abajo
    (positivo = hacia el suelo), por lo que la dirección 'arriba real' es (0,-1,0).

        0°  = tronco perfectamente vertical
        >0° = grado de inclinación (lateral o anteroposterior)
    """
    frame = compute_body_frame(world_landmarks, min_confidence)
    if frame is None:
        return None

    # En world_landmarks de MediaPipe: Y negativo = hacia arriba (contra la gravedad)
    gravity_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _angle_between(frame["up"].astype(np.float32), gravity_up.astype(np.float32))


def hip_abduction_2d(
    landmarks: dict,
    side: str = "right",
    min_confidence: float = 0.35,
) -> Optional[float]:
    """
    Ángulo de abducción de cadera en el plano de la imagen.
    Mide cuánto se desplaza lateralmente la pierna desde la línea media del cuerpo.

    Usa el punto medio hombros→caderas como eje vertical de referencia,
    y mide el ángulo del vector cadera→rodilla respecto a ese eje.

    0°  = pierna alineada verticalmente (abajo)
    >0° = pierna desplazada lateralmente (abducción visible)
    """
    shoulder_mid = _robust_point(landmarks, "left_shoulder", "right_shoulder", min_confidence)
    hip_mid = _robust_point(landmarks, "left_hip", "right_hip", min_confidence)
    if shoulder_mid is None or hip_mid is None:
        return None

    body_down = hip_mid - shoulder_mid
    norm_down = float(np.linalg.norm(body_down))
    if norm_down < 1e-6:
        return None

    hip = landmarks.get(f"{side}_hip")
    knee = landmarks.get(f"{side}_knee")
    if hip is None or knee is None:
        return None
    if _confidence(hip) < min_confidence or _confidence(knee) < min_confidence:
        return None

    thigh = np.array([knee.x - hip.x, knee.y - hip.y], dtype=np.float32)
    return _angle_between(body_down, thigh)


def hip_abduction_full_2d(
    landmarks: dict,
    side: str = "right",
    min_confidence: float = 0.35,
) -> Optional[float]:
    shoulder_mid = _robust_point(landmarks, "left_shoulder", "right_shoulder", min_confidence)
    hip_mid = _robust_point(landmarks, "left_hip", "right_hip", min_confidence)
    if shoulder_mid is None or hip_mid is None:
        return None

    body_down = hip_mid - shoulder_mid
    norm_down = float(np.linalg.norm(body_down))
    if norm_down < 1e-6:
        return None

    hip = landmarks.get(f"{side}_hip")
    foot = landmarks.get(f"{side}_foot_index") or landmarks.get(f"{side}_ankle")
    if hip is None or foot is None:
        return None
    if _confidence(hip) < min_confidence or _confidence(foot) < min_confidence:
        return None

    leg = np.array([foot.x - hip.x, foot.y - hip.y], dtype=np.float32)
    return _angle_between(body_down, leg)


def hip_abduction_weighted_3d(
    image_landmarks: dict,
    world_landmarks: Optional[dict],
    side: str = "right",
    min_confidence: float = 0.35,
) -> Optional[float]:
    """
    Abducción de cadera 3D combinando image_landmarks (x,y) y world_landmarks (z).
    Proyecta el muslo sobre el plano frontal del cuerpo.

    0°  = pierna alineada con el cuerpo
    >0° = pierna abducida (separada lateralmente)
    """
    hip = _weighted_point(image_landmarks, world_landmarks, f"{side}_hip", min_confidence)
    knee = _weighted_point(image_landmarks, world_landmarks, f"{side}_knee", min_confidence)
    rsh = _weighted_point(image_landmarks, world_landmarks, "right_shoulder", min_confidence)
    lsh = _weighted_point(image_landmarks, world_landmarks, "left_shoulder", min_confidence)
    rhip = _weighted_point(image_landmarks, world_landmarks, "right_hip", min_confidence)
    lhip = _weighted_point(image_landmarks, world_landmarks, "left_hip", min_confidence)

    if any(p is None for p in [hip, knee, rsh, lsh, rhip, lhip]):
        return None

    mid_shoulder = (rsh + lsh) / 2.0
    mid_hip = (rhip + lhip) / 2.0
    body_down = mid_hip - mid_shoulder
    norm_down = float(np.linalg.norm(body_down))
    if norm_down < 1e-6:
        return None

    thigh = knee - hip
    return _angle_between(body_down.astype(np.float32), thigh.astype(np.float32))


def compute_angles_33(
    image_landmarks: dict,
    world_landmarks: Optional[dict] = None,
    min_confidence: float = 0.35,
    include_3d: bool = True,
) -> dict:
    """
    Calcula ángulos principales desde MediaPipe 33.

    Devuelve un dict como:
        {
            "right_shoulder_flexion_2d": 18.2,
            "right_elbow_flexion_2d": 171.4,
            "trunk_lean_2d": 5.8,
            "right_shoulder_flexion_body_rel": 22.4,  # invariante al ángulo de cámara
            ...
        }
    """
    angles = {}

    for angle_name, (start, middle, end) in ANGLE_DEFINITIONS_2D.items():
        value = angle_2d(image_landmarks, start, middle, end, min_confidence)
        if value is not None:
            angles[angle_name] = value

    if include_3d:
        source_3d = world_landmarks if world_landmarks else image_landmarks
        for angle_name, (start, middle, end) in ANGLE_DEFINITIONS_3D.items():
            value = angle_3d(source_3d, start, middle, end, min_confidence)
            if value is not None:
                angles[angle_name] = value

        # Ángulos cuerpo-relativos — invariantes al ángulo de cámara.
        # Usan world_landmarks (espacio 3D métrico de MediaPipe) cuando están disponibles.
        wl = world_landmarks if world_landmarks else image_landmarks
        for side in ("right", "left"):
            v = shoulder_flexion_body_rel(wl, side, min_confidence)
            if v is not None:
                angles[f"{side}_shoulder_flexion_body_rel"] = v

            v = shoulder_abduction_body_rel(wl, side, min_confidence)
            if v is not None:
                angles[f"{side}_shoulder_abduction_body_rel"] = v

            v = hip_flexion_body_rel(wl, side, min_confidence)
            if v is not None:
                angles[f"{side}_hip_flexion_body_rel"] = v

        v = trunk_lean_body_rel(wl, min_confidence)
        if v is not None:
            angles["trunk_lean_body_rel"] = v

    # Métricas posturales adicionales
    trunk = trunk_lean_2d(image_landmarks, min_confidence)
    if trunk is not None:
        angles["trunk_lean_2d"] = trunk

    shoulder = shoulder_alignment_2d(image_landmarks, min_confidence)
    if shoulder is not None:
        angles["shoulder_alignment_2d"] = shoulder

    sw = shoulder_width_2d(image_landmarks, min_confidence)
    if sw is not None:
        angles["shoulder_width_2d"] = sw

    hip = hip_alignment_2d(image_landmarks, min_confidence)
    if hip is not None:
        angles["hip_alignment_2d"] = hip

    # Asimetría de hombro para rotación de tronco.
    # Usa _robust_point para caderas → tolera que una cadera tenga baja visibilidad.
    _r_sha = shoulder_asymmetry_signed_2d(image_landmarks, "right", min_confidence)
    if _r_sha is not None:
        angles["right_shoulder_asymmetry_2d"] = _r_sha
    _l_sha = shoulder_asymmetry_signed_2d(image_landmarks, "left", min_confidence)
    if _l_sha is not None:
        angles["left_shoulder_asymmetry_2d"] = _l_sha

    # Giro relativo de hombros (diferencia de profundidad z entre hombros).
    # Captura rotación del tronco incluso con visibilidad parcial.
    _twist = shoulder_twist_signed_3d(image_landmarks, world_landmarks, min_confidence)
    if _twist is not None:
        angles["shoulder_twist_signed_3d"] = _twist

    # Angulos de cuello robustos (vertice en base del cuello, relativos al tronco)
    neck_dev = neck_deviation_2d(image_landmarks, min_confidence)
    if neck_dev is not None:
        angles["neck_deviation_2d"] = neck_dev

    neck_lat = neck_lateral_signed_2d(image_landmarks, min_confidence)
    if neck_lat is not None:
        angles["neck_lateral_signed_2d"] = neck_lat

    neck_ext = neck_extension_side_2d(image_landmarks, min_confidence)
    if neck_ext is not None:
        angles["neck_extension_side_2d"] = neck_ext

    head_pitch = head_pitch_side_2d(image_landmarks, min_confidence)
    if head_pitch is not None:
        angles["head_pitch_side_2d"] = head_pitch

    head_lat = head_lateral_signed_2d(image_landmarks, min_confidence)
    if head_lat is not None:
        angles["head_lateral_signed_2d"] = head_lat

    # Métricas 3D ponderadas para pierna — combinan x/y de image_landmarks
    # con z de world_landmarks para capturar movimientos en profundidad.
    _hip_flex_w = angle_3d_weighted(
        image_landmarks, world_landmarks,
        "right_shoulder", "right_hip", "right_knee",
        min_confidence,
    )
    if _hip_flex_w is not None:
        angles["right_hip_flexion_weighted_3d"] = round(float(_hip_flex_w), 2)

    _hip_flex_lw = angle_3d_weighted(
        image_landmarks, world_landmarks,
        "left_shoulder", "left_hip", "left_knee",
        min_confidence,
    )
    if _hip_flex_lw is not None:
        angles["left_hip_flexion_weighted_3d"] = round(float(_hip_flex_lw), 2)

    _knee_flex_w = angle_3d_weighted(
        image_landmarks, world_landmarks,
        "right_hip", "right_knee", "right_ankle",
        min_confidence,
    )
    if _knee_flex_w is not None:
        angles["right_knee_flexion_weighted_3d"] = round(float(_knee_flex_w), 2)

    _knee_flex_lw = angle_3d_weighted(
        image_landmarks, world_landmarks,
        "left_hip", "left_knee", "left_ankle",
        min_confidence,
    )
    if _knee_flex_lw is not None:
        angles["left_knee_flexion_weighted_3d"] = round(float(_knee_flex_lw), 2)

    # Abducción de cadera 2D — para movimientos laterales de pierna.
    _r_hip_abd = hip_abduction_2d(image_landmarks, "right", min_confidence)
    if _r_hip_abd is not None:
        angles["right_hip_abduction_2d"] = _r_hip_abd

    _l_hip_abd = hip_abduction_2d(image_landmarks, "left", min_confidence)
    if _l_hip_abd is not None:
        angles["left_hip_abduction_2d"] = _l_hip_abd

    _r_hip_full = hip_abduction_full_2d(image_landmarks, "right", min_confidence)
    if _r_hip_full is not None:
        angles["right_hip_abduction_full_2d"] = _r_hip_full

    _l_hip_full = hip_abduction_full_2d(image_landmarks, "left", min_confidence)
    if _l_hip_full is not None:
        angles["left_hip_abduction_full_2d"] = _l_hip_full

    _r_hip_abd_3d = hip_abduction_weighted_3d(
        image_landmarks, world_landmarks, "right", min_confidence,
    )
    if _r_hip_abd_3d is not None:
        angles["right_hip_abduction_weighted_3d"] = round(float(_r_hip_abd_3d), 2)

    _l_hip_abd_3d = hip_abduction_weighted_3d(
        image_landmarks, world_landmarks, "left", min_confidence,
    )
    if _l_hip_abd_3d is not None:
        angles["left_hip_abduction_weighted_3d"] = round(float(_l_hip_abd_3d), 2)

    # Métrica compuesta para ejercicios de espalda/retracción escapular.
    # Requiere que ambos hombros/lados cumplan: usar min evita que un solo lado
    # o ruido del hombro derecho valide la repetición completa.
    _right_sh_flex = angles.get("right_shoulder_flexion_2d")
    _left_sh_flex = angles.get("left_shoulder_flexion_2d")
    if _right_sh_flex is not None and _left_sh_flex is not None:
        angles["back_bilateral_min_shoulder_flexion_2d"] = round(float(min(_right_sh_flex, _left_sh_flex)), 2)
        angles["back_bilateral_avg_shoulder_flexion_2d"] = round(float((_right_sh_flex + _left_sh_flex) / 2.0), 2)

    # Métricas específicas para ejercicio de espalda:
    # brazos/manos + cuerpo que abre/cierra.
    # Se usan proporciones 2D para ser más robustas a la distancia de cámara.
    def _back_p2(name):
        p = image_landmarks.get(name)
        if p is None:
            return None
        try:
            conf = _confidence(p)
        except Exception:
            conf = 1.0
        if conf < min_confidence:
            return None
        return np.array([float(p.x), float(p.y)], dtype=np.float32)

    def _back_dist(a, b):
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a - b))

    _ls = _back_p2("left_shoulder")
    _rs = _back_p2("right_shoulder")
    _lh = _back_p2("left_hip")
    _rh = _back_p2("right_hip")
    _le = _back_p2("left_elbow")
    _re = _back_p2("right_elbow")
    _lw = _back_p2("left_wrist")
    _rw = _back_p2("right_wrist")

    _shoulder_w = _back_dist(_ls, _rs)
    _hip_w = _back_dist(_lh, _rh)
    _elbow_w = _back_dist(_le, _re)
    _wrist_w = _back_dist(_lw, _rw)

    if _shoulder_w is not None and _shoulder_w > 1e-6:
        if _elbow_w is not None:
            angles["back_elbow_width_over_shoulder_width_2d"] = round(float(_elbow_w / _shoulder_w), 3)

        if _wrist_w is not None:
            angles["back_wrist_width_over_shoulder_width_2d"] = round(float(_wrist_w / _shoulder_w), 3)

        if _hip_w is not None and _hip_w > 1e-6:
            angles["back_shoulder_width_over_hip_width_2d"] = round(float(_shoulder_w / _hip_w), 3)

    # Score 0-100 aproximado:
    # - codos abiertos respecto a hombros
    # - muñecas abiertas respecto a hombros
    # - hombros/cadera para capturar "sacar/meter" cuerpo/escápulas
    _components = []

    _ew = angles.get("back_elbow_width_over_shoulder_width_2d")
    _ww = angles.get("back_wrist_width_over_shoulder_width_2d")
    _shhip = angles.get("back_shoulder_width_over_hip_width_2d")

    if _ew is not None:
        _components.append(max(0.0, min(100.0, (_ew - 0.95) / 0.55 * 100.0)))

    if _ww is not None:
        _components.append(max(0.0, min(100.0, (_ww - 0.90) / 0.70 * 100.0)))

    if _shhip is not None:
        _components.append(max(0.0, min(100.0, (_shhip - 0.70) / 0.45 * 100.0)))

    if _components:
        angles["back_body_open_score_2d"] = round(float(sum(_components) / len(_components)), 2)

    _arms_min = angles.get("back_bilateral_min_shoulder_flexion_2d")
    _body_score = angles.get("back_body_open_score_2d")

    if _arms_min is not None and _body_score is not None:
        # Convertimos brazos aprox 3-32 grados a score 0-100.
        _arms_score = max(0.0, min(100.0, (_arms_min - 3.0) / 29.0 * 100.0))

        # min() obliga a que pasen ambas cosas:
        # brazos/manos + cuerpo/espalda.
        angles["back_combo_body_arms_score_2d"] = round(float(min(_arms_score, _body_score)), 2)

    # Métricas 3D/Z para ejercicio de espalda con componente de profundidad.
    # Mide si codos/muñecas cambian en profundidad respecto al torso,
    # no solo si las manos se mueven en 2D.
    _depth_source = world_landmarks if world_landmarks else image_landmarks

    def _back_p3(name):
        p = _depth_source.get(name) if _depth_source else None
        if p is None:
            return None

        try:
            conf = _confidence(p)
        except Exception:
            conf = 1.0

        if conf < min_confidence:
            return None

        z = float(getattr(p, "z", 0.0))
        return np.array([float(p.x), float(p.y), z], dtype=np.float32)

    _ls3 = _back_p3("left_shoulder")
    _rs3 = _back_p3("right_shoulder")
    _lh3 = _back_p3("left_hip")
    _rh3 = _back_p3("right_hip")
    _le3 = _back_p3("left_elbow")
    _re3 = _back_p3("right_elbow")
    _lw3 = _back_p3("left_wrist")
    _rw3 = _back_p3("right_wrist")

    if _ls3 is not None and _rs3 is not None and _lh3 is not None and _rh3 is not None:
        _torso_z = float((_ls3[2] + _rs3[2] + _lh3[2] + _rh3[2]) / 4.0)

        _shoulder_w_3d = float(np.linalg.norm(_ls3 - _rs3))
        if _shoulder_w_3d < 1e-6:
            _shoulder_w_3d = 1.0

        _depth_components = []

        if _le3 is not None and _re3 is not None:
            _elbow_depth = (
                abs(float(_le3[2] - _torso_z)) +
                abs(float(_re3[2] - _torso_z))
            ) / 2.0

            _elbow_depth_norm = float(_elbow_depth / _shoulder_w_3d)
            angles["back_elbow_depth_delta_3d"] = round(_elbow_depth_norm, 4)

            # Normalización empírica:
            # 0.05 = bajo movimiento profundidad
            # 0.35 = alto movimiento profundidad
            _elbow_score = max(
                0.0,
                min(100.0, (_elbow_depth_norm - 0.05) / 0.30 * 100.0),
            )
            _depth_components.append(_elbow_score)

        if _lw3 is not None and _rw3 is not None:
            _wrist_depth = (
                abs(float(_lw3[2] - _torso_z)) +
                abs(float(_rw3[2] - _torso_z))
            ) / 2.0

            _wrist_depth_norm = float(_wrist_depth / _shoulder_w_3d)
            angles["back_wrist_depth_delta_3d"] = round(_wrist_depth_norm, 4)

            # Normalización empírica:
            # 0.07 = bajo movimiento profundidad
            # 0.45 = alto movimiento profundidad
            _wrist_score = max(
                0.0,
                min(100.0, (_wrist_depth_norm - 0.07) / 0.38 * 100.0),
            )
            _depth_components.append(_wrist_score)

        # También aprovechamos la métrica bilateral de brazos si existe.
        _arms_min = angles.get("back_bilateral_min_shoulder_flexion_2d")
        if _arms_min is not None:
            _arms_score = max(
                0.0,
                min(100.0, (float(_arms_min) - 3.0) / 29.0 * 100.0),
            )
            _depth_components.append(_arms_score)

        if _depth_components:
            # min() obliga a que no baste solo manos/brazos:
            # debe haber profundidad + movimiento bilateral.
            angles["back_depth_combo_score_3d"] = round(float(min(_depth_components)), 2)

    # Métrica de elevación de pierna 2D.
    # Mide cuánto suben rodilla/tobillo respecto a la cadera, normalizado por longitud de pierna.
    # Es más útil para "alzar la pierna y bajar" que hip_flexion_2d cuando la cámara está de lado.
    def _leg_raise_score_2d(side: str):
        hip = image_landmarks.get(f"{side}_hip")
        knee = image_landmarks.get(f"{side}_knee")
        ankle = image_landmarks.get(f"{side}_ankle")

        if hip is None or knee is None or ankle is None:
            return None

        try:
            if _confidence(hip) < min_confidence or _confidence(knee) < min_confidence or _confidence(ankle) < min_confidence:
                return None
        except Exception:
            pass

        hip_xy = np.array([float(hip.x), float(hip.y)], dtype=np.float32)
        knee_xy = np.array([float(knee.x), float(knee.y)], dtype=np.float32)
        ankle_xy = np.array([float(ankle.x), float(ankle.y)], dtype=np.float32)

        thigh = float(np.linalg.norm(hip_xy - knee_xy))
        shin = float(np.linalg.norm(knee_xy - ankle_xy))
        leg_len = thigh + shin

        if leg_len < 1e-6:
            return None

        # En coordenadas de imagen, y menor significa más arriba.
        knee_lift = (float(hip.y) - float(knee.y)) / leg_len
        ankle_lift = (float(hip.y) - float(ankle.y)) / leg_len

        # Score combinado. Si la pierna está abajo, esto tiende a 0.
        raw = (0.60 * knee_lift) + (0.40 * ankle_lift)

        # Normalización empírica:
        # 0.00 = pierna abajo
        # 0.45 = pierna claramente elevada
        score = max(0.0, min(100.0, raw / 0.45 * 100.0))

        return round(float(score), 2)

    _left_leg_raise = _leg_raise_score_2d("left")
    if _left_leg_raise is not None:
        angles["left_leg_raise_score_2d"] = _left_leg_raise

    _right_leg_raise = _leg_raise_score_2d("right")
    if _right_leg_raise is not None:
        angles["right_leg_raise_score_2d"] = _right_leg_raise

    # Métrica de elevación de tobillo 2D.
    # Pensada para ejercicios tipo flexión de pierna hacia atrás:
    # el tobillo/talón sube respecto al tobillo de apoyo.
    def _ankle_raise_score_2d():
        left_ankle = image_landmarks.get("left_ankle")
        right_ankle = image_landmarks.get("right_ankle")
        left_hip = image_landmarks.get("left_hip")
        right_hip = image_landmarks.get("right_hip")

        if left_ankle is None or right_ankle is None:
            return None

        try:
            if _confidence(left_ankle) < min_confidence or _confidence(right_ankle) < min_confidence:
                return None
        except Exception:
            pass

        # Escala corporal aproximada para normalizar.
        scale = None

        if left_hip is not None and right_hip is not None:
            try:
                if _confidence(left_hip) >= min_confidence and _confidence(right_hip) >= min_confidence:
                    hip_y = (float(left_hip.y) + float(right_hip.y)) / 2.0
                    ankle_y = (float(left_ankle.y) + float(right_ankle.y)) / 2.0
                    scale = abs(ankle_y - hip_y)
            except Exception:
                scale = None

        if scale is None or scale < 1e-6:
            # Fallback: distancia vertical entre tobillos o valor fijo.
            scale = max(0.15, abs(float(left_ankle.y) - float(right_ankle.y)))

        # En imagen, y menor = más arriba.
        # Si un tobillo sube, su y será menor que el tobillo de apoyo.
        left_up = float(right_ankle.y) - float(left_ankle.y)
        right_up = float(left_ankle.y) - float(right_ankle.y)

        # Tomamos el mayor: permite que el ejercicio funcione aunque MediaPipe invierta izquierda/derecha.
        lift = max(left_up, right_up, 0.0)

        # Normalización:
        # lift/scale = 0.00 pierna abajo
        # lift/scale ~0.35-0.45 pierna bastante elevada
        score = max(0.0, min(100.0, (lift / scale) / 0.40 * 100.0))

        return round(float(score), 2)

    _ankle_raise_score = _ankle_raise_score_2d()
    if _ankle_raise_score is not None:
        angles["ankle_raise_score_2d"] = _ankle_raise_score

    # Métrica para "estirar delante" evitando sentadilla.
    # Usa la flexión de cadera/tronco SOLO si la rodilla está relativamente extendida.
    # Si la rodilla se dobla mucho, como en sentadilla, el score queda en 0.
    def _hip_hinge_knee_extended_score(side: str):
        hip_hinge = angles.get(f"{side}_hip_flexion_body_rel")
        knee_3d = angles.get(f"{side}_knee_flexion_3d")
        knee_2d = angles.get(f"{side}_knee_flexion_2d")

        knee = knee_3d if knee_3d is not None else knee_2d

        if hip_hinge is None or knee is None:
            return None

        # Rodilla casi extendida.
        # 180 = totalmente recta.
        # Una sentadilla profunda suele bajar mucho de 130.
        knee_value = float(knee)

        if knee_value < 100.0:
            return None

        return round(float(hip_hinge), 2)

    _right_hinge_gate = _hip_hinge_knee_extended_score("right")
    if _right_hinge_gate is not None:
        angles["right_hip_hinge_knee_extended_score"] = _right_hinge_gate

    _left_hinge_gate = _hip_hinge_knee_extended_score("left")
    if _left_hinge_gate is not None:
        angles["left_hip_hinge_knee_extended_score"] = _left_hinge_gate

    # MÃ©trica para ejercicio de hombros: subir y bajar hombros.
    # No usa muÃ±eca ni Ã­ndice. Usa cabeza + hombros + caderas.
    # Cuando los hombros suben, la distancia cabeza-hombros disminuye y el score aumenta.
    def _shoulder_shrug_score_2d():
        nose = image_landmarks.get("nose")
        left_ear = image_landmarks.get("left_ear")
        right_ear = image_landmarks.get("right_ear")
        left_shoulder = image_landmarks.get("left_shoulder")
        right_shoulder = image_landmarks.get("right_shoulder")
        left_hip = image_landmarks.get("left_hip")
        right_hip = image_landmarks.get("right_hip")

        if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
            return None

        try:
            if (
                _confidence(left_shoulder) < min_confidence or
                _confidence(right_shoulder) < min_confidence or
                _confidence(left_hip) < min_confidence or
                _confidence(right_hip) < min_confidence
            ):
                return None
        except Exception:
            pass

        # Punto de cabeza: preferimos punto medio de orejas; si no, nariz.
        head_y = None

        if left_ear is not None and right_ear is not None:
            try:
                if _confidence(left_ear) >= min_confidence and _confidence(right_ear) >= min_confidence:
                    head_y = (float(left_ear.y) + float(right_ear.y)) / 2.0
            except Exception:
                head_y = None

        if head_y is None and nose is not None:
            try:
                if _confidence(nose) >= min_confidence:
                    head_y = float(nose.y)
            except Exception:
                head_y = float(nose.y)

        if head_y is None:
            return None

        shoulder_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2.0
        hip_y = (float(left_hip.y) + float(right_hip.y)) / 2.0

        torso_len = abs(hip_y - shoulder_y)
        if torso_len < 1e-6:
            return None

        # En coordenadas de imagen, y menor = mÃ¡s arriba.
        # gap_ratio alto = hombros lejos de la cabeza.
        # gap_ratio bajo = hombros subidos.
        gap_ratio = (shoulder_y - head_y) / torso_len

        # NormalizaciÃ³n empÃ­rica:
        # gap_ratio ~0.62 = hombros abajo / neutral
        # gap_ratio ~0.50 = hombros subidos
        score = max(0.0, min(100.0, (0.62 - gap_ratio) / 0.12 * 100.0))

        return round(float(score), 2)

    _shrug_score = _shoulder_shrug_score_2d()
    if _shrug_score is not None:
        angles["shoulder_shrug_score_2d"] = _shrug_score

    # Métrica para subir/bajar hombros sin depender de caderas.
    # Usa cabeza + hombros.
    # Hombros abajo => valor mayor.
    # Hombros arriba => valor menor.
    def _shoulder_head_gap_over_width_2d():
        nose = image_landmarks.get("nose")
        left_ear = image_landmarks.get("left_ear")
        right_ear = image_landmarks.get("right_ear")
        left_shoulder = image_landmarks.get("left_shoulder")
        right_shoulder = image_landmarks.get("right_shoulder")

        if left_shoulder is None or right_shoulder is None:
            return None

        try:
            if _confidence(left_shoulder) < min_confidence or _confidence(right_shoulder) < min_confidence:
                return None
        except Exception:
            pass

        head_y = None

        if left_ear is not None and right_ear is not None:
            try:
                if _confidence(left_ear) >= min_confidence and _confidence(right_ear) >= min_confidence:
                    head_y = (float(left_ear.y) + float(right_ear.y)) / 2.0
            except Exception:
                head_y = None

        if head_y is None and nose is not None:
            try:
                if _confidence(nose) >= min_confidence:
                    head_y = float(nose.y)
            except Exception:
                head_y = float(nose.y)

        if head_y is None:
            return None

        ls = np.array([float(left_shoulder.x), float(left_shoulder.y)], dtype=np.float32)
        rs = np.array([float(right_shoulder.x), float(right_shoulder.y)], dtype=np.float32)

        shoulder_width = float(np.linalg.norm(ls - rs))
        if shoulder_width < 1e-6:
            return None

        shoulder_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2.0

        # En imagen, y aumenta hacia abajo.
        gap = shoulder_y - head_y

        ratio = gap / shoulder_width

        # Multiplicamos por 100 para verlo cómodo en DIAG.
        return round(float(ratio * 100.0), 2)

    _shoulder_gap_width = _shoulder_head_gap_over_width_2d()
    if _shoulder_gap_width is not None:
        angles["shoulder_head_gap_over_width_2d"] = _shoulder_gap_width

    return angles


# -----------------------------------------------------------------------------
# Métricas posturales / compensaciones base
# -----------------------------------------------------------------------------


def midpoint_2d(landmarks: dict, a: str, b: str, min_confidence: float = 0.35) -> Optional[np.ndarray]:
    if not visible_landmarks(landmarks, [a, b], min_confidence):
        return None
    pa = landmarks[a]
    pb = landmarks[b]
    return np.array([(pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0], dtype=np.float32)


def trunk_lean_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    shoulder_mid = _robust_point(landmarks, "left_shoulder", "right_shoulder", min_confidence)
    hip_mid = _robust_point(landmarks, "left_hip", "right_hip", min_confidence)
    if shoulder_mid is None or hip_mid is None:
        return None

    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0], dtype=np.float32)
    return _angle_between(trunk_vec, vertical)


def shoulder_alignment_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """
    Pendiente de la línea entre hombros respecto a la horizontal.

    Útil para detectar elevación/compensación de un hombro.
    """
    if not visible_landmarks(landmarks, ["left_shoulder", "right_shoulder"], min_confidence):
        return None
    left = landmarks["left_shoulder"]
    right = landmarks["right_shoulder"]
    dx = float(right.x - left.x)
    dy = float(right.y - left.y)
    if abs(dx) < 1e-6:
        return 90.0
    return round(abs(float(math.degrees(math.atan2(dy, dx)))), 2)


def hip_alignment_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """
    Pendiente de la línea entre caderas respecto a la horizontal.
    """
    if not visible_landmarks(landmarks, ["left_hip", "right_hip"], min_confidence):
        return None
    left = landmarks["left_hip"]
    right = landmarks["right_hip"]
    dx = float(right.x - left.x)
    dy = float(right.y - left.y)
    if abs(dx) < 1e-6:
        return 90.0
    return round(abs(float(math.degrees(math.atan2(dy, dx)))), 2)


def _robust_point(landmarks: dict, a: str, b: str, min_confidence: float = 0.35) -> Optional[np.ndarray]:
    """
    Devuelve el punto medio 2D de `a` y `b` si ambos son visibles; si solo uno
    es visible devuelve ese; si ninguno, None.

    Util cuando el paciente esta de lado y el landmark del lado opuesto tiene
    baja confianza.
    """
    pa = landmarks.get(a)
    pb = landmarks.get(b)
    a_ok = pa is not None and _confidence(pa) >= min_confidence
    b_ok = pb is not None and _confidence(pb) >= min_confidence
    if a_ok and b_ok:
        return np.array([(pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0], dtype=np.float32)
    if a_ok:
        return np.array([pa.x, pa.y], dtype=np.float32)
    if b_ok:
        return np.array([pb.x, pb.y], dtype=np.float32)
    return None


def _robust_midpoint_3d(landmarks: dict, a: str, b: str, min_confidence: float = 0.35) -> Optional[np.ndarray]:
    pa = landmarks.get(a)
    pb = landmarks.get(b)
    a_ok = pa is not None and _confidence(pa) >= min_confidence
    b_ok = pb is not None and _confidence(pb) >= min_confidence
    if a_ok and b_ok:
        return np.array([(pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0, (pa.z + pb.z) / 2.0], dtype=np.float64)
    if a_ok:
        return np.array([pa.x, pa.y, pa.z], dtype=np.float64)
    if b_ok:
        return np.array([pb.x, pb.y, pb.z], dtype=np.float64)
    return None


def shoulder_asymmetry_signed_2d(
    landmarks: dict,
    side: str = "right",
    min_confidence: float = 0.35,
) -> Optional[float]:
    """
    Desviación horizontal del hombro indicado respecto al centro del cuerpo
    (punto medio de caderas mediante _robust_point).

    Para rotación de tronco:
      - Frontal: hombro derecho a la derecha (>0), izquierdo a la izquierda (<0)
      - Al rotar: el hombro se acerca al centro en 2D → valor absoluto disminuye

    Sirve como primary_angle para ejercicios de rotación de tronco:
      - Rotar a la derecha → usar side="right", direction="decreasing"
      - Rotar a la izquierda → usar side="left", direction="increasing"
    """
    shoulder = landmarks.get(f"{side}_shoulder")
    if shoulder is None or _confidence(shoulder) < min_confidence:
        return None
    hip_mid = _robust_point(landmarks, "right_hip", "left_hip", min_confidence)
    if hip_mid is None:
        return None
    offset = float(shoulder.x - hip_mid[0])
    return round(offset, 4)


def shoulder_twist_signed_3d(
    image_landmarks: dict,
    world_landmarks: Optional[dict] = None,
    min_confidence: float = 0.35,
) -> Optional[float]:
    """
    Diferencia de profundidad (z) entre hombro derecho e izquierdo como
    medida de rotación del tronco en 3D.

    Usa _weighted_point: x,y de image_landmarks, z de world_landmarks.

    ~0 = frontal (hombros alineados en profundidad)
    >0 = rotación a la izquierda (hombro derecho más cerca que el izquierdo)
    <0 = rotación a la derecha (hombro izquierdo más cerca que el derecho)
    """
    rs = _weighted_point(image_landmarks, world_landmarks, "right_shoulder", min_confidence)
    ls = _weighted_point(image_landmarks, world_landmarks, "left_shoulder", min_confidence)
    if rs is None or ls is None:
        return None
    z_diff = float(rs[2] - ls[2])
    return round(z_diff, 4)


def _head_point(landmarks: dict, min_confidence: float = 0.35) -> Optional[np.ndarray]:
    """
    Punto representativo de la cabeza, en orden de preferencia:
      1. nariz
      2. punto medio de las orejas
      3. una sola oreja visible
    """
    nose = landmarks.get("nose")
    if nose is not None and _confidence(nose) >= min_confidence:
        return np.array([nose.x, nose.y], dtype=np.float32)
    return _robust_point(landmarks, "left_ear", "right_ear", min_confidence)


def neck_deviation_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """
    Desviacion del cuello/cabeza respecto al eje del tronco (en grados).

    A diferencia de los angulos antiguos (vertice en el hombro), aqui el vertice
    es la base del cuello (punto medio de hombros) y se mide cuanto se desvia la
    cabeza del eje vertical del tronco. Esto:
      - Es mucho mas sensible al movimiento real del cuello.
      - Esta compensado contra la inclinacion del tronco (se mide RELATIVO al
        tronco), por lo que inclinar el cuerpo no lo falsea.
      - Funciona igual para flexion lateral o sagital: cualquier desviacion
        desde el neutro aumenta el valor.

        0 grados  = cabeza alineada con el tronco (postura neutra)
        valor alto = mayor inclinacion/rotacion del cuello en cualquier direccion
    """
    neck_base = _robust_point(landmarks, "left_shoulder", "right_shoulder", min_confidence)
    hip_base = _robust_point(landmarks, "left_hip", "right_hip", min_confidence)
    head = _head_point(landmarks, min_confidence)
    if neck_base is None or hip_base is None or head is None:
        return None
    trunk_up = neck_base - hip_base       # de cadera hacia hombros (arriba)
    neck_vec = head - neck_base           # de base del cuello hacia la cabeza
    return _angle_between(neck_vec, trunk_up)


def neck_lateral_signed_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """
    Inclinacion lateral del cuello CON signo (vista frontal).

        > 0  = cabeza inclinada hacia un lado
        < 0  = cabeza inclinada hacia el otro lado
        ~0   = neutro

    El signo se calcula con el producto cruzado 2D entre el eje del tronco y el
    vector del cuello.
    """
    neck_base = _robust_point(landmarks, "left_shoulder", "right_shoulder", min_confidence)
    hip_base = _robust_point(landmarks, "left_hip", "right_hip", min_confidence)
    head = _head_point(landmarks, min_confidence)
    if neck_base is None or hip_base is None or head is None:
        return None
    trunk_up = neck_base - hip_base
    neck_vec = head - neck_base
    magnitude = _angle_between(neck_vec, trunk_up)
    if magnitude is None:
        return None
    cross = float(trunk_up[0] * neck_vec[1] - trunk_up[1] * neck_vec[0])
    sign = 1.0 if cross >= 0 else -1.0
    return round(sign * magnitude, 2)



def neck_extension_side_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    # Extension/flexion de cuello en vista lateral SIN usar cadera.
    # Usa cabeza/nariz + hombro, comparado contra la vertical de la imagen.
    head = _head_point(landmarks, min_confidence)
    if head is None:
        return None
    rs = landmarks.get("right_shoulder")
    ls = landmarks.get("left_shoulder")
    if rs is not None and _confidence(rs) >= min_confidence:
        base = np.array([rs.x, rs.y], dtype=np.float32)
    elif ls is not None and _confidence(ls) >= min_confidence:
        base = np.array([ls.x, ls.y], dtype=np.float32)
    else:
        base = _robust_point(landmarks, "left_shoulder", "right_shoulder", min_confidence)
    if base is None:
        return None
    neck_vec = head - base
    vertical_up = np.array([0.0, -1.0], dtype=np.float32)
    return _angle_between(neck_vec, vertical_up)


def head_pitch_side_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """
    Pitch/inclinacion de cabeza en vista lateral usando SOLO cabeza:
    vector oreja -> nariz contra horizontal de imagen.

    No usa cadera ni hombro como referencia principal, por lo que mover brazo,
    hombro o tronco no deberia completar la repeticion.
    """
    nose = landmarks.get("nose")
    if nose is None or _confidence(nose) < min_confidence:
        return None

    ear = landmarks.get("right_ear")
    if ear is None or _confidence(ear) < min_confidence:
        ear = landmarks.get("left_ear")
    if ear is None or _confidence(ear) < min_confidence:
        return None

    dx = float(nose.x - ear.x)
    dy = float(nose.y - ear.y)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    angle = abs(float(math.degrees(math.atan2(dy, dx))))
    if angle > 90.0:
        angle = 180.0 - angle
    return round(max(0.0, min(90.0, angle)), 2)


def head_lateral_signed_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """
    Inclinación lateral de la cabeza con signo, solo con landmarks craneales.

    Vector oreja(s) → nariz respecto a la vertical de la imagen.
      ~0°  = cabeza centrada (nariz alineada verticalmente sobre orejas)
      >0°  = inclinación hacia la derecha
      <0°  = inclinación hacia la izquierda

    No usa hombros ni caderas → 100% inmune a compensaciones.
    """
    nose = landmarks.get("nose")
    if nose is None or _confidence(nose) < min_confidence:
        return None

    ear_mid = _robust_point(landmarks, "left_ear", "right_ear", min_confidence)
    if ear_mid is None:
        return None

    dx = float(nose.x - ear_mid[0])
    dy = float(nose.y - ear_mid[1])
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None

    # Ángulo absoluto respecto a la vertical de la imagen (hacia arriba)
    mag = math.hypot(dx, dy)
    cosang = -dy / mag
    cosang = max(-1.0, min(1.0, cosang))
    angle_deg = math.degrees(math.acos(cosang))

    # Signo: nariz a la derecha → positivo
    return round(angle_deg * (1.0 if dx >= 0 else -1.0), 2)


def shoulder_width_2d(landmarks: dict, min_confidence: float = 0.35) -> Optional[float]:
    """Distancia horizontal normalizada entre hombros. Perfil = bajo; frontal = alto."""
    if not visible_landmarks(landmarks, ["left_shoulder", "right_shoulder"], min_confidence):
        return None
    left = landmarks["left_shoulder"]
    right = landmarks["right_shoulder"]
    return round(abs(float(right.x - left.x)), 4)

def tracking_quality_for(landmarks: dict, required_landmarks: list, min_confidence: float = 0.35) -> dict:
    """
    Reporte simple de calidad de tracking para los landmarks requeridos.
    """
    details = {}
    values = []
    for name in required_landmarks:
        point = landmarks.get(name)
        conf = _confidence(point)
        details[name] = {
            "visible": point is not None and conf >= min_confidence,
            "confidence": round(conf, 4),
        }
        values.append(conf)
    return {
        "ok": all(item["visible"] for item in details.values()) if details else False,
        "mean_confidence": round(float(np.mean(values)), 4) if values else 0.0,
        "details": details,
    }
