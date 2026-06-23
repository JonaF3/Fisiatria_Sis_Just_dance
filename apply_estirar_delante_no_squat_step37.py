"""
apply_estirar_delante_no_squat_step37.py

Fix para 'estirar delante':
- Evita que una sentadilla cuente como repetición.
- El ejercicio debe validar inclinación/flexión de cadera hacia adelante,
  pero con la rodilla relativamente extendida.
- Si la rodilla se dobla demasiado, la métrica se vuelve 0 y no cuenta.

Uso:
    python apply_estirar_delante_no_squat_step37.py
    python -m py_compile .\\rehab_core\\angles_33.py
    python -m py_compile .\\just_dance_rehab_config.py
    python just_dance_gui.py
"""

from __future__ import annotations

import importlib.util
import pprint
import shutil
from pathlib import Path


ANGLES = Path("rehab_core/angles_33.py")
CONFIG = Path("just_dance_rehab_config.py")
EXERCISE_KEY = "estirar delante"

PRIMARY = "right_hip_hinge_knee_extended_score"


GATED_HINGE_BLOCK = '''
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

        if knee_value < 135.0:
            return 0.0

        return round(float(hip_hinge), 2)

    _right_hinge_gate = _hip_hinge_knee_extended_score("right")
    if _right_hinge_gate is not None:
        angles["right_hip_hinge_knee_extended_score"] = _right_hinge_gate

    _left_hinge_gate = _hip_hinge_knee_extended_score("left")
    if _left_hinge_gate is not None:
        angles["left_hip_hinge_knee_extended_score"] = _left_hinge_gate
'''


def backup(path: Path, suffix: str):
    bak = path.with_suffix(path.suffix + suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[BACKUP] {bak}")
    else:
        print(f"[BACKUP] Ya existe {bak}")


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location(
        "just_dance_rehab_config_current",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, "REHAB_EXERCISE_CONFIGS")


def patch_angles():
    if not ANGLES.exists():
        raise FileNotFoundError("No existe rehab_core/angles_33.py")

    backup(ANGLES, ".bak_estirar_delante_no_squat")

    text = ANGLES.read_text(encoding="utf-8", errors="replace")

    if "right_hip_hinge_knee_extended_score" in text:
        print("[OK] angles_33.py ya tiene right_hip_hinge_knee_extended_score")
        return

    marker = "    return angles\n"
    idx = text.find(marker)

    if idx < 0:
        raise RuntimeError("No encontré 'return angles' en rehab_core/angles_33.py")

    text = text[:idx] + GATED_HINGE_BLOCK + text[idx:]
    ANGLES.write_text(text, encoding="utf-8")

    print("[OK] Agregada métrica anti-sentadilla para estirar delante")


def patch_config():
    if not CONFIG.exists():
        raise FileNotFoundError("No existe just_dance_rehab_config.py")

    backup(CONFIG, ".bak_estirar_delante_no_squat")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontré {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_hip_hinge_no_squat"

    # Antes usaba right_hip_flexion_body_rel directamente.
    # Ahora usa la misma flexión, pero bloqueada si la rodilla se dobla demasiado.
    ex["primary_angle"] = PRIMARY
    ex["direction"] = "increasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Posición inicial: cuerpo casi vertical.
    ex["start_ranges"] = {
        PRIMARY: [-5.0, 25.0],
    }

    # Objetivo: inclinación clara hacia adelante.
    ex["target_ranges"] = {
        PRIMARY: [55.0, 125.0],
    }

    ex["safe_ranges"] = {
        PRIMARY: [0.0, 140.0],
    }

    # Requerimos rodilla y tobillo para poder detectar sentadilla.
    ex["required_landmarks"] = [
        "right_shoulder",
        "left_shoulder",
        "right_hip",
        "left_hip",
        "right_knee",
        "right_ankle",
    ]

    ex["optional_landmarks"] = [
        "left_knee",
        "left_ankle",
    ]

    ex["compensation_rules"] = []

    # Tracking algo permisivo porque en tus logs la rodilla a veces baja de confianza,
    # pero suficientemente estable para evitar falsos positivos.
    ex["min_tracking_confidence"] = 0.35
    ex["angle_smoothing"] = 0.65

    ex["min_frames_in_start"] = 5
    ex["min_frames_in_target"] = 8
    ex["min_frames_return"] = 5

    ex["return_tolerance"] = 12.0
    ex["safe_margin"] = 10.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cámara debe ver hombros, cadera, rodilla y tobillo. Aléjate un poco si se pierden las piernas.",
            "waiting_start": "Ponte de pie con las piernas casi rectas.",
            "go_to_target": "Inclínate hacia adelante desde la cadera, manteniendo las rodillas casi rectas.",
            "return_start": "Vuelve a la posición inicial de pie.",
            "rep_invalid": "No válido. Evita hacer sentadilla; las rodillas deben mantenerse casi rectas.",
        }
    )
    ex["feedback"] = feedback

    text = "# just_dance_rehab_config.py\n"
    text += "# Configuracion central de ejercicios de rehabilitacion fisica\n\n"
    text += "REHAB_EXERCISE_CONFIGS = "
    text += pprint.pformat(cfgs, width=120, sort_dicts=False)
    text += "\n"

    CONFIG.write_text(text, encoding="utf-8")

    print("[OK] estirar delante actualizado con bloqueo anti-sentadilla")
    print(f"     primary_angle = {PRIMARY}")
    print("     start  = [-5, 25]")
    print("     target = [55, 125]")
    print("     rodilla mínima extendida = 135 grados")


def main():
    patch_angles()
    patch_config()

    print("\nAhora ejecuta:")
    print("  python -m py_compile .\\rehab_core\\angles_33.py")
    print("  python -m py_compile .\\just_dance_rehab_config.py")
    print("  python just_dance_gui.py")
    print("\nEn DIAG debe aparecer:")
    print("  right_hip_hinge_knee_extended_score raw=...")


if __name__ == "__main__":
    main()