"""
apply_hombro_shrug_gap_step41.py

Fix para ejercicio 'hombro':
- El ejercicio real es subir y bajar hombros.
- El step40 saturaba shoulder_shrug_score_2d en 100.
- Este patch agrega una métrica cruda:
    shoulder_head_gap_ratio_2d

Interpretación:
- Hombros abajo: distancia cabeza-hombros mayor -> valor alto.
- Hombros arriba: distancia cabeza-hombros menor -> valor bajo.

Por eso se usa:
    direction = decreasing

Uso:
    python apply_hombro_shrug_gap_step41.py
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
EXERCISE_KEY = "hombro"

PRIMARY = "shoulder_head_gap_ratio_2d"


GAP_BLOCK = '''
    # Métrica cruda para ejercicio de hombros: subir/bajar hombros.
    # Mide distancia vertical cabeza-hombros normalizada por torso.
    # Hombros abajo => valor más alto.
    # Hombros arriba => valor más bajo.
    def _shoulder_head_gap_ratio_2d():
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

        head_y = None

        # Mejor usar orejas si están disponibles porque son más estables que la nariz.
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

        # En imagen, y aumenta hacia abajo.
        gap = shoulder_y - head_y

        ratio = gap / torso_len

        # Multiplicamos por 100 para que sea cómodo de leer en DIAG.
        return round(float(ratio * 100.0), 2)

    _gap_ratio = _shoulder_head_gap_ratio_2d()
    if _gap_ratio is not None:
        angles["shoulder_head_gap_ratio_2d"] = _gap_ratio
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

    backup(ANGLES, ".bak_hombro_shrug_gap")

    text = ANGLES.read_text(encoding="utf-8", errors="replace")

    if "shoulder_head_gap_ratio_2d" in text:
        print("[OK] angles_33.py ya tiene shoulder_head_gap_ratio_2d")
        return

    marker = "    return angles\\n"
    idx = text.find(marker)

    if idx < 0:
        raise RuntimeError("No encontré 'return angles' en rehab_core/angles_33.py")

    text = text[:idx] + GAP_BLOCK + text[idx:]
    ANGLES.write_text(text, encoding="utf-8")

    print("[OK] Agregada métrica shoulder_head_gap_ratio_2d")


def patch_config():
    if not CONFIG.exists():
        raise FileNotFoundError("No existe just_dance_rehab_config.py")

    backup(CONFIG, ".bak_hombro_shrug_gap")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontré {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_shoulder_shrug_gap"

    ex["primary_angle"] = PRIMARY

    # Importante: al subir hombros, el valor baja.
    ex["direction"] = "decreasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Estos rangos son iniciales.
    # Ajustamos con el DIAG real si hace falta.
    ex["start_ranges"] = {
        PRIMARY: [45.0, 90.0],
    }

    ex["target_ranges"] = {
        PRIMARY: [25.0, 43.0],
    }

    ex["safe_ranges"] = {
        PRIMARY: [15.0, 100.0],
    }

    ex["required_landmarks"] = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]

    ex["optional_landmarks"] = [
        "left_ear",
        "right_ear",
        "left_elbow",
        "right_elbow",
    ]

    ex["compensation_rules"] = []

    ex["min_tracking_confidence"] = 0.45
    ex["angle_smoothing"] = 0.60

    ex["min_frames_in_start"] = 5
    ex["min_frames_in_target"] = 6
    ex["min_frames_return"] = 5

    ex["return_tolerance"] = 8.0
    ex["safe_margin"] = 5.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cámara debe ver cabeza, hombros y cadera.",
            "waiting_start": "Relaja los hombros abajo.",
            "go_to_target": "Sube ambos hombros como en el video.",
            "return_start": "Baja los hombros a la posición inicial.",
            "rep_invalid": "No válido. Sube y baja los hombros sin mover demasiado la cabeza.",
        }
    )
    ex["feedback"] = feedback

    out = "# just_dance_rehab_config.py\\n"
    out += "# Configuracion central de ejercicios de rehabilitacion fisica\\n\\n"
    out += "REHAB_EXERCISE_CONFIGS = "
    out += pprint.pformat(cfgs, width=120, sort_dicts=False)
    out += "\\n"

    CONFIG.write_text(out, encoding="utf-8")

    print("[OK] hombro actualizado con shoulder_head_gap_ratio_2d")
    print(f"     primary_angle = {PRIMARY}")
    print("     direction = decreasing")
    print("     start  = [45, 90]")
    print("     target = [25, 43]")


def main():
    patch_angles()
    patch_config()

    print("\\nAhora ejecuta:")
    print("  python -m py_compile .\\\\rehab_core\\\\angles_33.py")
    print("  python -m py_compile .\\\\just_dance_rehab_config.py")
    print("  python just_dance_gui.py")
    print("\\nEn DIAG debe aparecer:")
    print("  shoulder_head_gap_ratio_2d raw=...")


if __name__ == "__main__":
    main()