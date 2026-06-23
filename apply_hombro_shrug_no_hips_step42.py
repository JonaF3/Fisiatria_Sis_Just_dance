"""
apply_hombro_shrug_no_hips_step42.py

Fix para ejercicio 'hombro':
- El movimiento real es subir y bajar hombros.
- No debe depender de muñeca, índice ni caderas.
- Agrega métrica:
    shoulder_head_gap_over_width_2d

Interpretación:
- Hombros abajo: cabeza-hombros más separado -> valor alto.
- Hombros arriba: cabeza-hombros menos separado -> valor bajo.
- Por eso direction = decreasing.

Uso:
    python apply_hombro_shrug_no_hips_step42.py
    python -m py_compile .\\rehab_core\\angles_33.py
    python -m py_compile .\\just_dance_rehab_config.py
    python just_dance_gui.py
"""

from __future__ import annotations

import importlib.util
import pprint
import re
import shutil
from pathlib import Path


ANGLES = Path("rehab_core/angles_33.py")
CONFIG = Path("just_dance_rehab_config.py")
EXERCISE_KEY = "hombro"

PRIMARY = "shoulder_head_gap_over_width_2d"


METRIC_BLOCK = '''
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

    backup(ANGLES, ".bak_hombro_shrug_no_hips")

    text = ANGLES.read_text(encoding="utf-8", errors="replace")

    if "shoulder_head_gap_over_width_2d" in text:
        print("[OK] angles_33.py ya tiene shoulder_head_gap_over_width_2d")
        return

    # Buscar cualquier línea que sea "return angles", aunque tenga espacios.
    matches = list(re.finditer(r"(?m)^\\s*return\\s+angles\\s*$", text))
    if not matches:
        raise RuntimeError("No encontré una línea 'return angles' en rehab_core/angles_33.py")

    # Insertamos antes del último return angles.
    idx = matches[-1].start()

    text = text[:idx] + METRIC_BLOCK + "\\n" + text[idx:]
    ANGLES.write_text(text, encoding="utf-8")

    print("[OK] Agregada métrica shoulder_head_gap_over_width_2d")


def patch_config():
    if not CONFIG.exists():
        raise FileNotFoundError("No existe just_dance_rehab_config.py")

    backup(CONFIG, ".bak_hombro_shrug_no_hips")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontré {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_shoulder_shrug_no_hips"

    ex["primary_angle"] = PRIMARY
    ex["direction"] = "decreasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Rangos iniciales. Luego los afinamos con DIAG.
    # Hombros abajo = valor más alto.
    # Hombros arriba = valor más bajo.
    ex["start_ranges"] = {
        PRIMARY: [80.0, 150.0],
    }

    ex["target_ranges"] = {
        PRIMARY: [45.0, 78.0],
    }

    ex["safe_ranges"] = {
        PRIMARY: [20.0, 180.0],
    }

    # No requerimos caderas, muñecas ni índice.
    ex["required_landmarks"] = [
        "nose",
        "left_shoulder",
        "right_shoulder",
    ]

    ex["optional_landmarks"] = [
        "left_ear",
        "right_ear",
        "left_hip",
        "right_hip",
        "left_elbow",
        "right_elbow",
    ]

    ex["compensation_rules"] = []

    ex["min_tracking_confidence"] = 0.40
    ex["angle_smoothing"] = 0.65

    ex["min_frames_in_start"] = 5
    ex["min_frames_in_target"] = 6
    ex["min_frames_return"] = 5

    ex["return_tolerance"] = 8.0
    ex["safe_margin"] = 5.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cámara debe ver cabeza y ambos hombros.",
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

    print("[OK] hombro actualizado sin depender de caderas")
    print(f"     primary_angle = {PRIMARY}")
    print("     direction = decreasing")
    print("     start  = [80, 150]")
    print("     target = [45, 78]")


def main():
    patch_angles()
    patch_config()

    print("\\nAhora ejecuta:")
    print("  python -m py_compile .\\rehab_core\\angles_33.py")
    print("  python -m py_compile .\\just_dance_rehab_config.py")
    print("  python just_dance_gui.py")
    print("\\nEn DIAG debe aparecer:")
    print("  shoulder_head_gap_over_width_2d raw=...")


if __name__ == "__main__":
    main()