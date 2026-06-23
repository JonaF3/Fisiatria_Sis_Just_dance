"""
apply_hombro_shrug_step40.py

Fix para ejercicio 'hombro':
- El ejercicio real es subir y bajar los hombros.
- Antes estaba usando right_wrist_pose_angle_2d, que depende de mano/muÃ±eca/Ã­ndice.
- Este patch agrega una mÃ©trica especÃ­fica:
    shoulder_shrug_score_2d

Uso:
    python apply_hombro_shrug_step40.py
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

PRIMARY = "shoulder_shrug_score_2d"


SHRUG_BLOCK = '''
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

    backup(ANGLES, ".bak_hombro_shrug")

    text = ANGLES.read_text(encoding="utf-8", errors="replace")

    if "shoulder_shrug_score_2d" in text:
        print("[OK] angles_33.py ya tiene shoulder_shrug_score_2d")
        return

    marker = "    return angles\n"
    idx = text.find(marker)

    if idx < 0:
        raise RuntimeError("No encontrÃ© 'return angles' en rehab_core/angles_33.py")

    text = text[:idx] + SHRUG_BLOCK + text[idx:]
    ANGLES.write_text(text, encoding="utf-8")

    print("[OK] Agregada mÃ©trica shoulder_shrug_score_2d")


def patch_config():
    if not CONFIG.exists():
        raise FileNotFoundError("No existe just_dance_rehab_config.py")

    backup(CONFIG, ".bak_hombro_shrug")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontrÃ© {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_shoulder_shrug"

    # Cambio principal: ya no usamos right_wrist_pose_angle_2d.
    ex["primary_angle"] = PRIMARY
    ex["direction"] = "increasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Hombros abajo / neutral.
    ex["start_ranges"] = {
        PRIMARY: [0.0, 35.0],
    }

    # Hombros elevados.
    # Si valida con poco movimiento, subir a [55, 100].
    # Si no valida nunca, bajar a [35, 100].
    ex["target_ranges"] = {
        PRIMARY: [45.0, 100.0],
    }

    ex["safe_ranges"] = {
        PRIMARY: [0.0, 100.0],
    }

    # No necesitamos muÃ±eca ni Ã­ndice.
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

    ex["return_tolerance"] = 10.0
    ex["safe_margin"] = 5.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cÃ¡mara debe ver cabeza, hombros y cadera.",
            "waiting_start": "Relaja los hombros abajo.",
            "go_to_target": "Sube ambos hombros como en el video.",
            "return_start": "Baja los hombros a la posiciÃ³n inicial.",
            "rep_invalid": "No vÃ¡lido. Sube y baja los hombros sin mover demasiado la cabeza.",
        }
    )
    ex["feedback"] = feedback

    out = "# just_dance_rehab_config.py\n"
    out += "# Configuracion central de ejercicios de rehabilitacion fisica\n\n"
    out += "REHAB_EXERCISE_CONFIGS = "
    out += pprint.pformat(cfgs, width=120, sort_dicts=False)
    out += "\n"

    CONFIG.write_text(out, encoding="utf-8")

    print("[OK] hombro actualizado para subir/bajar hombros")
    print(f"     primary_angle = {PRIMARY}")
    print("     start  = [0, 35]")
    print("     target = [45, 100]")
    print("     ya no usa muÃ±eca/Ã­ndice")


def main():
    patch_angles()
    patch_config()

    print("\nAhora ejecuta:")
    print("  python -m py_compile .\\rehab_core\\angles_33.py")
    print("  python -m py_compile .\\just_dance_rehab_config.py")
    print("  python just_dance_gui.py")
    print("\nEn DIAG debe aparecer:")
    print("  shoulder_shrug_score_2d raw=...")


if __name__ == "__main__":
    main()
