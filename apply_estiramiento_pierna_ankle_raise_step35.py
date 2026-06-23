"""
apply_estiramiento_pierna_ankle_raise_step35.py

Fix para 'estiramiento pierna':
- Deja de validar con rodilla/cadera.
- Valida cuánto sube el tobillo de una pierna respecto al tobillo de apoyo.
- Funciona mejor cuando la persona está de lado y MediaPipe confunde rodilla/cadera.

Métrica nueva:
    ankle_raise_score_2d

Uso:
    python apply_estiramiento_pierna_ankle_raise_step35.py
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
EXERCISE_KEY = "estiramiento pierna"

PRIMARY = "ankle_raise_score_2d"


ANKLE_RAISE_BLOCK = '''
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

    backup(ANGLES, ".bak_ankle_raise_score")

    text = ANGLES.read_text(encoding="utf-8", errors="replace")

    if "ankle_raise_score_2d" in text:
        print("[OK] angles_33.py ya tiene ankle_raise_score_2d")
        return

    marker = "    return angles\n"
    idx = text.find(marker)

    if idx < 0:
        raise RuntimeError("No encontré 'return angles' en rehab_core/angles_33.py")

    text = text[:idx] + ANKLE_RAISE_BLOCK + text[idx:]
    ANGLES.write_text(text, encoding="utf-8")

    print("[OK] Agregada métrica ankle_raise_score_2d")


def patch_config():
    if not CONFIG.exists():
        raise FileNotFoundError("No existe just_dance_rehab_config.py")

    backup(CONFIG, ".bak_estiramiento_pierna_ankle_raise")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontré {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_ankle_raise"

    ex["primary_angle"] = PRIMARY
    ex["direction"] = "increasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Inicio: ambos tobillos abajo/casi alineados.
    ex["start_ranges"] = {
        PRIMARY: [0.0, 15.0],
    }

    # Objetivo: un tobillo sube claramente.
    # Si valida con poco movimiento, subir a [45, 100].
    # Si no valida nunca, bajar a [25, 100].
    ex["target_ranges"] = {
        PRIMARY: [35.0, 100.0],
    }

    ex["safe_ranges"] = {
        PRIMARY: [0.0, 100.0],
    }

    # Solo exigimos tobillos y caderas. Rodillas quedan opcionales porque se mezclan.
    ex["required_landmarks"] = [
        "left_ankle",
        "right_ankle",
        "left_hip",
        "right_hip",
    ]

    ex["optional_landmarks"] = [
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
    ]

    ex["compensation_rules"] = []

    # Tracking más bajo para no perder tobillo, pero suficiente para evitar ruido extremo.
    ex["min_tracking_confidence"] = 0.45
    ex["angle_smoothing"] = 0.60

    ex["min_frames_in_start"] = 5
    ex["min_frames_in_target"] = 8
    ex["min_frames_return"] = 5

    ex["return_tolerance"] = 10.0
    ex["safe_margin"] = 5.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cámara debe ver ambos tobillos y la cadera. Aléjate un poco para que salgan los pies completos.",
            "waiting_start": "Coloca ambos pies abajo en posición inicial.",
            "go_to_target": "Sube el talón/tobillo como en el video.",
            "return_start": "Baja el pie a la posición inicial.",
            "rep_invalid": "No válido. El tobillo debe subir claramente y volver a bajar.",
        }
    )
    ex["feedback"] = feedback

    out = "# just_dance_rehab_config.py\n"
    out += "# Configuracion central de ejercicios de rehabilitacion fisica\n\n"
    out += "REHAB_EXERCISE_CONFIGS = "
    out += pprint.pformat(cfgs, width=120, sort_dicts=False)
    out += "\n"

    CONFIG.write_text(out, encoding="utf-8")

    print("[OK] estiramiento pierna actualizado")
    print(f"     primary_angle = {PRIMARY}")
    print("     start  = [0, 15]")
    print("     target = [35, 100]")


def main():
    patch_angles()
    patch_config()

    print("\nAhora ejecuta:")
    print("  python -m py_compile .\\rehab_core\\angles_33.py")
    print("  python -m py_compile .\\just_dance_rehab_config.py")
    print("  python just_dance_gui.py")
    print("\nEn DIAG debe aparecer:")
    print("  ankle_raise_score_2d raw=...")


if __name__ == "__main__":
    main()