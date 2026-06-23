from __future__ import annotations

import importlib.util
import pprint
import shutil
from pathlib import Path


CONFIG = Path("just_dance_rehab_config.py")
EXERCISE_KEY = "estiramiento pierna"

PRIMARY = "left_knee_flexion_3d"


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


def main():
    if not CONFIG.exists():
        raise FileNotFoundError("No existe just_dance_rehab_config.py")

    backup(CONFIG, ".bak_estiramiento_pierna_knee_decreasing")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontré {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_knee_flexion_decreasing"

    # Usar rodilla izquierda porque en tus logs esta fue la métrica que sí cambiaba bien.
    ex["primary_angle"] = PRIMARY
    ex["direction"] = "decreasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Pierna abajo/recta:
    # En tu log aparecía 160–175 cuando estaba abajo.
    ex["start_ranges"] = {
        PRIMARY: [150.0, 185.0],
    }

    # Pierna levantada/doblada:
    # En tu log aparecía 80–90 cuando levantabas.
    ex["target_ranges"] = {
        PRIMARY: [65.0, 125.0],
    }

    ex["safe_ranges"] = {
        PRIMARY: [40.0, 190.0],
    }

    ex["required_landmarks"] = [
        "left_hip",
        "left_knee",
        "left_ankle",
    ]

    ex["optional_landmarks"] = [
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_shoulder",
        "right_shoulder",
    ]

    ex["compensation_rules"] = []

    # Subimos tracking para evitar puntos falsos.
    ex["min_tracking_confidence"] = 0.60

    # Más suavizado y más frames para que no cuente por picos.
    ex["angle_smoothing"] = 0.70
    ex["min_frames_in_start"] = 5
    ex["min_frames_in_target"] = 8
    ex["min_frames_return"] = 5

    ex["return_tolerance"] = 12.0
    ex["safe_margin"] = 10.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cámara debe ver cadera, rodilla y tobillo de la pierna que se mueve.",
            "waiting_start": "Coloca la pierna abajo y recta.",
            "go_to_target": "Dobla y levanta la pierna como en el video.",
            "return_start": "Baja la pierna y vuelve a dejarla recta.",
            "rep_invalid": "No válido. La rodilla debe doblarse claramente y volver a extenderse.",
        }
    )
    ex["feedback"] = feedback

    text = "# just_dance_rehab_config.py\n"
    text += "# Configuracion central de ejercicios de rehabilitacion fisica\n\n"
    text += "REHAB_EXERCISE_CONFIGS = "
    text += pprint.pformat(cfgs, width=120, sort_dicts=False)
    text += "\n"

    CONFIG.write_text(text, encoding="utf-8")

    print("[OK] estiramiento pierna actualizado")
    print(f"     primary_angle = {PRIMARY}")
    print("     direction = decreasing")
    print("     start  = [150, 185]")
    print("     target = [65, 125]")
    print("     min_frames_in_target = 8")


if __name__ == "__main__":
    main()