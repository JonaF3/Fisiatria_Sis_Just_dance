from __future__ import annotations

import importlib.util
import pprint
import shutil
from pathlib import Path


CONFIG = Path("just_dance_rehab_config.py")
EXERCISE_KEY = "estirar delante"

PRIMARY = "right_hip_hinge_knee_extended_score"


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

    backup(CONFIG, ".bak_estirar_delante_side_view_step38")

    cfgs = load_config(CONFIG)

    if EXERCISE_KEY not in cfgs:
        raise KeyError(f"No encontré {EXERCISE_KEY!r}")

    ex = cfgs[EXERCISE_KEY]

    ex["tracking_type"] = "pose33"
    ex["landmark_schema"] = "mediapipe_pose_33"
    ex["validation_strategy"] = "pose33_hip_hinge_no_squat_side_view"

    # Mantiene el score anti-sentadilla creado en el step37.
    ex["primary_angle"] = PRIMARY
    ex["direction"] = "increasing"
    ex["auto_calibrate"] = False

    ex["active_angles"] = [PRIMARY]

    # Posición inicial: de pie.
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

    # IMPORTANTE:
    # Quitamos right_ankle de required porque de lado se pierde mucho.
    # Lo dejamos opcional.
    ex["required_landmarks"] = [
        "right_shoulder",
        "left_shoulder",
        "right_hip",
        "left_hip",
        "right_knee",
    ]

    ex["optional_landmarks"] = [
        "right_ankle",
        "left_knee",
        "left_ankle",
    ]

    ex["compensation_rules"] = []

    # Más tolerante para cámara lateral.
    ex["min_tracking_confidence"] = 0.30
    ex["angle_smoothing"] = 0.65

    ex["min_frames_in_start"] = 5
    ex["min_frames_in_target"] = 8
    ex["min_frames_return"] = 5

    ex["return_tolerance"] = 12.0
    ex["safe_margin"] = 10.0

    feedback = ex.get("feedback", {}) or {}
    feedback.update(
        {
            "lost_tracking": "La cámara debe ver hombros, cadera y rodilla. Si se pierde el tobillo, aléjate un poco.",
            "waiting_start": "Ponte de pie con las piernas casi rectas.",
            "go_to_target": "Inclínate hacia adelante desde la cadera, sin hacer sentadilla.",
            "return_start": "Vuelve a la posición inicial de pie.",
            "rep_invalid": "No válido. Evita hacer sentadilla; intenta mantener la rodilla casi recta.",
        }
    )
    ex["feedback"] = feedback

    text = "# just_dance_rehab_config.py\n"
    text += "# Configuracion central de ejercicios de rehabilitacion fisica\n\n"
    text += "REHAB_EXERCISE_CONFIGS = "
    text += pprint.pformat(cfgs, width=120, sort_dicts=False)
    text += "\n"

    CONFIG.write_text(text, encoding="utf-8")

    print("[OK] estirar delante actualizado para cámara lateral")
    print(f"     primary_angle = {PRIMARY}")
    print("     required = hombros + caderas + rodilla")
    print("     tobillo = opcional")
    print("     start  = [-5, 25]")
    print("     target = [55, 125]")


if __name__ == "__main__":
    main()



