from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
from rehab_core.controller_bridge import RehabControllerBridge

cfg = REHAB_EXERCISE_CONFIGS["right_knee_flexion_test"]

bridge = RehabControllerBridge(
    exercise_config=cfg,
    exercise_id="right_knee_flexion_test",
    target_repetitions=5,
)

bridge.start()

sequence = [
    160,  # posición inicial
    150,  # sigue en inicio
    120,  # bajando
    100,  # rango objetivo
    90,   # rango objetivo perfecto
    120,  # regresando
    150,  # regreso a posición inicial
    165,  # repetición debería completarse
]

for i, angle in enumerate(sequence, start=1):
    result = bridge.evaluate(
        angles={
            "right_knee": angle,
        },
        frame_index=i,
        timestamp_s=i * 0.1,
    )

    print(f"\nFRAME {i} | Ángulo: {angle}")
    print("Fase:", result["phase"])
    print("Score:", result["score"])
    print("Feedback:", result["feedback"])
    print("Rep completada:", result["rep_completed"])
    print("Reps válidas:", result["valid_reps"])
    print("Reps inválidas:", result["invalid_reps"])

print("\nRESUMEN FINAL:")
print(bridge.get_summary())