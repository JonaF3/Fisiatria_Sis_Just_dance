"""
just_dance_main.py - v3 rehab video resolver

Cambios:
  - run_game() acepta use_gpu=False/True (Condicion A/B)
  - Crea SessionMetrics con la condicion correcta
  - Pasa metrics al controller
  - Al terminar: lanza pantalla de resultados y guarda métricas
  - Nuevo: busca video desde just_dance_rehab_config.py usando "reference_video"
  - Nuevo: mantiene compatibilidad con carpeta vieja songs/
"""

from __future__ import annotations

import os
import pygame

from just_dance_model      import JustDanceModel
from just_dance_view       import JustDanceView
from just_dance_controller import JustDanceController


# ── Pose33: evitar carga innecesaria de MoveNet/ONNX ────────────────────────
def _is_pose33_exercise(song_key):
    try:
        from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
        cfg = REHAB_EXERCISE_CONFIGS.get(song_key, {}) or {}
        return cfg.get("tracking_type") == "pose33"
    except Exception:
        return False




def _is_hand21_exercise(song_key):
    try:
        from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
        cfg = REHAB_EXERCISE_CONFIGS.get(song_key, {}) or {}
        return cfg.get("tracking_type") == "hand21"
    except Exception:
        return False

class _Pose33DummyModel:
    """Modelo liviano usado solo cuando el controller usa MediaPipe Pose33."""
    def __init__(self, difficulty="NORMAL", use_gpu=False, **kwargs):
        self.difficulty = difficulty or "NORMAL"
        self.max_score = 50000
        self._actual_condition = "MEDIAPIPE_POSE33"
        self.angle_threshold = 18.0
        self.movement_threshold = 0.03
        self.distinctiveness_threshold = 0.05
        self.angle_weight = 0.65
        self.vector_weight = 0.35
        print("[INFO] MediaPipe dummy model activo: se omite carga de MoveNet/ONNX.")

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty or self.difficulty

    def run_inference(self, img):
        import numpy as np
        return np.zeros((1, 1, 17, 3), dtype=np.float32)

    def run_multi_inference(self, img, max_poses=2):
        return []

    def store_angles(self, *args, **kwargs):
        return None

from just_dance_metrics    import SessionMetrics
from just_dance_score      import load_profile


def _participant_context():
    profile = load_profile() or {}
    name = profile.get("name", "PACIENTE")

    participant_id = profile.get("participant_id")
    if not participant_id:
        participant_id = "".join(
            ch if ch.isalnum() else "_"
            for ch in name.strip().upper()
        )

    return participant_id or "UNKNOWN", name


def resolve_reference_video(song_key: str):
    """
    Busca el video de referencia del ejercicio.

    Prioridad:
      1. Ruta definida en just_dance_rehab_config.py como "reference_video".
      2. Carpeta songs/ (por nombre de ejercicio).
      3. Carpeta rehab_references/videos/.
    """
    candidates = []

    # 1. Ruta explícita en el config clínico
    try:
        from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
        reference_video = REHAB_EXERCISE_CONFIGS.get(song_key, {}).get("reference_video")
        if reference_video:
            candidates.append(reference_video)
    except Exception as e:
        print(f"[WARN] No se pudo leer reference_video desde config: {e}")

    # 2. Carpeta songs/
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        candidates.append(os.path.join("songs", f"{song_key}{ext}"))

    # 3. Carpeta rehab_references/videos/
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        candidates.append(os.path.join("rehab_references", "videos", f"{song_key}{ext}"))

    seen = set()
    for path in candidates:
        if not path:
            continue
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            print(f"[INFO] Video de referencia encontrado: {normalized}")
            return normalized

    print(f"[ERROR] No se encontro video para '{song_key}'")
    return None


def resolve_audio(song_key: str):
    """
    Busca audio opcional del ejercicio.
    Si no existe, el ejercicio se ejecuta en modo silencioso.
    """
    for ext in [".mp3", ".mpeg", ".wav", ".ogg"]:
        path = os.path.join("songs_audio", f"{song_key}{ext}")
        if os.path.exists(path):
            return path

    print(f"[WARN] Audio no encontrado para '{song_key}'. Ejecutando ejercicio en modo silencioso.")
    return None


class JustDanceGame:

    def __init__(
        self,
        model_path,
        video_path,
        camera_index,
        song_key=None,
        char_info=None,
        screen_w=1280,
        screen_h=720,
        difficulty="NORMAL",
        volume=0.8,
        use_gpu=False,
        expected_players=1,
        model_num_poses=1,
        repetitions=5,
    ):
        self._model_path = model_path
        self._video_path = video_path
        self._camera_index = camera_index
        self._song_key = song_key
        self._char_info = char_info
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._difficulty = difficulty
        self._volume = volume
        self._use_gpu = use_gpu
        self._expected_players = max(1, int(expected_players or 1))
        self._model_num_poses = max(1, int(model_num_poses or 1))
        self._repetitions = repetitions

        self.model = None
        self.view = None
        self.controller = None
        self.metrics = None
        self.score = 0

    def _find_video(self, song):
        """
        Método mantenido por compatibilidad.
        Ahora usa resolve_reference_video().
        """
        return resolve_reference_video(song)

    def run(self, song):
        audio_path = resolve_audio(song)

        # ── Construir modelo con condición CPU/GPU ────────────────────
        # Si el ejercicio usa MediaPipe Pose33, no cargamos MoveNet/ONNX.
        if _is_pose33_exercise(self._song_key) or _is_hand21_exercise(self._song_key):
            self.model = _Pose33DummyModel(
                difficulty=self._difficulty,
                use_gpu=self._use_gpu,
            )
        else:
            self.model = JustDanceModel(
                model_path=self._model_path,
                difficulty=self._difficulty,
                use_gpu=self._use_gpu,
                num_poses=self._model_num_poses,
            )

        # Warmup CUDA — muestra pantalla de carga
        if self._use_gpu and getattr(self.model, "_pose_api", None) == "onnx":

            import numpy as np

            screen = pygame.display.get_surface()
            if screen is None:
                screen = pygame.display.set_mode((1280, 720), pygame.NOFRAME)

            screen.fill((10, 8, 20))
            font_big = pygame.font.SysFont("Arial", 36, bold=True)
            font_small = pygame.font.SysFont("Arial", 22)

            txt1 = font_big.render("Iniciando aceleración GPU...", True, (0, 215, 255))
            txt2 = font_small.render("Esto solo ocurre una vez por sesión", True, (120, 115, 145))

            sw, sh = screen.get_size()
            screen.blit(txt1, ((sw - txt1.get_width()) // 2, sh // 2 - 40))
            screen.blit(txt2, ((sw - txt2.get_width()) // 2, sh // 2 + 20))
            pygame.display.flip()

            dummy = np.zeros(
                (
                    1,
                    self.model.ONNX_INPUT_SIZE,
                    self.model.ONNX_INPUT_SIZE,
                    3,
                ),
                dtype=np.int32,
            )

            for _ in range(3):
                self.model.pose_detector.run(
                    [self.model._onnx_output_name],
                    {self.model._onnx_input_name: dummy},
                )

            print("[INFO] CUDA warmup completado.")

        # Condición real activada por ONNX Runtime
        actual_condition = self.model._actual_condition

        # ── Crear métricas ────────────────────────────────────────────
        participant_id, participant_name = _participant_context()

        self.metrics = SessionMetrics(
            condition=actual_condition,
            song=song,
            difficulty=self._difficulty,
            participant_id=participant_id,
            participant_name=participant_name,
        )

        self.view = JustDanceView(model=self.model)

        self.controller = JustDanceController(
            model=self.model,
            video_path=self._video_path,
            camera_index=self._camera_index,
            song_key=self._song_key,
            char_info=self._char_info,
            screen_w=self._screen_w,
            screen_h=self._screen_h,
            difficulty=self._difficulty,
            volume=self._volume,
            metrics=self.metrics,
            expected_players=self._expected_players,
            repetitions=self._repetitions,
        )

        self.controller.process_frames(audio_path=audio_path)
        self.controller.release_capture()
        self.controller.close_windows()

    def calculate_final_score(self):
        self.score = getattr(self.controller, "total_points", 0)


def run_game(
    song,
    char_info=None,
    screen_w=1280,
    screen_h=720,
    difficulty="NORMAL",
    volume=0.8,
    use_gpu=False,
    repetitions=5,
):
    """
    Punto de entrada principal.

    Args:
        use_gpu:
            False -> Condicion A (ONNX CPUExecutionProvider)
            True  -> Condicion B (ONNX DmlExecutionProvider)
    """

    video_path = resolve_reference_video(song)

    if not video_path:
        return

    duo_mode = isinstance(char_info, dict) and char_info.get("mode") == "duo"

    model_path = (
        "model/pose_landmarker_lite.task"
        if duo_mode or _is_pose33_exercise(song) or _is_hand21_exercise(song)
        else "model/movenet_singlepose_thunder.onnx"
    )

    expected_players = 2 if duo_mode else 1

    if duo_mode:
        use_gpu = False

    game = JustDanceGame(
        model_path=model_path,
        video_path=video_path,
        camera_index=0,
        song_key=song,
        char_info=char_info,
        screen_w=screen_w,
        screen_h=screen_h,
        difficulty=difficulty,
        volume=volume,
        use_gpu=use_gpu,
        expected_players=expected_players,
        model_num_poses=expected_players,
        repetitions=repetitions,
    )

    game.run(song)
    game.calculate_final_score()

    # ── Pantalla de resultados ────────────────────────────────────────
    from just_dance_gui_score import Score
    import pygame

    best_combo = getattr(game.controller, "_best_combo", 0) if game.controller else 0
    joint_stats = getattr(game.controller, "joint_stats", {}) if game.controller else {}
    performance_stats = getattr(game.controller, "performance_stats", {}) if game.controller else {}

    score_window = Score(
        score=game.score,
        screen_w=screen_w,
        screen_h=screen_h,
        difficulty=difficulty,
        max_score=getattr(game.controller.model, "max_score", 12500) if game.controller else 12500,
        best_combo=best_combo,
        joint_stats=joint_stats,
        performance_stats=performance_stats,
        song_key=song,
        perfects_pct=performance_stats.get("perfect_pct", 0.0),
        song_duration=getattr(game.controller, "_game_duration", 0.0) if game.controller else 0.0,
        repetitions=repetitions,
        rep_results=getattr(game.controller, "rep_results", []),
    )

    score_window.mainloop()

    # ── Guardado de métricas ─────────────────────────────────────────
    if game.metrics is not None:
        game.metrics.finish(final_score=game.score)
        saved_path = game.metrics.save()
        print(f"[INFO] Metricas guardadas en: {saved_path}")

    # ── Volver al lobby ───────────────────────────────────────────────
    from just_dance_gui import run_lobby

    run_lobby(W=screen_w, H=screen_h)