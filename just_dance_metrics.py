from typing import Optional
"""
just_dance_metrics.py
=====================
Logger de métricas de rendimiento para el experimento CPU vs GPU.

Registra automáticamente durante cada sesión:
  - FPS (promedio, min, max)
  - Latencia promedio por frame (ms)
  - Tiempo de inferencia del modelo de pose (ms)
  - Tiempo total de procesamiento por frame (ms)
  - CPU % del proceso (psutil)
  - GPU % (GPUtil si disponible)
  - RAM MB (psutil)
  - Frames perdidos / tardíos
  - Condición experimental: "CPU" o "GPU"
  - Canción, dificultad, puntaje final

  Estimación de pose (nuevos):
  - Estabilidad temporal de keypoints (varianza posicional entre frames)
  - Pérdida promedio de keypoints por frame (dropout)
  - Variabilidad angular entre frames consecutivos
  - Consistencia de trayectoria corporal (suavidad)
  - Frames con movimiento rápido / errores de seguimiento

  - Respuestas cuestionario Likert (5 preguntas, escala 1–5)

Salida: metrics/sesion_YYYYMMDD_HHMMSS.csv
"""

import csv
import os
import time
import threading
import contextlib
from datetime import datetime

try:
    import psutil as _psutil
    _PROC      = _psutil.Process(os.getpid())
    _CPU_COUNT = max(1, _psutil.cpu_count(logical=True) or 1)
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    _PROC      = None
    _CPU_COUNT = 1

try:
    import GPUtil as _GPUtil
    _GPUTIL_OK = True
except ImportError:
    _GPUTIL_OK = False

import numpy as np


# ── Cuestionario Likert ───────────────────────────────────────────────────────

LIKERT_QUESTIONS = [
    "1. Fluidez percibida del sistema",
    "2. Facilidad de interaccion",
    "3. Percepcion de precision del puntaje",
    "4. Confianza en la retroalimentacion",
    "5. Satisfaccion general",
]

LIKERT_LABELS = {
    1: "Muy malo",
    2: "Malo",
    3: "Regular",
    4: "Bueno",
    5: "Muy bueno",
}

# ── Constantes para métricas de pose ─────────────────────────────────────────

_KP_CONF_THRESHOLD    = 0.25   # confianza mínima para considerar keypoint válido
_FAST_MOVE_THRESHOLD  = 30.0   # grados — delta angular que indica movimiento rápido/error
_POSE_JOINTS = [
    "left_arm", "right_arm", "left_elbow", "right_elbow",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ── SessionMetrics ────────────────────────────────────────────────────────────

class SessionMetrics:
    """
    Colecta métricas durante una sesión de juego y las persiste en CSV.
    Diseñado para overhead mínimo: las lecturas costosas (CPU/GPU) ocurren
    en un hilo background cada 0.5 s, sin bloquear el loop principal.
    """

    OUTPUT_DIR = "metrics"

    def __init__(
        self,
        condition: str,
        song: str,
        difficulty: str,
        participant_id: str = "unknown",
        participant_name: str = "PACIENTE",
    ):
        self.condition        = condition.upper()
        self.song             = song
        self.difficulty       = difficulty
        self.participant_id   = participant_id or "unknown"
        self.participant_name = participant_name or "PACIENTE"

        # ── Contadores de frames ──────────────────────────────────────
        self._frame_times: list         = []
        self._infer_times: list         = []
        self._dropped_frames: int       = 0
        self._video_dropped_frames: int = 0
        self._camera_dropped_frames: int = 0
        self._late_frames: int          = 0
        self._frame_target_s: float     = 1.0 / 30.0

        # ── Muestras de recursos (hilo background) ────────────────────
        self._cpu_samples: list = []
        self._gpu_samples: list = []
        self._ram_samples: list = []

        # ── Timers ────────────────────────────────────────────────────
        self._frame_start: float      = 0.0
        self._infer_start: float      = 0.0
        self._current_infer_ms: float = 0.0

        # ── Tiempos de sesión ─────────────────────────────────────────
        self._session_start: float = 0.0
        self._session_end:   float = 0.0

        # ── Resultados finales ────────────────────────────────────────
        self.final_score: int = 0
        self.likert: dict     = {}

        # ── Métricas de estimación de pose ────────────────────────────
        # Posiciones (x, y) normalizadas de los 17 keypoints por frame
        self._kp_positions: list   = []   # lista de np.array shape (17, 2)
        # Confianza de cada keypoint por frame
        self._kp_confidences: list = []   # lista de np.array shape (17,)
        # Dropout: número de keypoints < umbral por frame
        self._kp_dropout_per_frame: list = []   # lista de int
        # Historial de ángulos por joint: dict {joint: [val_t0, val_t1, ...]}
        self._angles_history: dict = {j: [] for j in _POSE_JOINTS}
        # Delta angular entre frames consecutivos por joint
        self._angle_deltas: dict   = {j: [] for j in _POSE_JOINTS}
        # Frames donde algún joint supera el umbral de movimiento rápido
        self._fast_movement_frames: int = 0

        # ── Hilo de muestreo de recursos ──────────────────────────────
        self._stop_bg   = threading.Event()
        self._bg_thread = threading.Thread(
            target=self._resource_sampler, daemon=True
        )

        # ── Overlay ───────────────────────────────────────────────────
        self.overlay_visible: bool = False

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def start(self):
        self._session_start = time.perf_counter()
        if _PSUTIL_OK:
            _PROC.cpu_percent(interval=None)
        self._bg_thread.start()

    def finish(self, final_score: int = 0):
        self._session_end = time.perf_counter()
        self.final_score  = final_score
        self._stop_bg.set()

    # ── Context managers ──────────────────────────────────────────────────────

    @contextlib.contextmanager
    def frame_timer(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._frame_times.append(time.perf_counter() - t0)

    @contextlib.contextmanager
    def inference_timer(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._infer_times.append(dt)
            self._current_infer_ms = dt * 1000.0

    def record_frame(self, dropped: bool = False, late: bool = False):
        if dropped:
            self._dropped_frames += 1
            return
        if not late and self._frame_times:
            late = self._frame_times[-1] > self._frame_target_s * 1.5
        if late:
            self._late_frames += 1

    def record_video_drop(self):
        self._video_dropped_frames += 1
        self._dropped_frames += 1

    def record_camera_drop(self):
        self._camera_dropped_frames += 1
        self._dropped_frames += 1

    # ── Métricas de pose ──────────────────────────────────────────────────────

    def log_pose_frame(self, keypoints_raw, angles: Optional[dict] = None):
        """
        Registrar un frame de estimación de pose.

        Args:
            keypoints_raw : salida del modelo — shape (1, 1, 17, 3) con [y, x, conf]
                            o None si no hay detección.
            angles        : dict {joint_name: float} con los ángulos del frame actual.
                            Puede ser None si el cuerpo no estaba visible.
        """
        # ── Keypoints ─────────────────────────────────────────────────
        if keypoints_raw is not None:
            try:
                kp = np.squeeze(np.array(keypoints_raw, dtype=np.float32))
                if kp.ndim == 2 and kp.shape == (17, 3):
                    positions   = kp[:, :2][:, ::-1]   # (17,2) en orden [x, y]
                    confidences = kp[:, 2]              # (17,)
                    dropout = int(np.sum(confidences < _KP_CONF_THRESHOLD))

                    self._kp_positions.append(positions)
                    self._kp_confidences.append(confidences)
                    self._kp_dropout_per_frame.append(dropout)
            except Exception:
                pass

        # ── Ángulos ───────────────────────────────────────────────────
        if angles:
            is_fast_frame = False
            for joint in _POSE_JOINTS:
                val = angles.get(joint)
                if val is None:
                    continue
                history = self._angles_history[joint]
                if history:
                    delta = abs(val - history[-1])
                    self._angle_deltas[joint].append(delta)
                    if delta > _FAST_MOVE_THRESHOLD:
                        is_fast_frame = True
                history.append(val)
            if is_fast_frame:
                self._fast_movement_frames += 1

    # ── Propiedades de rendimiento ────────────────────────────────────────────

    @property
    def fps_avg(self) -> float:
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def fps_min(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1.0 / max(self._frame_times)

    @property
    def fps_max(self) -> float:
        if not self._frame_times:
            return 0.0
        min_dt = min(self._frame_times)
        return 1.0 / min_dt if min_dt > 0 else 0.0

    @property
    def latency_avg_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return (sum(self._frame_times) / len(self._frame_times)) * 1000.0

    @property
    def inference_avg_ms(self) -> float:
        if not self._infer_times:
            return 0.0
        return (sum(self._infer_times) / len(self._infer_times)) * 1000.0

    @property
    def proc_total_avg_ms(self) -> float:
        return self.latency_avg_ms

    @property
    def cpu_avg(self) -> float:
        if not self._cpu_samples:
            return -1.0
        return sum(self._cpu_samples) / len(self._cpu_samples)

    @property
    def gpu_avg(self) -> float:
        if not self._gpu_samples:
            return -1.0
        return sum(self._gpu_samples) / len(self._gpu_samples)

    @property
    def ram_avg_mb(self) -> float:
        if not self._ram_samples:
            return -1.0
        return sum(self._ram_samples) / len(self._ram_samples)

    @property
    def current_infer_ms(self) -> float:
        return self._current_infer_ms

    # ── Propiedades de estimación de pose ─────────────────────────────────────

    @property
    def kp_stability(self) -> float:
        """
        Estabilidad temporal de keypoints.
        Promedio de la desviación estándar posicional (x,y) de cada keypoint
        a lo largo de los frames. Valor bajo = mayor estabilidad.
        Devuelve -1.0 si no hay datos suficientes.
        """
        if len(self._kp_positions) < 5:
            return -1.0
        positions = np.stack(self._kp_positions, axis=0)   # (T, 17, 2)
        confidences = np.stack(self._kp_confidences, axis=0)  # (T, 17)
        stds = []
        for kp_idx in range(17):
            valid_mask = confidences[:, kp_idx] >= _KP_CONF_THRESHOLD
            if np.sum(valid_mask) < 3:
                continue
            pos = positions[valid_mask, kp_idx, :]   # (N, 2)
            stds.append(float(np.mean(np.std(pos, axis=0))))
        return float(np.mean(stds)) if stds else -1.0

    @property
    def kp_dropout_avg(self) -> float:
        """
        Promedio de keypoints perdidos (confianza < umbral) por frame.
        Rango 0–17.
        """
        if not self._kp_dropout_per_frame:
            return -1.0
        return float(np.mean(self._kp_dropout_per_frame))

    @property
    def kp_dropout_pct(self) -> float:
        """Porcentaje de keypoints perdidos promedio por frame (0–100)."""
        avg = self.kp_dropout_avg
        return round(avg / 17.0 * 100.0, 2) if avg >= 0 else -1.0

    @property
    def angular_variability_avg(self) -> float:
        """
        Variabilidad angular promedio entre frames consecutivos.
        Promedio de |ángulo[t] - ángulo[t-1]| sobre todos los joints y frames.
        Valor alto puede indicar inestabilidad del tracking.
        """
        all_deltas = []
        for joint in _POSE_JOINTS:
            all_deltas.extend(self._angle_deltas[joint])
        if not all_deltas:
            return -1.0
        return float(np.mean(all_deltas))

    @property
    def trajectory_consistency(self) -> float:
        """
        Consistencia de trayectoria corporal (0–100, mayor = más suave).
        Calculada como 100 * (1 - coef_variacion_normalizado) de los deltas angulares.
        """
        all_deltas = []
        for joint in _POSE_JOINTS:
            all_deltas.extend(self._angle_deltas[joint])
        if len(all_deltas) < 10:
            return -1.0
        arr = np.array(all_deltas)
        mean = np.mean(arr)
        if mean < 1e-6:
            return 100.0
        cv = np.std(arr) / mean   # coeficiente de variación
        score = max(0.0, 100.0 * (1.0 - min(cv / 3.0, 1.0)))
        return round(float(score), 2)

    @property
    def fast_movement_frames(self) -> int:
        """
        Número de frames donde algún joint superó el umbral de delta angular
        (_FAST_MOVE_THRESHOLD). Indica posibles errores de seguimiento
        en movimientos rápidos.
        """
        return self._fast_movement_frames

    @property
    def fast_movement_pct(self) -> float:
        """Porcentaje de frames con movimiento rápido / error de seguimiento."""
        total = len(self._kp_dropout_per_frame)
        if total == 0:
            return -1.0
        return round(self._fast_movement_frames / total * 100.0, 2)

    # ── Hilo background ───────────────────────────────────────────────────────

    def _resource_sampler(self):
        while not self._stop_bg.is_set():
            if _PSUTIL_OK:
                try:
                    raw = _PROC.cpu_percent(interval=None)
                    self._cpu_samples.append(min(100.0, raw / _CPU_COUNT))
                    self._ram_samples.append(
                        _PROC.memory_info().rss / (1024 ** 2)
                    )
                except Exception:
                    pass
            if _GPUTIL_OK:
                try:
                    gpus = _GPUtil.getGPUs()
                    if gpus:
                        self._gpu_samples.append(gpus[0].load * 100.0)
                except Exception:
                    pass
            self._stop_bg.wait(0.5)

    # ── Overlay ───────────────────────────────────────────────────────────────

    def draw_overlay(self, frame):
        if not self.overlay_visible:
            return frame

        import cv2
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        px, py = w - 245, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 4, py - 4),
                      (px + 240, py + 164), (8, 6, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px - 4, py - 4),
                      (px + 240, py + 164), (0, 215, 255), 1)

        mode_color = (0, 255, 140) if self.condition == "CPU" else (255, 180, 0)

        dropout  = self.kp_dropout_avg
        stab     = self.kp_stability
        ang_var  = self.angular_variability_avg
        traj     = self.trajectory_consistency

        lines = [
            (f"MODO:  {self.condition}",                          mode_color),
            (f"FPS:   {self.fps_avg:5.1f}  [{self.fps_min:.0f}-{self.fps_max:.0f}]",
                                                                   (0, 215, 255)),
            (f"CPU%:  {self.cpu_avg:5.1f}",                      (200, 200, 200)),
            (f"GPU%:  {self.gpu_avg:5.1f}",                      (200, 200, 200)),
            (f"RAM:   {self.ram_avg_mb:5.0f} MB",                (170, 170, 255)),
            (f"Inf:   {self.current_infer_ms:5.1f} ms",          (255, 220, 100)),
            ("── Pose ──────────────────",                        (60, 60, 80)),
            (f"KP loss: {dropout:4.1f}/17  ({self.kp_dropout_pct:.0f}%)",
                                                                   (255, 140, 80)),
            (f"Estab:   {stab:.3f}" if stab >= 0 else "Estab:   ---",
                                                                   (180, 255, 180)),
            (f"Ang var: {ang_var:.1f} deg" if ang_var >= 0 else "Ang var: ---",
                                                                   (180, 200, 255)),
            (f"Traj:    {traj:.1f}/100" if traj >= 0 else "Traj:    ---",
                                                                   (255, 200, 100)),
        ]

        cy = py + 16
        for text, color in lines:
            cv2.putText(frame, text, (px, cy), font, 0.42, color, 1, cv2.LINE_AA)
            cy += 16

        cv2.putText(frame, "[M] ocultar metricas", (px, cy),
                    font, 0.36, (70, 70, 95), 1, cv2.LINE_AA)
        return frame

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible

    # ── Cuestionario Likert ───────────────────────────────────────────────────

    def run_likert_survey(self, screen) -> dict:
        import pygame

        pygame.font.init()
        font_title = pygame.font.SysFont("Arial", 26, bold=True)
        font_q     = pygame.font.SysFont("Arial", 20)
        font_hint  = pygame.font.SysFont("Arial", 15)

        sw, sh  = screen.get_size()
        answers = {}
        current_q = 0
        clock     = pygame.time.Clock()

        ACCENT    = (0, 215, 255)
        BG        = (10, 8, 20)
        PANEL_BG  = (18, 14, 35)
        SELECTED  = (0, 180, 60)
        UNSEL     = (55, 50, 80)
        TEXT_MAIN = (230, 230, 255)
        TEXT_DIM  = (120, 115, 145)

        while current_q < len(LIKERT_QUESTIONS):
            screen.fill(BG)
            pw, ph = 620, 340
            px_p   = (sw - pw) // 2
            py_p   = (sh - ph) // 2
            pygame.draw.rect(screen, PANEL_BG, (px_p, py_p, pw, ph), border_radius=12)
            pygame.draw.rect(screen, ACCENT,   (px_p, py_p, pw, ph), 2, border_radius=12)

            title_surf = font_title.render("EVALUACION DE EXPERIENCIA", True, ACCENT)
            screen.blit(title_surf, (px_p + (pw - title_surf.get_width()) // 2, py_p + 18))

            prog_surf = font_hint.render(
                f"Pregunta {current_q + 1} de {len(LIKERT_QUESTIONS)}", True, TEXT_DIM)
            screen.blit(prog_surf, (px_p + (pw - prog_surf.get_width()) // 2, py_p + 52))

            bar_x = px_p + 30; bar_y = py_p + 72; bar_w = pw - 60; bar_h = 5
            pygame.draw.rect(screen, UNSEL, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * current_q / len(LIKERT_QUESTIONS))
            if fill_w > 0:
                pygame.draw.rect(screen, ACCENT, (bar_x, bar_y, fill_w, bar_h), border_radius=3)

            q_surf = font_q.render(LIKERT_QUESTIONS[current_q], True, TEXT_MAIN)
            screen.blit(q_surf, (px_p + (pw - q_surf.get_width()) // 2, py_p + 100))

            btn_w, btn_h = 80, 70; gap = 14
            total = 5 * btn_w + 4 * gap
            bx    = px_p + (pw - total) // 2
            by    = py_p + 150
            mx, my = pygame.mouse.get_pos()

            for i, val in enumerate(range(1, 6)):
                bx_i = bx + i * (btn_w + gap)
                rect = pygame.Rect(bx_i, by, btn_w, btn_h)
                is_hov  = rect.collidepoint(mx, my)
                already = answers.get(current_q) == val
                color   = SELECTED if already else (ACCENT if is_hov else UNSEL)
                pygame.draw.rect(screen, color, rect, border_radius=8)
                pygame.draw.rect(screen, ACCENT if (already or is_hov) else (80, 75, 110),
                                 rect, 2, border_radius=8)
                num_surf = font_title.render(str(val), True, TEXT_MAIN)
                screen.blit(num_surf, (bx_i + (btn_w - num_surf.get_width()) // 2, by + 6))
                lbl_surf = font_hint.render(LIKERT_LABELS[val], True,
                                            TEXT_MAIN if (already or is_hov) else TEXT_DIM)
                screen.blit(lbl_surf, (bx_i + (btn_w - lbl_surf.get_width()) // 2, by + 44))

            hint_surf = font_hint.render(
                "Haz clic en una opcion para continuar", True, TEXT_DIM)
            screen.blit(hint_surf,
                        (px_p + (pw - hint_surf.get_width()) // 2, py_p + 255))

            cond_col  = (0, 255, 140) if self.condition == "CPU" else (255, 180, 0)
            cond_surf = font_hint.render(
                f"Sesion: {self.condition}  |  {self.song}  |  {self.difficulty}",
                True, cond_col)
            screen.blit(cond_surf,
                        (px_p + (pw - cond_surf.get_width()) // 2, py_p + ph - 30))

            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.likert = answers; return answers
                if event.type == pygame.KEYDOWN:
                    for k, v in [(pygame.K_1,1),(pygame.K_2,2),(pygame.K_3,3),
                                 (pygame.K_4,4),(pygame.K_5,5),
                                 (pygame.K_KP1,1),(pygame.K_KP2,2),(pygame.K_KP3,3),
                                 (pygame.K_KP4,4),(pygame.K_KP5,5)]:
                        if event.key == k:
                            answers[current_q] = v; current_q += 1; break
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, val in enumerate(range(1, 6)):
                        bx_i = bx + i * (btn_w + gap)
                        if pygame.Rect(bx_i, by, btn_w, btn_h).collidepoint(event.pos):
                            answers[current_q] = val; current_q += 1; break

        self.likert = answers
        return answers

    # ── Guardar CSV ───────────────────────────────────────────────────────────

    def save(self):
        """
        Guarda todas las métricas + respuestas Likert en
        metrics/sesion_YYYYMMDD_HHMMSS.csv
        """
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath  = os.path.join(self.OUTPUT_DIR, f"sesion_{timestamp}.csv")

        session_duration = (
            self._session_end - self._session_start
            if self._session_end > 0 else 0.0
        )

        gpu_available = _GPUTIL_OK and bool(
            _GPUtil.getGPUs() if _GPUTIL_OK else []
        )

        row = {
            # ── Identificación ────────────────────────────────────────
            "timestamp":              timestamp,
            "participant_id":         self.participant_id,
            "participant_name":       self.participant_name,
            "condition":              self.condition,
            "song":                   self.song,
            "difficulty":             self.difficulty,
            "final_score":            self.final_score,
            "session_duration_s":     round(session_duration, 2),
            # ── FPS ───────────────────────────────────────────────────
            "fps_avg":                round(self.fps_avg, 2),
            "fps_min":                round(self.fps_min, 2),
            "fps_max":                round(self.fps_max, 2),
            # ── Latencia y procesamiento ──────────────────────────────
            "latency_avg_ms":         round(self.latency_avg_ms, 3),
            "inference_avg_ms":       round(self.inference_avg_ms, 3),
            "proc_total_avg_ms":      round(self.proc_total_avg_ms, 3),
            # ── Recursos ─────────────────────────────────────────────
            "cpu_avg_pct":            round(self.cpu_avg, 2),
            "gpu_avg_pct":            round(self.gpu_avg, 2),
            "gpu_available":          gpu_available,
            "ram_avg_mb":             round(self.ram_avg_mb, 1),
            # ── Frames ───────────────────────────────────────────────
            "total_frames":           len(self._frame_times),
            "dropped_frames":         self._dropped_frames,
            "video_dropped_frames":   self._video_dropped_frames,
            "camera_dropped_frames":  self._camera_dropped_frames,
            "late_frames":            self._late_frames,
            # ── Estimación de pose ────────────────────────────────────
            "pose_frames_logged":     len(self._kp_dropout_per_frame),
            "kp_dropout_avg":         round(self.kp_dropout_avg, 3),
            "kp_dropout_pct":         round(self.kp_dropout_pct, 2),
            "kp_stability":           round(self.kp_stability, 5),
            "angular_variability_avg_deg": round(self.angular_variability_avg, 3),
            "trajectory_consistency": round(self.trajectory_consistency, 2),
            "fast_movement_frames":   self.fast_movement_frames,
            "fast_movement_pct":      round(self.fast_movement_pct, 2),
            # ── Likert ────────────────────────────────────────────────
            "likert_fluidez":           self.likert.get(0, ""),
            "likert_facilidad":         self.likert.get(1, ""),
            "likert_precision_puntaje": self.likert.get(2, ""),
            "likert_confianza":         self.likert.get(3, ""),
            "likert_satisfaccion":      self.likert.get(4, ""),
        }

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        print(f"[METRICS] Sesion guardada: {filepath}")
        return filepath
