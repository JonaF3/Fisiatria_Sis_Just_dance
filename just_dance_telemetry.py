"""
just_dance_telemetry.py — v2
Monitor de rendimiento en tiempo real para rehabilitacion fisica.

Cambios v2:
  - Slot visual para inference_avg_ms (alimentado externamente via set_inference_ms)
  - Sin conflicto con SessionMetrics: son sistemas paralelos e independientes
  - CPU/GPU history thread sigue igual

Métricas:
  - CPU %      (uso del proceso actual, normalizado por núcleos)
  - GPU %      (GPUtil si disponible)
  - RAM MB     (memoria residente del proceso)
  - FPS        (frames por segundo del loop principal)
  - Frame time (ms por frame)
  - Inference  (ms, alimentado desde _run_inference_on vía set_inference_ms)

Tecla T: toggle visibilidad del panel.
"""

import collections
import time

try:
    import psutil
    import os as _os
    _PROC      = psutil.Process(_os.getpid())
    _CPU_COUNT = max(1, psutil.cpu_count(logical=True) or 1)
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    _PROC      = None
    _CPU_COUNT = 1

try:
    import GPUtil
    _GPUTIL_OK = True
except ImportError:
    _GPUTIL_OK = False

import cv2
import numpy as np


_PANEL_W   = 210
_GRAPH_H   = 32
_GRAPH_LEN = 60
_SAMPLE_N  = 5


class TelemetryMonitor:
    """
    Monitorea CPU, GPU, RAM, FPS e inferencia.
    Compatible con SessionMetrics: ambos pueden correr simultáneamente.
    """

    _WINDOW = 30

    def __init__(self):
        self._frame_times: collections.deque = collections.deque(maxlen=self._WINDOW)
        self._cpu_history: collections.deque = collections.deque(
            [0.0] * _GRAPH_LEN, maxlen=_GRAPH_LEN
        )
        self._gpu_history: collections.deque = collections.deque(
            [0.0] * _GRAPH_LEN, maxlen=_GRAPH_LEN
        )
        self._fps_history: collections.deque = collections.deque(
            [0.0] * _GRAPH_LEN, maxlen=_GRAPH_LEN
        )

        self._last_tick    = time.perf_counter()
        self._visible      = True
        self._frame_count  = 0

        # Slot para inference ms (actualizado externamente)
        self._last_infer_ms: float = 0.0

        if _PSUTIL_OK:
            psutil.cpu_percent(interval=None)
            _PROC.cpu_percent(interval=None)

        import threading
        self._stop_thread = threading.Event()
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def _monitor_loop(self):
        while not self._stop_thread.is_set():
            if _PSUTIL_OK:
                try:
                    raw = _PROC.cpu_percent(interval=None)
                    proc_cpu = min(100.0, raw / _CPU_COUNT)
                    self._cpu_history.append(proc_cpu)
                except Exception:
                    pass
            if _GPUTIL_OK:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self._gpu_history.append(gpus[0].load * 100.0)
                except Exception:
                    pass
            time.sleep(0.5)

    # ── API pública ────────────────────────────────────────────────────────────

    def tick(self):
        now = time.perf_counter()
        dt  = now - self._last_tick
        self._last_tick = now
        self._frame_times.append(dt)
        self._frame_count += 1
        if self._frame_count % _SAMPLE_N == 0:
            self._fps_history.append(self.fps)

    def toggle_visible(self):
        self._visible = not self._visible

    def set_inference_ms(self, ms: float):
        """
        Alimenta el último tiempo de inferencia medido (ms).
        Llamar desde el controller después de cada invoke si no se usa SessionMetrics.
        Si SessionMetrics está activo, este slot queda como referencia visual extra.
        """
        self._last_infer_ms = ms

    # ── Métricas calculadas ────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def frame_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return (sum(self._frame_times) / len(self._frame_times)) * 1000.0

    @property
    def cpu_percent(self) -> float:
        if not _PSUTIL_OK or not self._cpu_history:
            return -1.0
        recent = list(self._cpu_history)[-10:]
        return sum(recent) / len(recent)

    @property
    def gpu_percent(self) -> float:
        if not _GPUTIL_OK or not self._gpu_history:
            return -1.0
        recent = list(self._gpu_history)[-10:]
        return sum(recent) / len(recent)

    @property
    def ram_mb(self) -> float:
        if not _PSUTIL_OK:
            return -1.0
        try:
            return _PROC.memory_info().rss / (1024 ** 2)
        except Exception:
            return -1.0

    # ── Overlay en pantalla ────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self._visible:
            return frame

        fps  = self.fps
        fms  = self.frame_ms
        cpu  = self.cpu_percent
        gpu  = self.gpu_percent
        ram  = self.ram_mb

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad  = 8

        # Altura del panel: base + línea extra para inference si disponible
        has_infer = self._last_infer_ms > 0
        panel_h   = 340 if has_infer else 320
        panel_w   = _PANEL_W
        px, py    = 10, h - panel_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (px - 2, py - 2),
                      (px + panel_w, py + panel_h),
                      (8, 6, 18), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame,
                      (px - 2, py - 2),
                      (px + panel_w, py + panel_h),
                      (0, 130, 170), 1)

        cy = py + pad + 14

        cv2.putText(frame, "TELEMETRIA", (px + 4, cy),
                    font, 0.48, (0, 215, 255), 1, cv2.LINE_AA)
        cy += 20

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_col = _fps_color(fps)
        cv2.putText(frame, f"FPS  {fps:5.1f}", (px + 4, cy),
                    font, 0.44, fps_col, 1, cv2.LINE_AA)
        cy += 18
        cv2.putText(frame, f"Frame {fms:5.1f} ms", (px + 4, cy),
                    font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        cy += 18

        # Inferencia (si disponible)
        if has_infer:
            cv2.putText(frame, f"Infer {self._last_infer_ms:5.1f} ms", (px + 4, cy),
                        font, 0.40, (255, 220, 100), 1, cv2.LINE_AA)
            cy += 18

        cy += 2
        _draw_sparkline(frame, list(self._fps_history),
                        px + 4, cy, panel_w - 10, _GRAPH_H,
                        max_val=60.0, color=fps_col, label="")
        cy += _GRAPH_H + 6

        # ── CPU ───────────────────────────────────────────────────────────────
        cpu_col = _cpu_color(cpu) if cpu >= 0 else (150, 150, 150)
        label_cpu = f"CPU programa {cpu:4.1f}%" if cpu >= 0 else "CPU  n/d"
        cv2.putText(frame, label_cpu, (px + 4, cy),
                    font, 0.40, cpu_col, 1, cv2.LINE_AA)
        cy += 18

        _draw_sparkline(frame, list(self._cpu_history),
                        px + 4, cy, panel_w - 10, _GRAPH_H,
                        max_val=100.0, color=cpu_col, label="")
        cy += _GRAPH_H + 4
        _draw_bar(frame, cpu if cpu >= 0 else 0,
                  px + 4, cy, panel_w - 10, 8, cpu_col)
        cy += 14

        # ── GPU ───────────────────────────────────────────────────────────────
        gpu_col = _cpu_color(gpu) if gpu >= 0 else (150, 150, 150)
        label_gpu = f"GPU uso total {gpu:4.1f}%" if gpu >= 0 else "GPU  n/d"
        cv2.putText(frame, label_gpu, (px + 4, cy),
                    font, 0.40, gpu_col, 1, cv2.LINE_AA)
        cy += 18

        _draw_sparkline(frame, list(self._gpu_history),
                        px + 4, cy, panel_w - 10, _GRAPH_H,
                        max_val=100.0, color=gpu_col, label="")
        cy += _GRAPH_H + 4
        _draw_bar(frame, gpu if gpu >= 0 else 0,
                  px + 4, cy, panel_w - 10, 8, gpu_col)
        cy += 14

        # ── RAM ───────────────────────────────────────────────────────────────
        if ram >= 0:
            cv2.putText(frame, f"RAM  {ram:6.0f} MB", (px + 4, cy),
                        font, 0.40, (170, 170, 255), 1, cv2.LINE_AA)
            cy += 18

        cv2.putText(frame, "[T] ocultar", (px + 4, cy),
                    font, 0.36, (70, 70, 95), 1, cv2.LINE_AA)

        return frame


# ── Funciones auxiliares de dibujo ────────────────────────────────────────────

def _draw_sparkline(frame, values, x, y, w, h, max_val, color, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 12, 32), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 40, 80), 1)
    mid_y = y + h // 2
    for gx in range(x, x + w, 4):
        frame[mid_y, min(gx, frame.shape[1] - 1)] = (40, 35, 60)
    n = len(values)
    if n < 2:
        return
    pts = []
    for i, v in enumerate(values):
        norm  = max(0.0, min(1.0, v / max_val)) if max_val > 0 else 0.0
        px_   = x + int(i * w / (n - 1))
        py_   = y + h - int(norm * (h - 2)) - 1
        pts.append((px_, py_))
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 2, color, -1)


def _draw_bar(frame, value, x, y, w, h, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 12, 32), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 40, 80), 1)
    fill = int(w * max(0.0, min(100.0, value)) / 100.0)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)


def _fps_color(fps):
    if fps >= 50: return (0, 220, 100)
    if fps >= 30: return (0, 200, 255)
    return (60, 60, 255)


def _cpu_color(cpu):
    if cpu < 50:  return (0, 220, 100)
    if cpu < 75:  return (0, 200, 255)
    if cpu < 90:  return (0, 200, 255)
    return (60, 60, 255)
