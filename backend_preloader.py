"""
backend_preloader.py
====================
Pre-carga los backends de MediaPipe en un hilo de fondo mientras el menu
del lobby esta visible, para que el usuario no espere al seleccionar un ejercicio.

Uso en just_dance_gui.py (al inicio del lobby):
    import backend_preloader
    backend_preloader.preload()

Uso en just_dance_controller.py (al iniciar ejercicio):
    import backend_preloader
    self.pose33_backend = backend_preloader.get_pose33() or MediaPipe33Backend(...)
"""

from __future__ import annotations

import threading

_pose33 = None
_hand21 = None
_lock = threading.Lock()
_started = False


def preload():
    """Lanza la pre-carga en un hilo daemon. Llama una sola vez."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_load_all, daemon=True, name="BackendPreloader")
    t.start()


def _load_all():
    global _pose33, _hand21
    try:
        from pose_backends.mediapipe33_backend import MediaPipe33Backend
        b = MediaPipe33Backend('model/pose_landmarker_lite.task')
        with _lock:
            _pose33 = b
        print('[PRELOAD] MediaPipe Pose33 listo.')
    except Exception as e:
        print(f'[PRELOAD] Error cargando Pose33: {e}')

    try:
        from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend
        b = MediaPipeHand21Backend('model/hand_landmarker.task', num_hands=1)
        with _lock:
            _hand21 = b
        print('[PRELOAD] MediaPipe Hand21 listo.')
    except Exception as e:
        print(f'[PRELOAD] Error cargando Hand21: {e}')


def get_pose33():
    """Retorna el backend Pose33 pre-cargado, o None si todavia no esta listo."""
    with _lock:
        return _pose33


def get_hand21():
    """Retorna el backend Hand21 pre-cargado, o None si todavia no esta listo."""
    with _lock:
        return _hand21


def is_preloaded_pose33(backend):
    """Retorna True si el backend dado es el que maneja el preloader (no cerrar)."""
    with _lock:
        return backend is not None and backend is _pose33


def is_preloaded_hand21(backend):
    """Retorna True si el backend dado es el que maneja el preloader (no cerrar)."""
    with _lock:
        return backend is not None and backend is _hand21
