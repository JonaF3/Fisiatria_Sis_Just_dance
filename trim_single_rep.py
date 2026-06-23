"""
trim_single_rep.py  — v4

Detecta una repetición completa en un video de ejercicio y lo recorta.
No requiere especificar el tipo de ejercicio — funciona con cualquier
movimiento (hombro, pierna, cuello, espalda, mano, etc.).

Uso:
    python trim_single_rep.py <video_path> [output_path]
    python trim_single_rep.py --batch <carpeta_videos>
"""

from __future__ import annotations
import sys, os, subprocess
import numpy as np
import cv2

PADDING_S  = 0.8
MIN_REP_S  = 1.0
SMOOTH_W   = 0.5
LO_THRESH  = 0.30
HI_THRESH  = 0.55


def motion_signal(video_path):
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vals, prev = [], None
    while True:
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (320, 240))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)
        vals.append(float(np.mean(np.abs(gray - prev))) if prev is not None else 0.0)
        prev = gray
    cap.release()
    sig = np.array(vals)
    w   = max(3, int(fps * SMOOTH_W) | 1)
    s   = np.convolve(sig, np.ones(w)/w, mode='same')
    rng = s.max() - s.min()
    s_n = (s - s.min()) / rng if rng > 0.01 else np.zeros_like(s)
    ts  = np.arange(len(s_n)) / fps
    return s_n, ts, fps, float(ts[-1]) if len(ts) else 0.0


def detect_rep(s_norm, ts, dur):
    for lo, hi in [(0.20, 0.55), (0.30, 0.55), (0.40, 0.50)]:
        state = "rest"
        rest_start = 0
        for i in range(1, len(s_norm)):
            val = s_norm[i]
            if state == "rest":
                if val > hi:
                    state = "moving"
            elif state == "moving":
                if val < lo:
                    elapsed = ts[i] - ts[rest_start]
                    if elapsed >= MIN_REP_S:
                        return max(0.0, ts[rest_start] - PADDING_S), min(dur, ts[i] + PADDING_S)
                    else:
                        state = "rest"
                        rest_start = i
        if state == "moving" and (dur - ts[rest_start]) >= MIN_REP_S:
            return max(0.0, ts[rest_start] - PADDING_S), dur
    return 0.0, dur


def trim_video(inp, out, t0, t1):
    cmd = ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-i", inp,
           "-t", f"{t1-t0:.3f}", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ffmpeg] {r.stderr[-200:]}")
        return False
    return True


def process_video(inp, out=None):
    if not os.path.exists(inp):
        print(f"  ERROR: No existe {inp}"); return False
    if out is None:
        base, ext = os.path.splitext(inp)
        out = f"{base}_1rep{ext}"
    s_norm, ts, fps, dur = motion_signal(inp)
    t0, t1 = detect_rep(s_norm, ts, dur)
    print(f"  Rep: {t0:.2f}s → {t1:.2f}s  ({t1-t0:.2f}s de {dur:.1f}s total)")
    ok = trim_video(inp, out, t0, t1)
    if ok: print(f"  OK: {os.path.basename(out)}")
    return ok


def process_batch(folder):
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    videos = sorted([f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in exts and "_1rep" not in f])
    if not videos:
        print("No se encontraron videos."); return
    print(f"\nProcesando {len(videos)} videos...\n{'='*55}")
    ok_n = fail_n = 0
    for fname in videos:
        inp = os.path.join(folder, fname)
        base, ext = os.path.splitext(fname)
        out = os.path.join(folder, f"{base}_1rep{ext}")
        print(f"\n{fname}")
        if process_video(inp, out): ok_n += 1
        else: fail_n += 1
    print(f"\n{'='*55}\nListo: {ok_n} OK  |  {fail_n} fallidos")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "--batch":
        process_batch(sys.argv[2] if len(sys.argv) > 2 else ".")
    else:
        ok = process_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        sys.exit(0 if ok else 1)
