"""
just_dance_scores.py — Capa de datos: registros, perfil y estadisticas.
Todos los demás módulos importan desde aquí; nunca tocan los JSON directamente.
"""
from __future__ import annotations
import csv
import json, os, datetime


DATA_DIR        = "data"
HIGHSCORES_FILE = os.path.join(DATA_DIR, "highscores.json")
PROFILE_FILE    = os.path.join(DATA_DIR, "player_profile.json")

AVATAR_COLORS = [
    (  0, 215, 255),   # 0 cyan
    (255,  30, 130),   # 1 pink
    (  0, 255, 140),   # 2 green
    (255, 220,   0),   # 3 yellow
    (200,   0, 255),   # 4 purple
    (255, 100,  30),   # 5 orange
    (100, 140, 255),   # 6 blue
    (255,  80,  80),   # 7 red
]


def get_avatar_color(index: int) -> tuple:
    return AVATAR_COLORS[int(index) % len(AVATAR_COLORS)]


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ── Estrellas ──────────────────────────────────────────────────────────────────

def compute_stars(score: int, max_score: int) -> int:
    """0-5 estrellas segun porcentaje del max_score."""
    if max_score <= 0:
        return 0
    p = score / max_score
    if p >= 0.95: return 5
    if p >= 0.80: return 4
    if p >= 0.60: return 3
    if p >= 0.40: return 2
    if p >= 0.20: return 1
    return 0


def _read_csv_scores(path: str) -> list[int]:
    if not os.path.exists(path):
        return []
    scores = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                scores.append(int(float(row[0])))
            except (TypeError, ValueError):
                continue
    return scores


def get_current_score(path: str = "leaderboard.csv") -> int:
    scores = _read_csv_scores(path)
    return scores[-1] if scores else 0


def get_leaderboard_scores(path: str = "leaderboard.csv") -> list[int]:
    return sorted(_read_csv_scores(path), reverse=True)[:5]


# ── Highscores ─────────────────────────────────────────────────────────────────

def _hs_key(song_key: str, difficulty: str) -> str:
    return f"{song_key}_{difficulty.upper()}"


def load_highscores() -> dict:
    _ensure_dir()
    if not os.path.exists(HIGHSCORES_FILE):
        return {}
    try:
        with open(HIGHSCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print("[WARN] highscores.json corrupto — reiniciando vacío.")
        return {}


def _save_highscores(data: dict) -> None:
    _ensure_dir()
    with open(HIGHSCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_best_score(song_key: str, difficulty: str) -> int:
    entries = load_highscores().get(_hs_key(song_key, difficulty), [])
    return entries[0]["score"] if entries else 0


def get_best_stars(song_key: str, difficulty: str) -> int:
    entries = load_highscores().get(_hs_key(song_key, difficulty), [])
    return entries[0]["stars"] if entries else 0


def get_top_stars_for_song(song_key: str) -> int:
    """Maximas estrellas obtenidas en cualquier dificultad para este ejercicio."""
    hs, best = load_highscores(), 0
    for d in ("EASY", "NORMAL", "HARD", "EXTREME"):
        entries = hs.get(_hs_key(song_key, d), [])
        if entries:
            best = max(best, entries[0]["stars"])
    return best


def get_best_score_any_diff(song_key: str) -> int:
    """Mejor score global sin importar dificultad."""
    hs = load_highscores()
    best = 0
    for d in ("EASY", "NORMAL", "HARD", "EXTREME"):
        entries = hs.get(_hs_key(song_key, d), [])
        if entries:
            best = max(best, entries[0]["score"])
    return best


def get_top5_for_song(song_key: str, difficulty: str) -> list:
    return load_highscores().get(_hs_key(song_key, difficulty), [])


def add_score(song_key: str, difficulty: str, score: int, max_score: int,
              best_combo: int, perfects_pct: float, player_name: str) -> tuple:
    """
    Guarda score en top-5 para song+difficulty.
    Devuelve (is_new_record: bool, stars: int).
    """
    hs      = load_highscores()
    key     = _hs_key(song_key or "unknown", difficulty)
    stars   = compute_stars(score, max_score)
    entries = hs.get(key, [])
    prev    = entries[0]["score"] if entries else 0

    entries.append({
        "score":        score,
        "stars":        stars,
        "date":         datetime.datetime.now().isoformat(timespec="seconds"),
        "player":       str(player_name)[:20],
        "combo":        int(best_combo),
        "perfects_pct": round(float(perfects_pct), 1),
    })
    entries.sort(key=lambda x: x["score"], reverse=True)
    hs[key] = entries[:5]
    _save_highscores(hs)
    return (score > prev), stars


# ── Perfil del paciente ─────────────────────────────────────────────────────────

def load_profile() -> dict | None:
    _ensure_dir()
    if not os.path.exists(PROFILE_FILE):
        return None
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print("[WARN] player_profile.json corrupto — reiniciando.")
        return None


def save_profile(name: str, color_index: int) -> dict:
    """Crea o actualiza perfil conservando stats existentes."""
    _ensure_dir()
    ex = load_profile() or {}
    p  = {
        "name":          (name.strip()[:20] or "PACIENTE"),
        "color_index":   int(color_index) % 8,
        "created_at":    ex.get("created_at",
                         datetime.datetime.now().isoformat(timespec="seconds")),
        "games_played":  ex.get("games_played", 0),
        "total_seconds": ex.get("total_seconds", 0.0),
        "best_combo":    ex.get("best_combo", 0),
        "total_perfects":ex.get("total_perfects", 0),
    }
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)
    return p


def update_stats_after_game(duration_secs: float, best_combo: int,
                            perfect_count: int) -> None:
    """Actualiza datos del paciente al terminar una sesion."""
    p = load_profile()
    if not p:
        return
    p["games_played"]    = p.get("games_played", 0) + 1
    p["total_seconds"]   = p.get("total_seconds", 0.0) + float(duration_secs)
    p["best_combo"]      = max(p.get("best_combo", 0), best_combo)
    p["total_perfects"]  = p.get("total_perfects", 0) + int(perfect_count)
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


# ── Stats globales ─────────────────────────────────────────────────────────────

def get_global_stats() -> dict:
    p   = load_profile() or {}
    hs  = load_highscores()

    best_score, best_song = 0, None
    song_counts: dict[str, int] = {}
    all_entries: list[dict] = []

    for key, entries in hs.items():
        sk = key.rsplit("_", 1)[0]
        song_counts[sk] = song_counts.get(sk, 0) + len(entries)
        all_entries.extend(entries)
        if entries and entries[0]["score"] > best_score:
            best_score = entries[0]["score"]
            best_song  = sk

    most_played = (max(song_counts, key=song_counts.get)
                   if song_counts else None)
    all_entries.sort(key=lambda x: x["score"], reverse=True)

    return {
        "games_played":      p.get("games_played", 0),
        "total_seconds":     p.get("total_seconds", 0.0),
        "best_combo":        p.get("best_combo", 0),
        "total_perfects":    p.get("total_perfects", 0),
        "best_song_key":     best_song,
        "best_score_global": best_score,
        "most_played":       most_played,
        "top3":              all_entries[:3],
    }


# ── Difficulty Manager ─────────────────────────────────────────────────────────

class DifficultyManager:
    def __init__(self, difficulty: str = "NORMAL"):
        self.difficulty = difficulty.upper()
        self.set_difficulty(self.difficulty)

    def set_difficulty(self, difficulty: str):
        self.difficulty = difficulty.upper()
        if self.difficulty == "EASY":
            self.angle_threshold = 40.0
            self.score_multiplier = 0.8
            self.max_score = 10000
            self.movement_threshold = 5.0
            self.distinctiveness_threshold = 15.0
            self.angle_weight = 0.55
            self.vector_weight = 0.45
        elif self.difficulty == "HARD":
            self.angle_threshold = 15.0
            self.score_multiplier = 1.2
            self.max_score = 15000
            self.movement_threshold = 10.0
            self.distinctiveness_threshold = 10.0
            self.angle_weight = 0.35
            self.vector_weight = 0.65
        elif self.difficulty == "EXTREME":
            self.angle_threshold = 8.0
            self.score_multiplier = 1.5
            self.max_score = 20000
            self.movement_threshold = 12.0
            self.distinctiveness_threshold = 8.0
            self.angle_weight = 0.25
            self.vector_weight = 0.75
        else:  # NORMAL
            self.difficulty = "NORMAL"
            self.angle_threshold = 28.0
            self.score_multiplier = 1.0
            self.max_score = 12500
            self.movement_threshold = 8.0
            self.distinctiveness_threshold = 12.0
            self.angle_weight = 0.45
            self.vector_weight = 0.55

    def get_rating_points(self, rating: str) -> int:
        base_points = {
            "PERFECT": 100,
            "GREAT": 80,
            "GOOD": 60,
            "OK": 40,
            "MISS": 0
        }
        pts = base_points.get(rating.upper(), 0)
        return round(pts * self.score_multiplier)
