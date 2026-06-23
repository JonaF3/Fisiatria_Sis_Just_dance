from __future__ import annotations
"""
just_dance_gui.py — v3 (UI overhaul)
Lobby con artwork generado, chip de jugador, pantallas de perfil y stats.
"""
import sys, math, random, json, os, platform, time, threading
import pygame, cv2

from just_dance_main   import run_game
from just_dance_gui_analytics import run_analytics
from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
from just_dance_score import (
    load_profile, save_profile, get_global_stats,
    get_top_stars_for_song, get_best_score, get_best_stars,
    get_best_score_any_diff, get_top5_for_song,
    AVATAR_COLORS, get_avatar_color, compute_stars,
)

# Generado dinámicamente desde REHAB_EXERCISE_CONFIGS
SONGS = [
    (cfg["name"], key)
    for key, cfg in REHAB_EXERCISE_CONFIGS.items()
]

SONG_ARTISTS = {
    key: cfg.get("artist", "Rehabilitacion fisica")
    for key, cfg in REHAB_EXERCISE_CONFIGS.items()
}

SONG_COLORS = {
    key: cfg.get("color", (100, 100, 150))
    for key, cfg in REHAB_EXERCISE_CONFIGS.items()
}

SONG_COMPLEXITY = {
    key: cfg.get("complexity", "MEDIO")
    for key, cfg in REHAB_EXERCISE_CONFIGS.items()
}

SONG_ACCENTS = [
    cfg.get("color", (100, 100, 150))
    for cfg in REHAB_EXERCISE_CONFIGS.values()
]

# ── Colores base ───────────────────────────────────────────────────────────────
BG_DARK      = (8,  4, 18)
NEON_CYAN    = (0,  230, 255)
NEON_PINK    = (255, 30, 130)
NEON_YELLOW  = (255, 220, 0)
NEON_GREEN   = (0,  255, 140)
WHITE        = (255, 255, 255)
GRAY         = (140, 130, 160)
DARK_CARD    = (22,  14, 45)
DARKER_CARD  = (16,  10, 35)


# BAJO:  movimientos lentos, simetricos, poca variacion corporal
# MEDIO: cambios de brazos, piernas, desplazamientos moderados
# ALTO:  movimientos rapidos, cambios de orientacion, coordinacion exigente


COMPLEXITY_COLORS = {
    "BAJO":  ( 0, 220, 120),   # verde
    "MEDIO": (255, 200,   0),  # amarillo
    "ALTO":  (255,  50,  80),  # rojo
}

COMPLEXITY_LABELS = {
    "BAJO": "BAJO - Controlado",
    "MEDIO": "MEDIO - Coordinacion",
    "ALTO": "ALTO - Mayor rango",
}




RESOLUTIONS = [
    (1280, 720,  "1280 × 720  (HD)"),
    (1600, 900,  "1600 × 900  (HD+)"),
    (1920, 1080, "1920 × 1080 (Full HD)"),
]

PREVIEW_START_MS   = 20000
PREVIEW_DURATION   = 6000
PREVIEW_STOP_EVENT = pygame.USEREVENT + 1

CHAR_COLORS = [(255,80,80),(0,220,220),(255,200,0),(180,80,255)]

# ── Artwork generado en código ─────────────────────────────────────────────────
_artwork_cache: dict = {}

def _generate_artwork(song_key: str, song_name: str, w: int, h: int) -> "pygame.Surface":
    color = SONG_COLORS.get(song_key, (100,100,150))
    surf  = pygame.Surface((w, h))
    surf.fill(color)

    # Glow radial suave
    ov = pygame.Surface((w, h), pygame.SRCALPHA)
    light = tuple(min(255, c + 90) for c in color)
    for r in range(min(w,h)//2, 4, -18):
        a = int(38 * (1 - r / (min(w,h)//2)))
        pygame.draw.circle(ov, (*light, a), (w//2, int(h*0.42)), r)
    surf.blit(ov, (0,0))

    # Degradado oscuro en el tercio inferior
    grad = pygame.Surface((w, h//3), pygame.SRCALPHA)
    dark = tuple(max(0, c//2) for c in color)
    grad.fill((*dark, 210))
    surf.blit(grad, (0, h - h//3))

    # Nota musical (círculo + línea + dos banderines)
    wh = (255,255,255)
    cx, cy = w//2, int(h * 0.38)
    r2 = max(7, min(w,h)//9)
    pygame.draw.ellipse(surf, wh, (cx - r2, cy + r2//2, r2*2, int(r2*1.35)))
    stx = cx + r2 - 3
    pygame.draw.line(surf, wh, (stx, cy + r2), (stx, cy - r2*3), max(2, r2//4))
    for fy in (cy - r2*3, cy - r2*2):
        pygame.draw.line(surf, wh, (stx, fy), (stx + r2 + 4, fy + r2), max(2, r2//5))

    # Nombre de la canción
    fs = max(11, min(19, w//16))
    nf = pygame.font.SysFont("impact", fs)
    ns = nf.render(song_name.upper(), True, wh)
    if ns.get_width() > w - 16:
        ns = pygame.transform.smoothscale(ns, (w-16, ns.get_height()))
    surf.blit(ns, (w//2 - ns.get_width()//2, h - h//5))
    return surf


def load_song_artwork(song_key: str, song_name: str, w: int, h: int) -> "pygame.Surface":
    """Carga artwork desde PNG o genera en código. Cachea el resultado."""
    key = (song_key, w, h)
    if key in _artwork_cache:
        return _artwork_cache[key]
    path = os.path.join("songs_covers", f"{song_key}.png")
    if os.path.exists(path):
        try:
            img = pygame.image.load(path).convert()
            img = pygame.transform.smoothscale(img, (w, h))
            _artwork_cache[key] = img
            return img
        except Exception:
            pass
    surf = _generate_artwork(song_key, song_name, w, h)
    _artwork_cache[key] = surf
    return surf


# ── Helpers visuales ───────────────────────────────────────────────────────────

def _draw_star(surface, cx, cy, r, color, filled=True):
    r_in = max(1, int(r * 0.42))
    pts  = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        rad   = r if i % 2 == 0 else r_in
        pts.append((int(cx + rad*math.cos(angle)), int(cy + rad*math.sin(angle))))
    if filled:
        pygame.draw.polygon(surface, color, pts)
    else:
        pygame.draw.polygon(surface, color, pts, 1)


def _draw_stars_row(surface, x, y, n_filled, total=5, r=9, gap=22):
    for i in range(total):
        cx = x + i * gap
        if i < n_filled:
            _draw_star(surface, cx, y, r, (255,215,0), filled=True)
            _draw_star(surface, cx, y, r, (255,255,150), filled=False)
        else:
            _draw_star(surface, cx, y, r, (45,38,65), filled=True)
            _draw_star(surface, cx, y, r, (75,65,100), filled=False)


def _draw_avatar(surface, cx, cy, r, color_idx, name=""):
    color    = get_avatar_color(color_idx)
    initials = "".join(w[0] for w in name.split() if w)[:2].upper() or "?"
    pygame.draw.circle(surface, color, (cx, cy), r)
    pygame.draw.circle(surface, WHITE, (cx, cy), r, 2)
    fs   = max(10, int(r * 1.1))
    fnt  = pygame.font.SysFont("impact", fs)
    txt  = fnt.render(initials, True, (15,8,30))
    surface.blit(txt, (cx - txt.get_width()//2, cy - txt.get_height()//2))


class Particle:
    def __init__(self, W, H):
        self.W = W; self.H = H; self.reset()
    def reset(self):
        self.x     = random.randint(0, self.W)
        self.y     = random.randint(0, self.H)
        self.size  = random.uniform(1, 3)
        self.speed = random.uniform(0.2, 0.8)
        self.alpha = random.randint(40, 180)
        self.color = random.choice([NEON_CYAN, NEON_PINK, NEON_YELLOW, NEON_GREEN])
        self.drift = random.uniform(-0.3, 0.3)
    def update(self):
        self.y -= self.speed; self.x += self.drift; self.alpha -= 0.4
        if self.y < 0 or self.alpha <= 0:
            self.reset(); self.y = self.H
    def draw(self, surface):
        if self.alpha <= 0: return
        s = pygame.Surface((int(self.size*2), int(self.size*2)), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, int(self.alpha)),
                           (int(self.size), int(self.size)), int(self.size))
        surface.blit(s, (int(self.x-self.size), int(self.y-self.size)))


def draw_glow_text(surface, text, font, color, x, y, glow_radius=8, center=True):
    for r in range(glow_radius, 0, -2):
        gs = font.render(text, True, color)
        gs.set_alpha(int(60*(1-r/glow_radius)))
        gb = pygame.transform.scale(gs, (gs.get_width()+r*2, gs.get_height()+r*2))
        bx = x-gb.get_width()//2 if center else x-r
        by = y-gb.get_height()//2 if center else y-r
        surface.blit(gb, (bx, by))
    ms  = font.render(text, True, color)
    mx2 = x-ms.get_width()//2 if center else x
    my2 = y-ms.get_height()//2 if center else y
    surface.blit(ms, (mx2, my2))


def draw_rounded_rect(surface, color, rect, radius, alpha=255, border=0, border_color=None):
    w, h = max(1, rect[2]), max(1, rect[3])
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0,0,w,h), border_radius=radius)
    if border and border_color:
        pygame.draw.rect(s, (*border_color, alpha), (0,0,w,h), width=border, border_radius=radius)
    surface.blit(s, (rect[0], rect[1]))


def apply_always_on_top():
    if platform.system() != "Windows": return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, "Rehabilitacion Fisica")
        if hwnd:
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0,0,0,0, 0x0002|0x0001)
    except Exception: pass


def play_song_preview(song_key, volume=0.8):
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(f"songs_audio/{song_key}.mp3")
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(start=PREVIEW_START_MS/1000.0)
        pygame.time.set_timer(PREVIEW_STOP_EVENT, PREVIEW_DURATION)
    except Exception: pass


def load_song_config(song_key):
    path = os.path.join("songs_beats", f"{song_key}_beats.json")
    if not os.path.exists(path): return None
    try:
        with open(path,"r",encoding="utf-8") as f: data = json.load(f)
        return {"players": data.get("players",1), "characters": data.get("characters",[])}
    except Exception: return None


# ── Video preview (lobby / dificultad) ────────────────────────────────────────
VIDEO_PREVIEW_CACHE   = {"surface": None}
CURRENT_PREVIEW_SONG  = None

def _video_preview_loop(song_key, target_w, target_h):
    global CURRENT_PREVIEW_SONG
    extensions = [".mp4",".mov",".avi",".mkv"]
    path = None
    for ext in extensions:
        p = os.path.join("songs", f"{song_key}{ext}")
        if os.path.exists(p): path = p; break
    if not path: return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*15))
    while CURRENT_PREVIEW_SONG == song_key:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*15)); continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh,fw = frame.shape[:2]
        scale = min(target_w/fw, target_h/fh)
        sw,sh = int(fw*scale), int(fh*scale)
        frame = cv2.resize(frame,(sw,sh))
        surf  = pygame.surfarray.make_surface(frame.swapaxes(0,1))
        VIDEO_PREVIEW_CACHE["surface"] = surf
        time.sleep(max(0, 1/fps - (time.time()-t0)))
    cap.release()

def get_video_preview_surface(song_key, target_w, target_h):
    global CURRENT_PREVIEW_SONG
    if CURRENT_PREVIEW_SONG != song_key:
        CURRENT_PREVIEW_SONG = song_key
        VIDEO_PREVIEW_CACHE["surface"] = None
        threading.Thread(target=_video_preview_loop,
                         args=(song_key,target_w,target_h), daemon=True).start()
    return VIDEO_PREVIEW_CACHE.get("surface")


# ── Pantalla de Perfil ──────────────────────────────────────────────────────────

def run_profile_screen(screen, clock, W, H, first_run=False):
    """
    Pantalla de edición de perfil.
    Devuelve True si se guardó, False/None si se canceló (solo posible si !first_run).
    """
    profile      = load_profile() or {"name":"", "color_index":0}
    player_name  = profile.get("name","")
    color_idx    = profile.get("color_index", 0)
    particles    = [Particle(W,H) for _ in range(80)]

    font_title = pygame.font.SysFont("impact",       52)
    font_label = pygame.font.SysFont("trebuchetms",  18, bold=True)
    font_input = pygame.font.SysFont("trebuchetms",  36, bold=True)
    font_hint  = pygame.font.SysFont("trebuchetms",  15)
    font_btn   = pygame.font.SysFont("impact",       28)

    tick = 0; error_msg = ""

    while True:
        clock.tick(60); tick += 1
        mx, my = pygame.mouse.get_pos()
        pulse  = math.sin(tick*0.05)*0.5+0.5

        # ── Layout ─────────────────────────────────────────────────────
        CX     = W//2
        AV_Y   = 160
        INP_Y  = AV_Y + 110
        COL_Y  = INP_Y + 90
        SAVE_Y = COL_Y + 90

        inp_rect  = pygame.Rect(CX-220, INP_Y-12, 440, 52)
        save_rect = pygame.Rect(CX-120, SAVE_Y, 240, 52)
        back_rect = pygame.Rect(CX-80, SAVE_Y+64, 160, 44)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if not first_run: return False
                elif event.key == pygame.K_RETURN:
                    if player_name.strip():
                        save_profile(player_name.strip(), color_idx)
                        return True
                    else:
                        error_msg = "Ingresa un nombre"
                elif event.key == pygame.K_BACKSPACE:
                    player_name = player_name[:-1]; error_msg = ""
                elif len(player_name) < 16:
                    ch = event.unicode
                    if ch.isalnum() or ch in " _-":
                        player_name += ch.upper(); error_msg = ""
            if event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
                # Color circles
                for i in range(8):
                    ccx = CX - (8*40)//2 + i*40 + 20
                    ccy = COL_Y + 20
                    if math.hypot(mx-ccx, my-ccy) <= 18:
                        color_idx = i
                if save_rect.collidepoint(mx,my):
                    if player_name.strip():
                        save_profile(player_name.strip(), color_idx)
                        return True
                    else:
                        error_msg = "Ingresa un nombre"
                if back_rect.collidepoint(mx,my) and not first_run:
                    return False

        # ── Dibujo ─────────────────────────────────────────────────────
        screen.fill(BG_DARK)
        for gx in range(0,W,60): pygame.draw.line(screen,(30,18,55),(gx,0),(gx,H),1)
        for gy in range(0,H,60): pygame.draw.line(screen,(30,18,55),(0,gy),(W,gy),1)
        for p in particles: p.update(); p.draw(screen)

        tc = (int(NEON_CYAN[0]+(NEON_PINK[0]-NEON_CYAN[0])*pulse),
              int(NEON_CYAN[1]+(NEON_PINK[1]-NEON_CYAN[1])*pulse),
              int(NEON_CYAN[2]+(NEON_PINK[2]-NEON_CYAN[2])*pulse))

        title = "BIENVENIDO — CREA TU PERFIL" if first_run else "EDITAR PERFIL"
        draw_glow_text(screen, title, font_title, tc, CX, 70, glow_radius=10)
        pygame.draw.line(screen, NEON_CYAN, (CX-200,100),(CX+200,100),2)

        # Avatar grande
        _draw_avatar(screen, CX, AV_Y, 44, color_idx, player_name or "?")

        # Input nombre
        draw_glow_text(screen,"NOMBRE DE JUGADOR", font_label, GRAY, CX, INP_Y-28)
        is_hov = inp_rect.collidepoint(mx,my)
        draw_rounded_rect(screen,(14,9,28), inp_rect, 10,
                          border=2, border_color=NEON_CYAN if is_hov else (60,45,90))
        ns = font_input.render(player_name, True, WHITE)
        screen.blit(ns, (inp_rect.centerx-ns.get_width()//2,
                          inp_rect.centery-ns.get_height()//2))
        if (tick//28)%2==0:
            cx2 = inp_rect.centerx+ns.get_width()//2+5
            pygame.draw.line(screen,WHITE,(cx2,inp_rect.y+10),(cx2,inp_rect.y+42),2)
        if error_msg:
            ef = pygame.font.SysFont("trebuchetms",13)
            es = ef.render(error_msg, True, (255,80,80))
            screen.blit(es,(inp_rect.centerx-es.get_width()//2, inp_rect.bottom+4))

        # Selector de color
        draw_glow_text(screen,"COLOR DE AVATAR", font_label, GRAY, CX, COL_Y-10)
        for i in range(8):
            ccx = CX-(8*40)//2+i*40+20
            ccy = COL_Y+20
            col = get_avatar_color(i)
            selected = (i == color_idx)
            pygame.draw.circle(screen, col, (ccx,ccy), 18)
            if selected:
                pygame.draw.circle(screen, WHITE, (ccx,ccy), 18, 3)
                pygame.draw.circle(screen, col,   (ccx,ccy),  9)
            else:
                pygame.draw.circle(screen, (60,50,80), (ccx,ccy), 18, 2)

        # Botón guardar
        sh2 = save_rect.collidepoint(mx,my)
        av  = get_avatar_color(color_idx)
        draw_rounded_rect(screen, av if sh2 else DARK_CARD, save_rect, 14,
                          alpha=255 if sh2 else 200, border=2, border_color=av)
        draw_glow_text(screen,"GUARDAR [ENTER]", font_btn, WHITE,
                       save_rect.centerx, save_rect.centery, glow_radius=4 if sh2 else 1)

        if not first_run:
            bh2 = back_rect.collidepoint(mx,my)
            draw_rounded_rect(screen, DARKER_CARD, back_rect, 10, alpha=200,
                              border=2, border_color=NEON_CYAN if bh2 else (60,45,90))
            draw_glow_text(screen,"< VOLVER", font_hint, NEON_CYAN if bh2 else GRAY,
                           back_rect.centerx, back_rect.centery)
        else:
            draw_glow_text(screen,"Ingresa tu nombre para continuar",
                           font_hint, GRAY, CX, SAVE_Y+118)

        pygame.display.flip()


# ── Pantalla de Estadísticas ───────────────────────────────────────────────────

def run_stats_screen(screen, clock, W, H):
    profile    = load_profile() or {}
    stats      = get_global_stats()
    particles  = [Particle(W,H) for _ in range(70)]

    font_title = pygame.font.SysFont("impact",      48)
    font_val   = pygame.font.SysFont("impact",      42)
    font_label = pygame.font.SysFont("trebuchetms", 14, bold=True)
    font_hint  = pygame.font.SysFont("trebuchetms", 14)
    font_name  = pygame.font.SysFont("impact",      22)

    tick   = 0
    name   = profile.get("name","PACIENTE")
    cidx   = profile.get("color_index",0)
    total_h = stats["total_seconds"]
    hrs    = int(total_h//3600)
    mins   = int((total_h%3600)//60)

    # Nombre amigable de la canción más jugada
    def friendly(sk):
        if not sk: return "---"
        for n,k in SONGS:
            if k == sk: return n
        return sk.upper()

    while True:
        clock.tick(60); tick += 1
        mx, my = pygame.mouse.get_pos()
        pulse  = math.sin(tick*0.05)*0.5+0.5

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        screen.fill(BG_DARK)
        for gx in range(0,W,60): pygame.draw.line(screen,(30,18,55),(gx,0),(gx,H),1)
        for gy in range(0,H,60): pygame.draw.line(screen,(30,18,55),(0,gy),(W,gy),1)
        for p in particles: p.update(); p.draw(screen)

        tc = (int(NEON_CYAN[0]+(NEON_PINK[0]-NEON_CYAN[0])*pulse),
              int(NEON_CYAN[1]+(NEON_PINK[1]-NEON_CYAN[1])*pulse),
              int(NEON_CYAN[2]+(NEON_PINK[2]-NEON_CYAN[2])*pulse))

        draw_glow_text(screen,"ESTADÍSTICAS", font_title, tc, W//2, 52, glow_radius=10)
        pygame.draw.line(screen, NEON_CYAN,(W//2-200,80),(W//2+200,80),2)

        # Chip de jugador
        _draw_avatar(screen, W//2-110, 110, 28, cidx, name)
        ns2 = font_name.render(name.upper(), True, WHITE)
        screen.blit(ns2,(W//2-110+36, 110-ns2.get_height()//2))

        # ── Grid de 4 tarjetas ────────────────────────────────────────
        CARD_W, CARD_H = 260, 110
        gap = 20
        total_cw = 2*(CARD_W+gap)-gap
        cx0 = W//2 - total_cw//2
        cy0 = 160

        card_data = [
            ("PARTIDAS",    str(stats["games_played"]),         NEON_CYAN,  0, 0),
            ("TIEMPO",      f"{hrs}h {mins:02d}m",              NEON_GREEN, 1, 0),
            ("MEJOR COMBO", f"×{stats['best_combo']}",          NEON_YELLOW,0, 1),
            ("TOTAL PERFECT",str(stats["total_perfects"]),      NEON_PINK,  1, 1),
        ]

        for label, val, col, ci, ri in card_data:
            rx = cx0 + ci*(CARD_W+gap)
            ry = cy0 + ri*(CARD_H+gap)
            draw_rounded_rect(screen, DARK_CARD,(rx,ry,CARD_W,CARD_H),
                              16, alpha=230, border=2, border_color=col)
            lf = pygame.font.SysFont("trebuchetms",12,bold=True)
            ls = lf.render(label,True,col)
            screen.blit(ls,(rx+14,ry+12))
            vf = pygame.font.SysFont("impact",38)
            vs = vf.render(val, True, WHITE)
            screen.blit(vs,(rx+CARD_W//2-vs.get_width()//2, ry+CARD_H//2-vs.get_height()//2+8))

        # ── Mejor canción ─────────────────────────────────────────────
        best_y = cy0 + 2*(CARD_H+gap) + 10
        best_song_name = friendly(stats.get("best_song_key"))
        best_sc        = stats.get("best_score_global",0)

        draw_rounded_rect(screen, DARK_CARD,
                          (cx0, best_y, total_cw, 70), 12, alpha=220,
                          border=2, border_color=NEON_YELLOW)
        bf1 = pygame.font.SysFont("trebuchetms",13,bold=True)
        bf2 = pygame.font.SysFont("impact",26)
        screen.blit(bf1.render("MEJOR CANCIÓN", True, NEON_YELLOW),(cx0+14, best_y+10))
        bsn = bf2.render(f"{best_song_name}  —  {best_sc:,} pts", True, WHITE)
        screen.blit(bsn,(cx0+14, best_y+30))

        # ── Top-3 global ──────────────────────────────────────────────
        top_y = best_y + 90
        draw_glow_text(screen,"TOP 3 GLOBAL", font_label, NEON_PINK,
                       W//2, top_y, glow_radius=4)
        top3 = stats.get("top3",[])
        rank_colors = [(255,200,0),(180,180,200),(200,120,60)]
        for i, entry in enumerate(top3[:3]):
            ey = top_y + 20 + i*44
            rc = rank_colors[i]
            draw_rounded_rect(screen, DARK_CARD,
                              (cx0, ey, total_cw, 38), 8, alpha=200,
                              border=1, border_color=rc)
            draw_rounded_rect(screen, rc, (cx0, ey, 6, 38), 3)
            rkf = pygame.font.SysFont("impact",20)
            screen.blit(rkf.render(f"#{i+1}", True, rc),(cx0+14,ey+9))
            pf  = pygame.font.SysFont("trebuchetms",16,bold=True)
            screen.blit(pf.render(entry.get("player","?"),True,WHITE),(cx0+55,ey+10))
            sf  = pygame.font.SysFont("impact",20)
            sc2 = sf.render(f"{entry['score']:,} pts",True,NEON_YELLOW)
            screen.blit(sc2,(cx0+total_cw-sc2.get_width()-14,ey+9))

        if not top3:
            draw_glow_text(screen,"Sin partidas registradas aún",
                           font_hint, GRAY, W//2, top_y+50)

        hf = font_hint.render("ESC = volver al lobby", True, (70,60,100))
        screen.blit(hf,(W//2-hf.get_width()//2, H-28))
        pygame.display.flip()


# ── Pantalla de selección de personaje ────────────────────────────────────────
def select_character_screen(screen, clock, song_key, song_name, accent, W, H):
    config = load_song_config(song_key)
    if not config or config["players"] < 2:
        return {"mode":"single","char_idx":0,"char_info":None}
    characters  = config["characters"]
    particles   = [Particle(W,H) for _ in range(80)]
    font_title  = pygame.font.SysFont("impact",60)
    font_sub    = pygame.font.SysFont("impact",28)
    font_hint   = pygame.font.SysFont("trebuchetms",16)
    font_btn    = pygame.font.SysFont("impact",30)
    phase = 1; tick = 0
    while True:
        clock.tick(60); tick += 1
        mx,my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.display.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key==pygame.K_ESCAPE: return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if phase==1:
                    btn1=pygame.Rect(W//2-280,H//2,240,80)
                    btn2=pygame.Rect(W//2+40,H//2,240,80)
                    if btn1.collidepoint(mx,my): phase=2
                    elif btn2.collidepoint(mx,my): return {"mode":"duo","char_idx":None,"char_info":None}
                elif phase==2:
                    for i,char in enumerate(characters):
                        bx=W//2-(len(characters)*150)//2+i*160
                        btn=pygame.Rect(bx,H//2,140,180)
                        if btn.collidepoint(mx,my):
                            return {"mode":"single","char_idx":i,"char_info":char}
                    back=pygame.Rect(W//2-80,H-90,160,50)
                    if back.collidepoint(mx,my): phase=1
        screen.fill(BG_DARK)
        for gx in range(0,W,60): pygame.draw.line(screen,(30,18,55),(gx,0),(gx,H),1)
        for gy in range(0,H,60): pygame.draw.line(screen,(30,18,55),(0,gy),(W,gy),1)
        for p in particles: p.update(); p.draw(screen)
        draw_glow_text(screen,song_name.upper(),font_title,accent,W//2,100,glow_radius=10)
        pygame.draw.line(screen,accent,(W//2-200,140),(W//2+200,140),2)
        if phase==1:
            draw_glow_text(screen,"COMO QUIERES JUGAR?",font_sub,WHITE,W//2,200,glow_radius=4)
            btn1=pygame.Rect(W//2-280,H//2-20,240,100)
            btn2=pygame.Rect(W//2+40,H//2-20,240,100)
            h1=btn1.collidepoint(mx,my); h2=btn2.collidepoint(mx,my)
            draw_rounded_rect(screen,DARK_CARD,btn1,18,alpha=240,border=3,border_color=NEON_CYAN if h1 else (60,40,90))
            draw_glow_text(screen,"1 JUGADOR",font_btn,WHITE if h1 else GRAY,btn1.centerx,btn1.centery,glow_radius=4 if h1 else 1)
            draw_rounded_rect(screen,DARK_CARD,btn2,18,alpha=240,border=3,border_color=NEON_PINK if h2 else (60,40,90))
            draw_glow_text(screen,"2 JUGADORES",font_btn,WHITE if h2 else GRAY,btn2.centerx,btn2.centery,glow_radius=4 if h2 else 1)
            draw_glow_text(screen,"ESC = volver al lobby",font_hint,GRAY,W//2,H-30)
        elif phase==2:
            draw_glow_text(screen,"QUE PERSONAJE QUIERES IMITAR?",font_sub,WHITE,W//2,200,glow_radius=4)
            for i,char in enumerate(characters):
                char_color=CHAR_COLORS[i%len(CHAR_COLORS)]
                bx=W//2-(len(characters)*155)//2+i*165
                btn=pygame.Rect(bx,H//2-60,145,190)
                hov=btn.collidepoint(mx,my)
                draw_rounded_rect(screen,DARK_CARD,btn,18,alpha=240,border=3,border_color=char_color if hov else (60,40,90))
                icon_y=btn.y+50
                pygame.draw.circle(screen,char_color,(btn.centerx,icon_y),22,0 if hov else 2)
                pygame.draw.line(screen,char_color,(btn.centerx,icon_y+22),(btn.centerx,icon_y+70),4)
                pygame.draw.line(screen,char_color,(btn.centerx-28,icon_y+40),(btn.centerx+28,icon_y+40),4)
                pygame.draw.line(screen,char_color,(btn.centerx,icon_y+70),(btn.centerx-20,icon_y+100),4)
                pygame.draw.line(screen,char_color,(btn.centerx,icon_y+70),(btn.centerx+20,icon_y+100),4)
                nf=pygame.font.SysFont("trebuchetms",15,bold=True)
                ns_=nf.render(char["name"].upper(),True,WHITE if hov else GRAY)
                screen.blit(ns_,(btn.centerx-ns_.get_width()//2,btn.bottom-35))
            back=pygame.Rect(W//2-80,H-90,160,50)
            bh=back.collidepoint(mx,my)
            draw_rounded_rect(screen,DARK_CARD,back,12,alpha=220,border=2,border_color=GRAY)
            draw_glow_text(screen,"< VOLVER",font_btn,WHITE if bh else GRAY,back.centerx,back.centery,glow_radius=2)
        pygame.display.flip()


# ── Pantalla de selección de dificultad (v3 con best scores) ──────────────────

def select_difficulty_screen(screen, clock, W, H, song_key=None, song_name=""):
       
    DIFFICULTIES = [
        ("EASY",    (  0,220,120), "Fácil - Movilidad Reducida (±40°)", "★",    10000),
        ("NORMAL",  (255,200,  0), "Medio - Movilidad Moderada (±28°)", "★★",   12500),
        ("HARD",    (255, 50, 80), "Difícil - Movilidad Avanzada (±15°)", "★★★",  15000),
        ("EXTREME", (200,  0,255), "Extremo - Rango Completo (±8°)", "★★★★", 20000),
    ]

    font_title = pygame.font.SysFont("impact",      46)
    font_name  = pygame.font.SysFont("impact",      30)
    font_desc  = pygame.font.SysFont("trebuchetms", 13)
    font_hint  = pygame.font.SysFont("trebuchetms", 14)
    particles  = [Particle(W,H) for _ in range(90)]
    tick       = 0

    prev_w  = min(560, W-200)
    prev_h  = int(prev_w*9/16)
    video_y = 96
    video_surf = get_video_preview_surface(song_key, prev_w, prev_h) if song_key else None

    BTN_W, BTN_H = 240, 122
    gap      = 14
    total_bw = len(DIFFICULTIES)*BTN_W+(len(DIFFICULTIES)-1)*gap
    bx_start = W//2-total_bw//2
    btn_y    = video_y+prev_h+50

    buttons = [pygame.Rect(bx_start+i*(BTN_W+gap), btn_y, BTN_W, BTN_H)
               for i in range(len(DIFFICULTIES))]

    # Pre-cargar best scores por dificultad
    def best_info(diff, max_s):
        if not song_key: return 0, 0
        sc = get_best_score(song_key, diff)
        st = get_best_stars(song_key, diff)
        return sc, st

    while True:
        clock.tick(60); tick += 1
        mx,my = pygame.mouse.get_pos()
        new_surf = get_video_preview_surface(song_key, prev_w, prev_h) if song_key else None
        if new_surf is not None: video_surf = new_surf

        for event in pygame.event.get():
            if event.type==pygame.QUIT: pygame.display.quit(); import sys; sys.exit()
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: return None
                if event.key==pygame.K_1: return "EASY"
                if event.key==pygame.K_2: return "NORMAL"
                if event.key==pygame.K_3: return "HARD"
                if event.key==pygame.K_4: return "EXTREME"
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                for i,rect in enumerate(buttons):
                    if rect.collidepoint(mx,my): return DIFFICULTIES[i][0]

        screen.fill(BG_DARK)
        for gx in range(0,W,60): pygame.draw.line(screen,(30,18,55),(gx,0),(gx,H),1)
        for gy in range(0,H,60): pygame.draw.line(screen,(30,18,55),(0,gy),(W,gy),1)
        for p in particles: p.update(); p.draw(screen)

        pulse = math.sin(tick*0.05)*0.5+0.5
        tc = (int(NEON_CYAN[0]+(NEON_PINK[0]-NEON_CYAN[0])*pulse),
              int(NEON_CYAN[1]+(NEON_PINK[1]-NEON_CYAN[1])*pulse),
              int(NEON_CYAN[2]+(NEON_PINK[2]-NEON_CYAN[2])*pulse))

        draw_glow_text(screen,"SELECCIONA DIFICULTAD",font_title,tc,W//2,54,glow_radius=10)
        pygame.draw.line(screen,NEON_CYAN,(W//2-200,78),(W//2+200,78),2)

        # Video / artwork preview
        if video_surf is not None:
            vs_w,vs_h = video_surf.get_width(),video_surf.get_height()
            vx2 = W//2-vs_w//2
            glow_s = pygame.Surface((vs_w+20,vs_h+20),pygame.SRCALPHA)
            pygame.draw.rect(glow_s,(0,215,255,int(30+30*pulse)),(0,0,vs_w+20,vs_h+20),border_radius=10)
            screen.blit(glow_s,(vx2-10,video_y-10))
            screen.blit(video_surf,(vx2,video_y))
            pygame.draw.rect(screen,(0,215,255),(vx2-2,video_y-2,vs_w+4,vs_h+4),2,border_radius=6)
        else:
            vx  = W//2-prev_w//2
            art = load_song_artwork(song_key or "unknown", song_name, prev_w, prev_h)
            screen.blit(art,(vx,video_y))
            pygame.draw.rect(screen,(0,215,255),(vx-2,video_y-2,prev_w+4,prev_h+4),2,border_radius=6)

        if song_name:
            sf = pygame.font.SysFont("trebuchetms", 14, bold=True)
            ss = sf.render(song_name.upper(), True, (180, 170, 200))
            screen.blit(ss, (W//2 - ss.get_width()//2, video_y + prev_h + 6))

            if song_key:
                complexity = SONG_COMPLEXITY.get(song_key, "MEDIO")
                comp_color = COMPLEXITY_COLORS[complexity]
                comp_label = COMPLEXITY_LABELS[complexity]
                badge_font = pygame.font.SysFont("trebuchetms", 13, bold=True)
                badge_surf = badge_font.render(f"  NIVEL: {comp_label}  ", True, comp_color)
                badge_x    = W//2 - badge_surf.get_width()//2
                badge_y    = video_y + prev_h + 26
                badge_bg   = pygame.Rect(badge_x - 4, badge_y - 2,
                                         badge_surf.get_width() + 8,
                                         badge_surf.get_height() + 4)
                draw_rounded_rect(screen, (20, 14, 35), badge_bg, 6,
                                  alpha=220, border=1, border_color=comp_color)
                screen.blit(badge_surf, (badge_x, badge_y))

        sub_f = pygame.font.SysFont("trebuchetms", 13)
        sub_s = sub_f.render("¿Qué tan exigente quieres la evaluación?", True, GRAY)
        screen.blit(sub_s, (W//2 - sub_s.get_width()//2, video_y + prev_h + 50))

        # Botones de dificultad
        for i,(rect,diff_data) in enumerate(zip(buttons,DIFFICULTIES)):
            label,color,desc,stars_str,max_s = diff_data
            hov   = rect.collidepoint(mx,my)
            scale = 1.04 if hov else 1.0
            sw2,sh2 = int(BTN_W*scale),int(BTN_H*scale)
            sx,sy   = rect.centerx-sw2//2, rect.centery-sh2//2
            sr2     = pygame.Rect(sx,sy,sw2,sh2)

            ga = int(80+80*pulse) if hov else 30
            gs2 = pygame.Surface((sw2+20,sh2+20),pygame.SRCALPHA)
            pygame.draw.rect(gs2,(*color,ga),(0,0,sw2+20,sh2+20),border_radius=20)
            screen.blit(gs2,(sx-10,sy-10))
            draw_rounded_rect(screen,DARK_CARD,sr2,16,alpha=240,border=3,border_color=color)

            # Estrellas indicador de dificultad
            sf2 = pygame.font.SysFont("segoeuisymbol",14)
            ss2 = sf2.render(stars_str,True,color)
            screen.blit(ss2,(sr2.centerx-ss2.get_width()//2, sr2.y+8))

            draw_glow_text(screen,label,font_name,color,
                           sr2.centerx, sr2.y+46, glow_radius=10 if hov else 4)

            ds = font_desc.render(desc, True, WHITE if hov else GRAY)
            screen.blit(ds,(sr2.centerx-ds.get_width()//2, sr2.y+68))

            # Mejor puntuación personal para esta dificultad
            bsc, bst = best_info(label, max_s)
            if bsc > 0:
                bf = pygame.font.SysFont("trebuchetms",12,bold=True)
                bs = bf.render(f"MEJOR: {bsc:,}", True, color)
                screen.blit(bs,(sr2.centerx-bs.get_width()//2, sr2.y+84))
                _draw_stars_row(screen, sr2.centerx-5*10, sr2.y+100, bst, total=5, r=6, gap=12)
            else:
                nf2 = pygame.font.SysFont("trebuchetms",11)
                ns2 = nf2.render("Sin registro",True,(70,60,90))
                screen.blit(ns2,(sr2.centerx-ns2.get_width()//2, sr2.y+90))

            kf = pygame.font.SysFont("trebuchetms",11)
            ks = kf.render(f"[{i+1}]",True,(90,80,120))
            screen.blit(ks,(sr2.x+7,sr2.y+7))

        hint = font_hint.render("ESC = volver  |  [1] EASY  [2] NORMAL  [3] HARD  [4] EXTREME",
                                True,(80,70,110))
        screen.blit(hint,(W//2-hint.get_width()//2, H-30))
        pygame.display.flip()


def select_repetitions_screen(screen, clock, W, H, song_key, song_name, difficulty):
    font_title = pygame.font.SysFont("impact", 54)
    font_sub = pygame.font.SysFont("trebuchetms", 18)
    font_number = pygame.font.SysFont("impact", 100)
    font_btn = pygame.font.SysFont("impact", 30)
    font_hint = pygame.font.SysFont("trebuchetms", 14)
    
    reps = 5
    particles = [Particle(W, H) for _ in range(60)]
    tick = 0
    
    CX = W // 2
    CY = H // 2
    
    btn_minus_rect = pygame.Rect(CX - 160, CY - 40, 80, 80)
    btn_plus_rect = pygame.Rect(CX + 80, CY - 40, 80, 80)
    
    presets = [
        (3, pygame.Rect(CX - 150, CY + 80, 60, 40)),
        (5, pygame.Rect(CX - 70, CY + 80, 60, 40)),
        (10, pygame.Rect(CX + 10, CY + 80, 60, 40)),
        (15, pygame.Rect(CX + 90, CY + 80, 60, 40)),
    ]
    
    start_rect = pygame.Rect(CX - 180, CY + 150, 360, 60)
    back_rect = pygame.Rect(CX - 80, H - 70, 160, 40)
    
    accent = (0, 205, 210)
    
    while True:
        clock.tick(60)
        tick += 1
        mx, my = pygame.mouse.get_pos()
        pulse = math.sin(tick * 0.05) * 0.5 + 0.5
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                import sys
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return reps
                if event.key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS):
                    reps = max(1, reps - 1)
                if event.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_EQUALS):
                    reps = min(30, reps + 1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_minus_rect.collidepoint(mx, my):
                    reps = max(1, reps - 1)
                elif btn_plus_rect.collidepoint(mx, my):
                    reps = min(30, reps + 1)
                elif start_rect.collidepoint(mx, my):
                    return reps
                elif back_rect.collidepoint(mx, my):
                    return None
                for val, r in presets:
                    if r.collidepoint(mx, my):
                        reps = val
                        
        screen.fill(BG_DARK)
        for gx in range(0, W, 60):
            pygame.draw.line(screen, (18, 42, 46), (gx, 0), (gx, H), 1)
        for gy in range(0, H, 60):
            pygame.draw.line(screen, (18, 42, 46), (0, gy), (W, gy), 1)
        for p in particles:
            p.update()
            p.draw(screen)
            
        tc = (int(NEON_CYAN[0]+(NEON_PINK[0]-NEON_CYAN[0])*pulse),
              int(NEON_CYAN[1]+(NEON_PINK[1]-NEON_CYAN[1])*pulse),
              int(NEON_CYAN[2]+(NEON_PINK[2]-NEON_CYAN[2])*pulse))
        draw_glow_text(screen, "CONFIGURAR REPETICIONES", font_title, tc, CX, 90, glow_radius=10)
        pygame.draw.line(screen, accent, (CX - 220, 130), (CX + 220, 130), 2)
        
        sub_text = f"Ejercicio: {song_name.upper()}  |  Precisión: {difficulty}"
        draw_glow_text(screen, sub_text, font_sub, WHITE, CX, 160, glow_radius=2)
        
        num_s = font_number.render(str(reps), True, WHITE)
        screen.blit(num_s, (CX - num_s.get_width() // 2, CY - num_s.get_height() // 2 - 10))
        
        hm = btn_minus_rect.collidepoint(mx, my)
        draw_rounded_rect(screen, (26, 56, 62) if hm else DARK_CARD, btn_minus_rect, 40,
                          alpha=230 if hm else 180, border=2, border_color=accent)
        draw_glow_text(screen, "-", font_btn, accent if hm else WHITE, btn_minus_rect.centerx, btn_minus_rect.centery)
        
        hp = btn_plus_rect.collidepoint(mx, my)
        draw_rounded_rect(screen, (26, 56, 62) if hp else DARK_CARD, btn_plus_rect, 40,
                          alpha=230 if hp else 180, border=2, border_color=accent)
        draw_glow_text(screen, "+", font_btn, accent if hp else WHITE, btn_plus_rect.centerx, btn_plus_rect.centery)
        
        for val, r in presets:
            hpr = r.collidepoint(mx, my) or reps == val
            draw_rounded_rect(screen, (26, 56, 62) if hpr else DARK_CARD, r, 10,
                              alpha=230 if hpr else 180, border=2, border_color=accent if hpr else (60, 45, 90))
            draw_glow_text(screen, f"{val} Reps", font_hint, accent if hpr else GRAY, r.centerx, r.centery)
            
        hs = start_rect.collidepoint(mx, my)
        draw_rounded_rect(screen, accent if hs else DARK_CARD, start_rect, 15,
                          alpha=255 if hs else 220, border=2, border_color=accent)
        draw_glow_text(screen, "INICIAR TERAPIA", font_btn, WHITE if hs else accent, start_rect.centerx, start_rect.centery, glow_radius=8 if hs else 2)
        
        hb = back_rect.collidepoint(mx, my)
        draw_rounded_rect(screen, DARKER_CARD, back_rect, 10, alpha=200,
                          border=2, border_color=accent if hb else (60, 45, 90))
        draw_glow_text(screen, "< VOLVER", font_hint, accent if hb else GRAY, back_rect.centerx, back_rect.centery)
        
        pygame.display.flip()


# ── Pantalla de Opciones ───────────────────────────────────────────────────────
def run_options(screen, clock, W, H, particles, volume, res_index):
    font_title=pygame.font.SysFont("impact",64)
    font_label=pygame.font.SysFont("trebuchetms",22,bold=True)
    font_hint=pygame.font.SysFont("trebuchetms",15)
    font_val=pygame.font.SysFont("impact",26)
    center_x=W//2; bar_w=min(480,W-240); tick=0
    while True:
        clock.tick(60); tick+=1
        mx,my=pygame.mouse.get_pos()
        vol_y=H//2-120; res_y=H//2+50
        bar_rect=pygame.Rect(center_x-bar_w//2,vol_y+8,bar_w,22)
        fill_w=int(bar_w*volume)
        res_left_rect=pygame.Rect(center_x-bar_w//2,res_y+4,38,38)
        res_right_rect=pygame.Rect(center_x+bar_w//2-38,res_y+4,38,38)
        apply_rect=pygame.Rect(center_x-115,res_y+58,230,42)
        back_rect=pygame.Rect(center_x-95,H-88,190,48)
        for event in pygame.event.get():
            if event.type==pygame.QUIT: pygame.mixer.music.stop(); pygame.display.quit(); sys.exit()
            if event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE: return volume,res_index,None
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                if bar_rect.collidepoint(mx,my):
                    volume=max(0.0,min(1.0,(mx-bar_rect.x)/bar_w))
                    pygame.mixer.music.set_volume(volume)
                if res_left_rect.collidepoint(mx,my): res_index=(res_index-1)%len(RESOLUTIONS)
                if res_right_rect.collidepoint(mx,my): res_index=(res_index+1)%len(RESOLUTIONS)
                if apply_rect.collidepoint(mx,my):
                    nw,nh,_=RESOLUTIONS[res_index]; return volume,res_index,(nw,nh)
                if back_rect.collidepoint(mx,my): return volume,res_index,None
            if event.type==pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0] and bar_rect.collidepoint(mx,my):
                    volume=max(0.0,min(1.0,(mx-bar_rect.x)/bar_w))
                    pygame.mixer.music.set_volume(volume)
        screen.fill(BG_DARK)
        for gx in range(0,W,60): pygame.draw.line(screen,(30,18,55),(gx,0),(gx,H),1)
        for gy in range(0,H,60): pygame.draw.line(screen,(30,18,55),(0,gy),(W,gy),1)
        for p in particles: p.update(); p.draw(screen)
        pulse=math.sin(tick*0.04)*0.5+0.5
        tc=(int(NEON_CYAN[0]+(NEON_PINK[0]-NEON_CYAN[0])*pulse),
            int(NEON_CYAN[1]+(NEON_PINK[1]-NEON_CYAN[1])*pulse),
            int(NEON_CYAN[2]+(NEON_PINK[2]-NEON_CYAN[2])*pulse))
        draw_glow_text(screen,"OPCIONES",font_title,tc,center_x,80,glow_radius=14)
        pygame.draw.line(screen,NEON_CYAN,(center_x-170,126),(center_x+170,126),2)
        cv=pygame.Rect(center_x-bar_w//2-28,vol_y-48,bar_w+56,132)
        draw_rounded_rect(screen,DARK_CARD,cv,18,alpha=220,border=2,border_color=NEON_CYAN)
        draw_glow_text(screen,"VOLUMEN",font_label,NEON_CYAN,center_x,vol_y-20,glow_radius=6)
        draw_rounded_rect(screen,(40,25,70),bar_rect,10,alpha=220)
        if fill_w>0: draw_rounded_rect(screen,NEON_CYAN,(bar_rect.x,bar_rect.y,fill_w,22),10,alpha=240)
        pygame.draw.rect(screen,NEON_CYAN,bar_rect,2,border_radius=10)
        kx=bar_rect.x+int(bar_w*volume)
        kh=abs(mx-kx)<14 and abs(my-bar_rect.centery)<14
        pygame.draw.circle(screen,WHITE,(kx,bar_rect.centery),14 if kh else 11)
        pygame.draw.circle(screen,NEON_CYAN,(kx,bar_rect.centery),14 if kh else 11,2)
        draw_glow_text(screen,f"{int(volume*100)}%",font_val,WHITE,center_x,vol_y+52,glow_radius=4)
        l0=font_hint.render("0%",True,GRAY); l100=font_hint.render("100%",True,GRAY)
        screen.blit(l0,(bar_rect.x,vol_y+36)); screen.blit(l100,(bar_rect.right-l100.get_width(),vol_y+36))
        cr=pygame.Rect(center_x-bar_w//2-28,res_y-48,bar_w+56,158)
        draw_rounded_rect(screen,DARK_CARD,cr,18,alpha=220,border=2,border_color=NEON_PINK)
        draw_glow_text(screen,"RESOLUCION",font_label,NEON_PINK,center_x,res_y-20,glow_radius=6)
        lh=res_left_rect.collidepoint(mx,my)
        draw_rounded_rect(screen,(50,28,80) if lh else DARK_CARD,res_left_rect,8,alpha=220,border=2,border_color=NEON_PINK if lh else GRAY)
        draw_glow_text(screen,"<",font_label,NEON_PINK if lh else GRAY,res_left_rect.centerx,res_left_rect.centery,glow_radius=4)
        _,_,rlabel=RESOLUTIONS[res_index]
        draw_glow_text(screen,rlabel,font_val,WHITE,center_x,res_y+24,glow_radius=3)
        rh=res_right_rect.collidepoint(mx,my)
        draw_rounded_rect(screen,(50,28,80) if rh else DARK_CARD,res_right_rect,8,alpha=220,border=2,border_color=NEON_PINK if rh else GRAY)
        draw_glow_text(screen,">",font_label,NEON_PINK if rh else GRAY,res_right_rect.centerx,res_right_rect.centery,glow_radius=4)
        ah=apply_rect.collidepoint(mx,my)
        draw_rounded_rect(screen,NEON_PINK if ah else DARK_CARD,apply_rect,10,alpha=255 if ah else 200,border=2,border_color=NEON_PINK)
        draw_glow_text(screen,"APLICAR RESOLUCION",font_hint,WHITE,apply_rect.centerx,apply_rect.centery,glow_radius=4 if ah else 1)
        note=font_hint.render("* Reinicia el lobby con la nueva resolución.",True,GRAY)
        screen.blit(note,(center_x-note.get_width()//2,res_y+112))
        bh2=back_rect.collidepoint(mx,my)
        draw_rounded_rect(screen,DARKER_CARD,back_rect,12,alpha=200,border=2,border_color=NEON_CYAN if bh2 else (80,60,110))
        draw_glow_text(screen,"< VOLVER",font_label,NEON_CYAN if bh2 else GRAY,back_rect.centerx,back_rect.centery,glow_radius=5 if bh2 else 2)
        esc=font_hint.render("ESC  para volver",True,GRAY)
        screen.blit(esc,(center_x-esc.get_width()//2,H-34))
        pygame.display.flip()


# ── Lobby principal ────────────────────────────────────────────────────────────


def run_lobby(W=None, H=None, volume=0.8, res_index=0):
    """Lobby clínico rediseñado para Rehabilitación Física."""
    pygame.init(); pygame.mixer.init()

    # Pre-cargar backends de MediaPipe en segundo plano mientras el menu esta visible
    try:
        import backend_preloader
        backend_preloader.preload()
    except Exception:
        pass
    if W is None or H is None:
        pygame.display.init()
        try:
            info = pygame.display.Info(); W, H = info.current_w, info.current_h
        except Exception:
            W, H = 1366, 768
        if W < 800 or H < 600:
            W, H = 1366, 768

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Rehabilitación Física")
    apply_always_on_top()
    clock = pygame.time.Clock()

    if load_profile() is None:
        run_profile_screen(screen, clock, W, H, first_run=True)

    profile = load_profile() or {"name": "PACIENTE", "color_index": 0}
    pname = profile.get("name", "PACIENTE")
    pidx = int(profile.get("color_index", 0) or 0)

    # Tema clínico
    BG = (12, 20, 40)
    CARD = (20, 35, 65)
    CARD_DARK = (14, 26, 50)
    TEAL = (0, 180, 180)
    GREEN = (40, 200, 120)
    AMBER = (255, 160, 40)
    TEXT = (235, 245, 250)
    MUTED = (150, 170, 185)
    DANGER = (230, 80, 80)
    BORDER = (35, 60, 95)

    font_title = pygame.font.SysFont("segoeui", 42, bold=True)
    font_sub = pygame.font.SysFont("segoeui", 18)
    font_card_title = pygame.font.SysFont("segoeui", 24, bold=True)
    font_card = pygame.font.SysFont("segoeui", 15)
    font_small = pygame.font.SysFont("segoeui", 13)
    font_btn = pygame.font.SysFont("segoeui", 18, bold=True)
    font_badge = pygame.font.SysFont("segoeui", 13, bold=True)

    def rounded(surface, color, rect, radius=16, border=0, border_color=None, alpha=245):
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
        if border and border_color:
            pygame.draw.rect(s, (*border_color, 255), (0, 0, rect[2], rect[3]), width=border, border_radius=radius)
        surface.blit(s, (rect[0], rect[1]))

    def fit(text, font, max_w):
        text = str(text)
        if font.size(text)[0] <= max_w:
            return text
        ell = "..."
        out = ""
        for ch in text:
            if font.size(out + ch + ell)[0] > max_w:
                break
            out += ch
        return out.rstrip() + ell

    def get_region(key):
        cfg = REHAB_EXERCISE_CONFIGS.get(key, {})
        if cfg.get("body_region"):
            return str(cfg.get("body_region")).upper()
        parts = cfg.get("body_parts", []) or []
        if parts:
            return str(parts[0]).upper()
        return "GENERAL"

    def get_side_label(cfg):
        side = str(cfg.get("side", "")).lower()
        if side == "right": return "Derecho"
        if side == "left": return "Izquierdo"
        return "Bilateral" if side else "General"

    def get_sessions_completed(song_key):
        try:
            top = get_top5_for_song(song_key) or []
            return len(top)
        except Exception:
            return 0

    all_items = [(cfg.get("name", key), key) for key, cfg in REHAB_EXERCISE_CONFIGS.items()]
    categories = ["TODOS"]
    for _, key in all_items:
        r = get_region(key)
        if r not in categories:
            categories.append(r)
    selected_cat = "TODOS"
    selected = 0
    reps = 5
    difficulty = "EASY"
    use_gpu = True
    scroll = 0

    def filtered_items():
        if selected_cat == "TODOS":
            return all_items[:]
        return [(name, key) for name, key in all_items if get_region(key) == selected_cat]

    def start_session():
        items = filtered_items()
        if not items:
            return
        name, key = items[max(0, min(selected, len(items)-1))]
        char_info = {"mode": "single", "char_idx": 0, "char_info": None}
        run_game(key, char_info=char_info, screen_w=W, screen_h=H,
                 difficulty=difficulty, volume=volume, use_gpu=use_gpu,
                 repetitions=reps)

    while True:
        clock.tick(60)
        mx, my = pygame.mouse.get_pos()
        items = filtered_items()
        if selected >= len(items):
            selected = max(0, len(items)-1)

        # Layout
        header_h = 96
        left_w = int(W * 0.36)
        margin = 28
        tabs_y = header_h + 18
        list_x = left_w + margin
        list_y = tabs_y + 58
        row_h = 86
        visible_rows = max(4, int((H - list_y - 105) // row_h))
        if selected < scroll: scroll = selected
        if selected >= scroll + visible_rows: scroll = selected - visible_rows + 1

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.mixer.music.stop(); pygame.display.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.mixer.music.stop(); pygame.display.quit(); sys.exit()
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = min(selected + 1, max(0, len(items)-1))
                elif event.key in (pygame.K_UP, pygame.K_w):
                    selected = max(selected - 1, 0)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    start_session(); return
                elif event.key in (pygame.K_RIGHT, pygame.K_PLUS, pygame.K_KP_PLUS):
                    reps = min(30, reps + 1)
                elif event.key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
                    reps = max(1, reps - 1)
                elif event.key == pygame.K_g:
                    use_gpu = not use_gpu
            if event.type == pygame.MOUSEWHEEL:
                selected = max(0, min(max(0, len(items)-1), selected - event.y))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Tabs
                tx = margin
                for cat in categories:
                    tw = max(92, font_badge.size(cat)[0] + 28)
                    rect = pygame.Rect(tx, tabs_y, tw, 34)
                    if rect.collidepoint(mx, my):
                        selected_cat = cat; selected = 0; scroll = 0
                    tx += tw + 10

                # Lista
                for idx in range(scroll, min(len(items), scroll + visible_rows)):
                    ry = list_y + (idx - scroll) * row_h
                    rect = pygame.Rect(list_x, ry, W - list_x - margin, row_h - 12)
                    if rect.collidepoint(mx, my):
                        selected = idx
                        if mx > rect.right - 90:
                            start_session(); return

                # Reps controls and buttons
                minus = pygame.Rect(margin + 30, H - 98, 42, 42)
                plus = pygame.Rect(margin + 170, H - 98, 42, 42)
                start_btn = pygame.Rect(left_w - 230, H - 98, 190, 48)
                stats_btn = pygame.Rect(W - 280, 28, 90, 38)
                profile_btn = pygame.Rect(W - 176, 28, 118, 38)
                gpu_btn = pygame.Rect(margin + 226, H - 98, 92, 42)
                if minus.collidepoint(mx, my): reps = max(1, reps - 1)
                if plus.collidepoint(mx, my): reps = min(30, reps + 1)
                if gpu_btn.collidepoint(mx, my): use_gpu = not use_gpu
                if start_btn.collidepoint(mx, my): start_session(); return
                if stats_btn.collidepoint(mx, my): run_stats_screen(screen, clock, W, H)
                if profile_btn.collidepoint(mx, my): run_profile_screen(screen, clock, W, H, first_run=False); profile.update(load_profile() or {})

        # Fondo
        screen.fill(BG)
        for gx in range(0, W, 64): pygame.draw.line(screen, (16, 30, 55), (gx, 0), (gx, H), 1)
        for gy in range(0, H, 64): pygame.draw.line(screen, (16, 30, 55), (0, gy), (W, gy), 1)

        # Header
        pygame.draw.rect(screen, CARD_DARK, (0, 0, W, header_h))
        pygame.draw.line(screen, TEAL, (0, header_h), (W, header_h), 2)
        screen.blit(font_title.render("Rehabilitación Física", True, TEXT), (margin, 22))
        screen.blit(font_sub.render("Selecciona un ejercicio terapéutico y configura la sesión", True, MUTED), (margin, 62))

        stats_btn = pygame.Rect(W - 280, 28, 90, 38)
        profile_btn = pygame.Rect(W - 176, 28, 118, 38)
        rounded(screen, CARD, stats_btn, 12, 1, TEAL if stats_btn.collidepoint(mx, my) else BORDER)
        rounded(screen, CARD, profile_btn, 12, 1, TEAL if profile_btn.collidepoint(mx, my) else BORDER)
        screen.blit(font_badge.render("STATS", True, TEXT), (stats_btn.centerx - 21, stats_btn.centery - 8))
        screen.blit(font_badge.render("PERFIL", True, TEXT), (profile_btn.centerx - 25, profile_btn.centery - 8))

        # Perfil mini
        av_col = get_avatar_color(pidx)
        pygame.draw.circle(screen, av_col, (margin + 12, 16), 12)
        pygame.draw.circle(screen, TEXT, (margin + 12, 16), 12, 1)
        screen.blit(font_small.render(str(pname).upper(), True, TEXT), (margin + 34, 8))

        # Tabs superiores
        tx = margin
        for cat in categories:
            tw = max(92, font_badge.size(cat)[0] + 28)
            rect = pygame.Rect(tx, tabs_y, tw, 34)
            active = cat == selected_cat
            rounded(screen, TEAL if active else CARD, rect, 12, 1, TEAL if active else BORDER, alpha=255 if active else 220)
            col = BG if active else MUTED
            label = font_badge.render(cat, True, col)
            screen.blit(label, (rect.centerx - label.get_width()//2, rect.centery - label.get_height()//2))
            tx += tw + 10

        # Panel izquierdo detalle
        panel = pygame.Rect(margin, list_y, left_w - margin * 2, H - list_y - 128)
        rounded(screen, CARD_DARK, panel, 22, 2, BORDER)
        if items:
            name, key = items[selected]
            cfg = REHAB_EXERCISE_CONFIGS.get(key, {})
            region = get_region(key)
            side = get_side_label(cfg)
            body_parts = ", ".join(cfg.get("body_parts", [region]) or [region])
            sessions = get_sessions_completed(key)
            # Badge grande
            badge = pygame.Rect(panel.x + 24, panel.y + 24, 160, 38)
            rounded(screen, TEAL, badge, 14)
            screen.blit(font_badge.render(region, True, BG), (badge.x + 18, badge.y + 10))
            screen.blit(font_card_title.render(fit(name, font_card_title, panel.w - 48), True, TEXT), (panel.x + 24, panel.y + 86))
            screen.blit(font_card.render(f"Zona evaluada: {body_parts}", True, MUTED), (panel.x + 24, panel.y + 126))
            screen.blit(font_card.render(f"Lado: {side}", True, MUTED), (panel.x + 24, panel.y + 154))
            screen.blit(font_card.render(f"Sesiones completadas: {sessions}", True, GREEN), (panel.x + 24, panel.y + 182))
            mode = cfg.get("evaluation_mode", "rangos")
            screen.blit(font_card.render(f"Validación: recorrido completo", True, TEAL), (panel.x + 24, panel.y + 210))

            # Reps controls
            screen.blit(font_label.render("REPETICIONES", True, MUTED) if 'font_label' in globals() else font_badge.render("REPETICIONES", True, MUTED), (panel.x + 24, H - 112))
        # Controles inferiores
        minus = pygame.Rect(margin + 30, H - 98, 42, 42)
        plus = pygame.Rect(margin + 170, H - 98, 42, 42)
        gpu_btn = pygame.Rect(margin + 226, H - 98, 92, 42)
        start_btn = pygame.Rect(left_w - 230, H - 98, 190, 48)
        for rect, label in [(minus, "-"), (plus, "+")]:
            rounded(screen, CARD, rect, 12, 1, TEAL if rect.collidepoint(mx, my) else BORDER)
            t = font_btn.render(label, True, TEXT)
            screen.blit(t, (rect.centerx - t.get_width()//2, rect.centery - t.get_height()//2))
        rep_txt = font_value if False else pygame.font.SysFont("segoeui", 30, bold=True)
        rt = rep_txt.render(str(reps), True, TEXT)
        screen.blit(rt, (margin + 105 - rt.get_width()//2, H - 95))
        rounded(screen, GREEN if use_gpu else CARD, gpu_btn, 12, 1, GREEN if use_gpu else BORDER)
        gt = font_badge.render("GPU" if use_gpu else "CPU", True, BG if use_gpu else TEXT)
        screen.blit(gt, (gpu_btn.centerx - gt.get_width()//2, gpu_btn.centery - gt.get_height()//2))
        rounded(screen, TEAL, start_btn, 14, 0)
        st = font_btn.render("INICIAR SESIÓN", True, BG)
        screen.blit(st, (start_btn.centerx - st.get_width()//2, start_btn.centery - st.get_height()//2))

        # Lista derecha
        title = font_sub.render(f"Ejercicios disponibles · {len(items)}", True, TEXT)
        screen.blit(title, (list_x, list_y - 36))
        for idx in range(scroll, min(len(items), scroll + visible_rows)):
            name, key = items[idx]
            cfg = REHAB_EXERCISE_CONFIGS.get(key, {})
            region = get_region(key)
            ry = list_y + (idx - scroll) * row_h
            rect = pygame.Rect(list_x, ry, W - list_x - margin, row_h - 12)
            active = idx == selected
            rounded(screen, CARD if active else CARD_DARK, rect, 16, 2, TEAL if active else BORDER, alpha=245 if active else 205)
            badge_rect = pygame.Rect(rect.x + 18, rect.y + 18, 118, 28)
            rounded(screen, TEAL if active else CARD, badge_rect, 10, 1, TEAL)
            btxt = font_badge.render(region[:18], True, BG if active else TEAL)
            screen.blit(btxt, (badge_rect.centerx - btxt.get_width()//2, badge_rect.centery - btxt.get_height()//2))
            title_txt = font_card_title.render(fit(name, font_card_title, rect.w - 240), True, TEXT if active else MUTED)
            screen.blit(title_txt, (rect.x + 154, rect.y + 14))
            sub = cfg.get("artist", "Rehabilitación física")
            screen.blit(font_small.render(fit(sub, font_small, rect.w - 260), True, MUTED), (rect.x + 154, rect.y + 48))
            if active:
                tri = [(rect.right - 34, rect.centery - 12), (rect.right - 34, rect.centery + 12), (rect.right - 16, rect.centery)]
                pygame.draw.polygon(screen, TEAL, tri)

        # Ayuda inferior
        help_txt = "↑/↓ navegar · ENTER iniciar · +/- repeticiones · G CPU/GPU · ESC salir"
        screen.blit(font_small.render(help_txt, True, MUTED), (list_x, H - 34))
        pygame.display.flip()

if __name__ == "__main__":
    run_lobby()
