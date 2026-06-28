from __future__ import annotations
"""
just_dance_gui_score.py — v3 (UI overhaul)
Pantalla de resultados con animacion de nuevo record e historial por ejercicio.
NOTA: just_dance_main.py debe pasar song_key al construir Score para que el
historial sea especifico al ejercicio. Si no se pasa, se guarda en "unknown".
"""
import sys, math, random, time
import pygame

from just_dance_score import (
    add_score, get_top5_for_song, load_profile,
    update_stats_after_game, AVATAR_COLORS, get_avatar_color, compute_stars
)

BG_DARK      = (7, 18, 20)
NEON_CYAN    = (0, 205, 210)
NEON_PINK    = (70, 135, 255)
NEON_YELLOW  = (255, 190, 70)
NEON_GREEN   = (50, 220, 150)
WHITE        = (255, 255, 255)
GRAY         = (140, 130, 160)
DARK_CARD    = (18, 32, 38)
DARKER_CARD  = (16,  10, 35)

RANK_COLORS = [(255,200,0),(180,180,200),(200,120,60),(100,90,140),(80,75,120)]

DIFF_COLORS = {
    "EASY":    (0,  220,120),
    "NORMAL":  (0,  200,255),
    "HARD":    (60,  60,255),
    "EXTREME": (200,  0,255),
}


# ── Partículas de nuevo récord ─────────────────────────────────────────────────

class RecordParticle:
    __slots__ = ("x","y","vx","vy","color","size","life","decay")
    def __init__(self, cx, cy):
        self.x     = cx + random.randint(-60,60)
        self.y     = cy + random.randint(-30,30)
        self.vx    = random.uniform(-11,11)
        self.vy    = random.uniform(-18,-4)
        col        = random.choice([(255,215,0),(255,255,100),(255,180,0),(255,140,0)])
        self.color = col
        self.size  = random.randint(5,12)
        self.life  = 1.0
        self.decay = random.uniform(0.018,0.04)
    def update(self):
        self.x   += self.vx; self.y += self.vy
        self.vy  += 0.6; self.vx  *= 0.97
        self.life -= self.decay
        return self.life > 0
    def draw(self, frame):
        if self.life<=0: return
        c = tuple(int(ch*self.life) for ch in self.color)
        pygame.draw.circle(frame, c, (int(self.x),int(self.y)), max(1,int(self.size*self.life)))


class Particle:
    def __init__(self,W,H):
        self.W=W;self.H=H;self.reset()
    def reset(self):
        self.x=random.randint(0,self.W); self.y=random.randint(0,self.H)
        self.size=random.uniform(1,2.5); self.speed=random.uniform(0.2,0.6)
        self.alpha=random.randint(30,150)
        self.color=random.choice([NEON_CYAN,NEON_PINK,NEON_YELLOW])
        self.drift=random.uniform(-0.2,0.2)
    def update(self):
        self.y-=self.speed;self.x+=self.drift;self.alpha-=0.3
        if self.y<0 or self.alpha<=0: self.reset();self.y=self.H
    def draw(self,surface):
        if self.alpha<=0:return
        s=pygame.Surface((int(self.size*2+1),int(self.size*2+1)),pygame.SRCALPHA)
        pygame.draw.circle(s,(*self.color,int(self.alpha)),(int(self.size),int(self.size)),int(self.size))
        surface.blit(s,(int(self.x),int(self.y)))


def draw_glow_text(surface,text,font,color,x,y,glow=6,center=True):
    for r in range(glow,0,-2):
        s=font.render(text,True,color);s.set_alpha(int(50*(1-r/glow)))
        sb=pygame.transform.scale(s,(s.get_width()+r*2,s.get_height()+r*2))
        bx=x-sb.get_width()//2 if center else x-r
        by=y-sb.get_height()//2 if center else y-r
        surface.blit(sb,(bx,by))
    ms=font.render(text,True,color)
    surface.blit(ms,(x-ms.get_width()//2 if center else x,y-ms.get_height()//2 if center else y))


def draw_rounded_rect(surface,color,rect,radius,alpha=255,border=0,border_color=None):
    w,h=max(1,rect[2]),max(1,rect[3])
    s=pygame.Surface((w,h),pygame.SRCALPHA)
    pygame.draw.rect(s,(*color,alpha),(0,0,w,h),border_radius=radius)
    if border and border_color:
        pygame.draw.rect(s,(*border_color,alpha),(0,0,w,h),width=border,border_radius=radius)
    surface.blit(s,(rect[0],rect[1]))


def _fit_text(text, font, max_width):
    if font.size(text)[0] <= max_width:
        return text
    ellipsis = "..."
    limit = max(1, max_width - font.size(ellipsis)[0])
    out = ""
    for ch in text:
        if font.size(out + ch)[0] > limit:
            break
        out += ch
    return out.rstrip() + ellipsis


def _draw_fit(surface, text, font, color, x, y, max_width):
    surface.blit(font.render(_fit_text(text, font, max_width), True, color), (x, y))


def _wrap_text(text, font, max_width, max_lines=4):
    words = str(text).split()
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if font.size(candidate)[0] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
            if len(lines) >= max_lines:
                break
    if line and len(lines) < max_lines:
        lines.append(line)
    if len(lines) == max_lines and words:
        lines[-1] = _fit_text(lines[-1], font, max_width)
    return lines or [""]


def _pct_color(value):
    return NEON_GREEN if value >= 80 else NEON_YELLOW if value >= 55 else (220, 70, 80)


def _draw_info_button(surface, rect, mx, my):
    hov = rect.collidepoint(mx, my)
    col = NEON_CYAN if hov else (95, 85, 125)
    pygame.draw.circle(surface, col, rect.center, rect.width // 2, 1)
    f = pygame.font.SysFont("trebuchetms", 12, bold=True)
    q = f.render("?", True, col)
    surface.blit(q, (rect.centerx - q.get_width() // 2, rect.centery - q.get_height() // 2))


def _draw_help_popup(surface, SW, SH, title, body_lines):
    popup_w = min(560, SW - 90)
    popup_h = 178
    px = SW // 2 - popup_w // 2
    py = SH // 2 - popup_h // 2
    shade = pygame.Surface((SW, SH), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 135))
    surface.blit(shade, (0, 0))
    draw_rounded_rect(surface, (20, 12, 38), (px, py, popup_w, popup_h), 16,
                      alpha=245, border=2, border_color=NEON_CYAN)
    title_f = pygame.font.SysFont("trebuchetms", 20, bold=True)
    body_f = pygame.font.SysFont("trebuchetms", 15)
    surface.blit(title_f.render(title, True, NEON_CYAN), (px + 24, py + 18))
    close_rect = pygame.Rect(px + popup_w - 42, py + 14, 28, 28)
    pygame.draw.circle(surface, (85, 60, 115), close_rect.center, 14, 1)
    x = body_f.render("X", True, WHITE)
    surface.blit(x, (close_rect.centerx - x.get_width() // 2,
                     close_rect.centery - x.get_height() // 2))
    text_y = py + 58
    for line in body_lines[:5]:
        surface.blit(body_f.render(line, True, (205, 198, 225)), (px + 24, text_y))
        text_y += 23
    return close_rect


def _draw_star(surface,cx,cy,r,color,filled=True):
    r_in=max(1,int(r*0.42)); pts=[]
    for i in range(10):
        angle=math.radians(-90+i*36); rad=r if i%2==0 else r_in
        pts.append((int(cx+rad*math.cos(angle)),int(cy+rad*math.sin(angle))))
    if filled: pygame.draw.polygon(surface,color,pts)
    else:       pygame.draw.polygon(surface,color,pts,1)


def _draw_stars_row(surface,x,y,n_filled,total=5,r=11,gap=26):
    for i in range(total):
        cx=x+i*gap
        if i<n_filled:
            _draw_star(surface,cx,y,r,(255,215,0),filled=True)
            _draw_star(surface,cx,y,r,(255,255,150),filled=False)
        else:
            _draw_star(surface,cx,y,r,(45,38,65),filled=True)
            _draw_star(surface,cx,y,r,(75,65,100),filled=False)


# ── Clase principal ─────────────────────────────────────────────────────────────

class Score:
    def __init__(self, score=0, screen_w=1280, screen_h=720,
                 difficulty="NORMAL", max_score=12500, best_combo=0,
                 joint_stats=None, song_key=None, player_name=None,
                 perfects_pct=0.0, song_duration=0.0,
                 performance_stats=None, repetitions=5, rep_results=None,
                 error_count=0, error_attempts=None):
        self.screen_w    = screen_w
        self.screen_h    = screen_h
        self.score       = score
        self.difficulty  = difficulty
        self.max_score   = max_score
        self.best_combo  = best_combo
        self.joint_stats = joint_stats or {}
        self.performance_stats = performance_stats or {}
        self.song_key    = song_key or "unknown"
        self.perfects_pct = float(perfects_pct)
        self.song_duration = float(song_duration)
        self.repetitions = repetitions
        self.rep_results = rep_results or []
        self.error_count = int(error_count)
        self.error_attempts = error_attempts or []

        # Cargar nombre del paciente desde perfil si no se paso
        if player_name is None:
            profile = load_profile() or {}
            player_name = profile.get("name","PACIENTE")
        self.player_name = player_name

        self.run()

    def run(self):
        """Pantalla clínica de resumen de sesión."""
        SW, SH = self.screen_w, self.screen_h
        pygame.init()
        screen = pygame.display.set_mode((SW, SH), pygame.NOFRAME)
        pygame.display.set_caption("Rehabilitación Física - Resumen de sesión")
        clock = pygame.time.Clock()

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
        font_sub = pygame.font.SysFont("segoeui", 20)
        font_label = pygame.font.SysFont("segoeui", 16, bold=True)
        font_value = pygame.font.SysFont("segoeui", 34, bold=True)
        font_small = pygame.font.SysFont("segoeui", 15)
        font_btn = pygame.font.SysFont("segoeui", 20, bold=True)

        valid_reps = 0
        invalid_reps = 0
        best_rep = 0.0
        for rep in self.rep_results:
            status = str(rep.get("status", "")).upper()
            sim = float(rep.get("similarity", rep.get("rehab_score", 0.0)) or 0.0)
            best_rep = max(best_rep, sim)
            if status in ("MISS", "OMITIDA", "SKIPPED", "INCORRECTO"):
                invalid_reps += 1
            else:
                valid_reps += 1
        total_reps = max(int(self.repetitions or 0), len(self.rep_results), 1)
        completed = len(self.rep_results)
        effectiveness = round((valid_reps / total_reps) * 100.0, 1) if total_reps else 0.0

        perf = self.performance_stats or {}
        rehab_summary = perf.get("rehab_summary", {}) if isinstance(perf, dict) else {}
        if isinstance(rehab_summary, dict):
            valid_reps = int(rehab_summary.get("valid_reps", valid_reps) or valid_reps)
            invalid_reps = int(rehab_summary.get("invalid_reps", invalid_reps) or invalid_reps)
            completed = int(rehab_summary.get("completed_reps", completed) or completed)
            effectiveness = float(rehab_summary.get("valid_percentage", effectiveness) or effectiveness)

        def rounded(surface, color, rect, radius=16, border=0, border_color=None):
            s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, 245), (0, 0, rect[2], rect[3]), border_radius=radius)
            if border and border_color:
                pygame.draw.rect(s, (*border_color, 255), (0, 0, rect[2], rect[3]), width=border, border_radius=radius)
            surface.blit(s, (rect[0], rect[1]))

        def metric_card(x, y, w, h, label, value, color):
            rounded(screen, CARD, (x, y, w, h), 18, 2, BORDER)
            screen.blit(font_label.render(label.upper(), True, MUTED), (x + 20, y + 16))
            screen.blit(font_value.render(str(value), True, color), (x + 20, y + 46))

        running = True
        while running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    running = False

            screen.fill(BG)
            for gx in range(0, SW, 64):
                pygame.draw.line(screen, (16, 30, 55), (gx, 0), (gx, SH), 1)
            for gy in range(0, SH, 64):
                pygame.draw.line(screen, (16, 30, 55), (0, gy), (SW, gy), 1)

            # Header
            pygame.draw.rect(screen, CARD_DARK, (0, 0, SW, 92))
            pygame.draw.line(screen, TEAL, (0, 92), (SW, 92), 2)
            screen.blit(font_title.render("Resumen de sesión", True, TEXT), (42, 24))
            screen.blit(font_sub.render("Rehabilitación Física", True, TEAL), (42, 64))

            exercise = str(self.song_key or "Ejercicio")
            right_text = font_small.render(exercise, True, MUTED)
            screen.blit(right_text, (SW - right_text.get_width() - 42, 38))

            # Main card
            main_x, main_y = 42, 124
            main_w, main_h = SW - 84, SH - 190
            rounded(screen, CARD_DARK, (main_x, main_y, main_w, main_h), 22, 2, BORDER)
            screen.blit(font_label.render("EJERCICIO", True, TEAL), (main_x + 28, main_y + 24))
            screen.blit(font_sub.render(exercise, True, TEXT), (main_x + 28, main_y + 52))

            # Metrics row
            card_w = (main_w - 76) // 4
            y = main_y + 110
            metric_card(main_x + 28, y, card_w, 120, "Reps válidas", f"{valid_reps}/{total_reps}", GREEN)
            metric_card(main_x + 48 + card_w, y, card_w, 120, "Efectividad", f"{effectiveness:.0f}%", GREEN if effectiveness >= 80 else AMBER)
            metric_card(main_x + 68 + card_w * 2, y, card_w, 120, "Mejor rep.", f"{best_rep:.0f}%", TEAL)
            metric_card(main_x + 88 + card_w * 3, y, card_w, 120, "FALLAS", invalid_reps, DANGER if invalid_reps > 0 else MUTED)

            # Barra de errores de movimiento (debajo de las métricas)
            err_y = y + 130
            err_color = DANGER if self.error_count > 0 else MUTED
            screen.blit(font_label.render("ERRORES DE MOVIMIENTO", True, TEAL), (main_x + 28, err_y))
            err_rect = pygame.Rect(main_x + 28, err_y + 24, 180, 32)
            draw_rounded_rect(screen, CARD, (err_rect.x, err_rect.y, err_rect.w, err_rect.h), 12, 245, 2, err_color)
            err_text = font_value.render(str(self.error_count), True, err_color)
            screen.blit(err_text, (err_rect.x + 16, err_rect.y + 4))
            err_label = font_small.render("movimientos incorrectos detectados", True, MUTED)
            screen.blit(err_label, (err_rect.x + err_rect.w + 16, err_rect.y + 10))

            # ── Tabla clínica de repeticiones ─────────────────────────────
            list_y = y + 150
            screen.blit(font_label.render("DETALLE DE REPETICIONES", True, TEAL), (main_x + 28, list_y))

            # Cabecera de columnas
            col_x   = [main_x + 28, main_x + 95, main_x + 210, main_x + 310, main_x + 430]
            headers = ["REP", "SCORE", "DURACIÓN", "ESTADO", "COMPENSACIONES"]
            header_y = list_y + 30
            for hx, htxt in zip(col_x, headers):
                screen.blit(font_small.render(htxt, True, MUTED), (hx, header_y))
            pygame.draw.line(screen, BORDER, (main_x + 28, header_y + 20), (main_x + main_w - 56, header_y + 20), 1)

            row_y = header_y + 28
            ROW_H = 30
            max_rows = max(1, (main_h - (row_y - main_y) - 20) // ROW_H)

            # Combinar rep_results + error_attempts ordenados por _chrono_idx
            all_entries = sorted(
                list(self.rep_results) + list(self.error_attempts),
                key=lambda e: int(e.get("_chrono_idx", 0))
            )

            if all_entries:
                for entry in all_entries[:max_rows]:
                    idx      = int(entry.get("rep_idx", 0))
                    sim      = float(entry.get("similarity", entry.get("rehab_score", 0.0)) or 0.0)
                    status   = str(entry.get("status", "")).upper()
                    dur      = entry.get("duration_s")
                    too_fast = bool(entry.get("too_fast", False))
                    comps    = entry.get("compensations") or {}
                    is_error = entry.get("error_type") == "aborted" or status == "INCORRECTO"

                    # Color de fila según estado
                    is_miss = status in ("MISS", "OMITIDA", "SKIPPED", "INCORRECTO")
                    row_col = DANGER if is_miss else (GREEN if sim >= 70 else AMBER)

                    # Zebra suave
                    if idx % 2 == 0:
                        zs = pygame.Surface((main_w - 56, ROW_H - 4), pygame.SRCALPHA)
                        zs.fill((255, 255, 255, 8))
                        screen.blit(zs, (main_x + 28, row_y))

                    # REP
                    screen.blit(font_small.render(str(idx), True, row_col), (col_x[0], row_y + 6))

                    # SCORE
                    if is_error:
                        screen.blit(font_small.render("0%", True, row_col), (col_x[1], row_y + 6))
                    else:
                        screen.blit(font_small.render(f"{sim:.0f}%", True, row_col), (col_x[1], row_y + 6))

                    # DURACIÓN
                    if dur is not None:
                        dur_txt = f"{dur:.1f}s"
                        dur_col = DANGER if too_fast else TEXT
                        flag    = " ⚡" if too_fast else ""
                        screen.blit(font_small.render(dur_txt + flag, True, dur_col), (col_x[2], row_y + 6))
                    else:
                        screen.blit(font_small.render("--", True, MUTED), (col_x[2], row_y + 6))

                    # ESTADO
                    if is_error:
                        estado_txt = "INCORRECTO"
                        estado_col = DANGER
                    elif too_fast:
                        estado_txt = "RÁPIDO"
                        estado_col = AMBER
                    elif is_miss:
                        estado_txt = "INVÁLIDA"
                        estado_col = DANGER
                    else:
                        estado_txt = status if status else "VÁLIDA"
                        estado_col = GREEN
                    screen.blit(font_small.render(estado_txt, True, estado_col), (col_x[3], row_y + 6))

                    # COMPENSACIONES
                    if isinstance(comps, dict) and any(comps.values()):
                        names = {"trunk_lean": "tronco", "elbow_bend": "codo", "shoulder_hike": "hombro"}
                        active = [names.get(k, k) for k, v in comps.items() if v]
                        comp_txt = ", ".join(active)
                    else:
                        comp_txt = "—"
                    screen.blit(font_small.render(comp_txt, True, AMBER if comp_txt != "—" else MUTED),
                                (col_x[4], row_y + 6))

                    row_y += ROW_H
            else:
                screen.blit(font_small.render("No se registraron repeticiones en esta sesión.", True, MUTED),
                            (main_x + 34, row_y + 6))

            # Footer
            hint = font_btn.render("Presiona ENTER, ESPACIO o clic para volver", True, TEXT)
            btn_w = hint.get_width() + 42
            btn_h = 48
            btn_x = SW // 2 - btn_w // 2
            btn_y = SH - 68
            rounded(screen, TEAL, (btn_x, btn_y, btn_w, btn_h), 14)
            screen.blit(hint, (btn_x + 21, btn_y + 13))

            pygame.display.flip()

    def mainloop(self):
        pass

if __name__ == "__main__":
    Score(
        score=9500,
        difficulty="NORMAL",
        max_score=12500,
        best_combo=42,
        song_key="Shoulder Flexion and Extension",
        rep_results=[
            {"rep_idx": 1, "status": "GREAT",   "similarity": 88.0, "duration_s": 7.2, "too_fast": False, "compensations": {}},
            {"rep_idx": 2, "status": "MISS",     "similarity": 42.0, "duration_s": 1.8, "too_fast": True,  "compensations": {}},
            {"rep_idx": 3, "status": "PERFECT",  "similarity": 95.0, "duration_s": 8.5, "too_fast": False, "compensations": {"trunk_lean": True}},
        ]
    )
