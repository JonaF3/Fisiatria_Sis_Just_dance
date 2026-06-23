from __future__ import annotations
"""
just_dance_gui_analytics.py — v4 (tooltips de info por pestaña)
Añade botón [?] en cada pestaña que muestra un cuadro explicativo
con qué mide cada métrica, valores de referencia y conclusión.
"""
import sys, os, csv, math, random
import pygame

# ── Paleta ─────────────────────────────────────────────────────────────────────
BG_DARK      = (6,   3,  16)
PANEL_BG     = (14,  8,  32)
CARD_BG      = (20, 12,  42)
CARD_BG2     = (28, 16,  54)
NEON_CYAN    = (0,  220, 255)
NEON_PINK    = (255,  30, 130)
NEON_YELLOW  = (255, 215,   0)
NEON_GREEN   = (0,  255, 130)
WHITE        = (255, 255, 255)
GRAY         = (145, 135, 165)
GRAY_DIM     = (80,  72, 100)
CPU_COLOR    = (70,  155, 255)
GPU_COLOR    = (255, 105,  50)
DELTA_POS    = NEON_GREEN
DELTA_NEG    = (225,  65,  75)

METRICS_DIR  = "metrics"

# ── Info de cada pestaña ───────────────────────────────────────────────────────
TAB_INFO = {
    "FPS": {
        "titulo": "¿QUÉ MIDE FPS?",
        "lineas": [
            ("¿Qué es?",        "Frames por segundo que procesa el game loop completo:"),
            ("",                "leer frame del video + detectar pose + dibujar en pantalla."),
            ("¿Por qué importa?","Más FPS = animación más fluida, menos lag visual"),
            ("",                "entre el movimiento real y lo que ve el paciente."),
            ("Resultado",       "CPU: 27.1 fps  |  GPU: 29.8 fps  (+9.9%)"),
            ("¿Por qué tan poco?","MoveNet Thunder solo tiene ~4M parámetros."),
            ("",                "La GPU lo despacha en microsegundos y queda idle."),
            ("",                "Inferencia cada 3 frames (SEND_EVERY=3) → solo ~10/seg."),
            ("Outlier S16",     "GPU llego a ~58 fps - ejercicio corto, poca inferencia."),
        ]
    },
    "LATENCIA": {
        "titulo": "¿QUÉ MIDE LATENCIA?",
        "lineas": [
            ("Latencia total",  "Tiempo completo de un ciclo del game loop:"),
            ("",                "leer frame + inferir pose + dibujar HUD."),
            ("Resultado",       "CPU: 38.9ms  |  GPU: 34.2ms  (−12.2%, GPU gana)"),
            ("Inferencia",      "Solo el tiempo que tarda MoveNet en detectar la pose."),
            ("",                "Es el cuello de botella — casi el 40% de la latencia total."),
            ("Resultado",       "CPU: 15.8ms  |  GPU: 14.4ms  (−8.9%, GPU gana)"),
            ("¿Por qué importa?","Menos latencia = la evaluación de pose corresponde"),
            ("",                "mejor al momento real del movimiento, scoring mas justo."),
        ]
    },
    "CALIDAD POSE": {
        "titulo": "¿QUÉ MIDE CALIDAD DE POSE?",
        "lineas": [
            ("Dropout KP",      "% de keypoints perdidos (confianza < umbral 0.25)."),
            ("",                "CPU: 0.7%  |  GPU: 8.6%  (+1064%, CPU gana por mucho)"),
            ("",                "GPU pierde más keypoints — inestabilidad en la detección."),
            ("Estabilidad KP",  "Qué tanto 'tiemblan' los keypoints entre frames."),
            ("",                "CPU: −0.8 (artefacto de cálculo)  |  GPU: 0.1"),
            ("",                "En la práctica GPU es más estable frame a frame."),
            ("Varianza angular", "Cuánto varían los ángulos cuando el usuario está quieto."),
            ("",                "CPU: 1.5°  |  GPU: 6.2°  (+307%, CPU gana)"),
            ("",                "GPU tiene 4× más ruido — ángulos saltan aunque no te muevas,"),
            ("",                "lo que puede perjudicar el scoring injustamente."),
        ]
    },
    "RESUMEN": {
        "titulo": "¿QUÉ MUESTRA EL RESUMEN?",
        "lineas": [
            ("¿Qué es?",        "Grid de tarjetas con el promedio de todas las sesiones."),
            ("Barra inferior",  "Proporción CPU vs GPU del valor total combinado."),
            ("Flecha verde ▲",  "GPU mejora respecto a CPU en esa métrica."),
            ("Flecha roja ▼",   "CPU es mejor — GPU empeora en esa métrica."),
            ("RAM",             "GPU consume ~27% más RAM (1020 MB vs 1294 MB)."),
            ("Frames perdidos", "GPU pierde 114% más frames — cola de inferencia se satura."),
            ("Conclusión",      "GPU gana en velocidad (FPS, latencia)."),
            ("",                "CPU gana en calidad de pose (dropout, ruido angular)."),
            ("",                "Hay un trade-off real, no una respuesta única."),
        ]
    },
    "ESTADÍSTICAS": {
        "titulo": "¿QUÉ MUESTRA LA TABLA ESTADÍSTICA?",
        "lineas": [
            ("μ (media)",       "Promedio de todas las sesiones de esa condición."),
            ("σ (desv. estándar)","Qué tan dispersos están los valores entre sesiones."),
            ("IC 95%",          "Intervalo de confianza: μ ± 1.96·(σ/√n)."),
            ("",                "Si los intervalos no se solapan → diferencia real."),
            ("Mejora %",        "(CPU−GPU)/CPU×100  (o inverso si mayor=mejor)."),
            ("Verde",           "GPU mejora estadísticamente sobre CPU."),
            ("Rojo",            "CPU es mejor — GPU no supera en esa métrica."),
            ("Clave",           "Frames perdidos GPU: 540 vs CPU: 252 (+114%)."),
            ("",                "RAM GPU: 1294 MB vs CPU: 1020 MB (+27%)."),
        ]
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(v, lo=0, hi=255): return max(lo, min(hi, int(v)))

def draw_glow_text(surface, text, font, color, x, y, glow=8, center=True):
    base = font.render(text, True, color)
    for r in range(glow, 0, -2):
        g = pygame.transform.scale(base, (base.get_width()+r*2, base.get_height()+r*2))
        g.set_alpha(_clamp(55*(1-r/glow)))
        bx = x - g.get_width()//2 if center else x - r
        by = y - g.get_height()//2 if center else y - r
        surface.blit(g, (bx, by))
    surface.blit(base, (x - base.get_width()//2 if center else x,
                         y - base.get_height()//2 if center else y))

def rr(surface, color, rect, radius=12, alpha=255, border=0, bc=None):
    w, h = max(1, rect[2]), max(1, rect[3])
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0, 0, w, h), border_radius=radius)
    if border and bc:
        pygame.draw.rect(s, (*bc, alpha), (0, 0, w, h), width=border, border_radius=radius)
    surface.blit(s, (rect[0], rect[1]))

def draw_gradient_bar(surface, rect, color_a, color_b, radius=4):
    x, y, w, h = rect
    for i in range(w):
        t = i / max(w-1, 1)
        c = tuple(_clamp(color_a[j]*(1-t) + color_b[j]*t) for j in range(3))
        pygame.draw.line(surface, c, (x+i, y), (x+i, y+h-1))

class Particle:
    def __init__(self, W, H):
        self.W=W; self.H=H; self.reset()
    def reset(self):
        self.x=random.randint(0,self.W); self.y=random.randint(0,self.H)
        self.size=random.uniform(1,2.5); self.speed=random.uniform(.15,.5)
        self.alpha=random.randint(20,110)
        self.color=random.choice([NEON_CYAN,NEON_PINK,NEON_YELLOW])
        self.drift=random.uniform(-.2,.2)
    def update(self):
        self.y-=self.speed; self.x+=self.drift; self.alpha-=.25
        if self.y<0 or self.alpha<=0: self.reset(); self.y=self.H
    def draw(self, surface):
        if self.alpha<=0: return
        s=pygame.Surface((int(self.size*2+1),int(self.size*2+1)),pygame.SRCALPHA)
        pygame.draw.circle(s,(*self.color,int(self.alpha)),(int(self.size),int(self.size)),int(self.size))
        surface.blit(s,(int(self.x),int(self.y)))

# ── Info Tooltip ───────────────────────────────────────────────────────────────

def draw_info_modal(surface, tab_name, SW, SH):
    """
    Dibuja un cuadro modal semi-transparente con la explicación de la pestaña.
    """
    info = TAB_INFO.get(tab_name, {})
    if not info:
        return

    lineas = info.get("lineas", [])
    titulo = info.get("titulo", "INFO")

    # Dimensiones del modal
    MOD_W = min(680, SW - 80)
    LINE_H = 26
    PAD = 28
    HDR_H = 52
    MOD_H = HDR_H + len(lineas) * LINE_H + PAD * 2 + 20

    mx = SW // 2 - MOD_W // 2
    my = SH // 2 - MOD_H // 2

    # Overlay oscuro de fondo
    overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    # Sombra del modal
    rr(surface, (0, 0, 0), (mx + 8, my + 8, MOD_W, MOD_H), 18, alpha=120)

    # Fondo del modal
    rr(surface, (16, 8, 36), (mx, my, MOD_W, MOD_H), 18, alpha=255,
       border=2, bc=NEON_CYAN)

    # Línea de acento superior
    pygame.draw.rect(surface, NEON_CYAN, (mx, my, MOD_W, 4), border_radius=18)

    # Título
    tf = pygame.font.SysFont("impact", 22)
    ts = tf.render(titulo, True, NEON_CYAN)
    surface.blit(ts, (mx + PAD, my + 14))

    # Línea separadora
    pygame.draw.line(surface, (50, 35, 80),
                     (mx + PAD, my + HDR_H),
                     (mx + MOD_W - PAD, my + HDR_H), 1)

    # Contenido
    label_f = pygame.font.SysFont("trebuchetms", 14, bold=True)
    val_f   = pygame.font.SysFont("trebuchetms", 14)
    LABEL_W = 175

    for i, (label, texto) in enumerate(lineas):
        ly = my + HDR_H + PAD // 2 + i * LINE_H

        if label:
            ls = label_f.render(label, True, NEON_YELLOW)
            surface.blit(ls, (mx + PAD, ly))

        vs = val_f.render(texto, True, WHITE)
        surface.blit(vs, (mx + PAD + LABEL_W, ly))

    # Hint cerrar
    hf = pygame.font.SysFont("trebuchetms", 13)
    hs = hf.render("Click o ESC para cerrar", True, (80, 70, 105))
    surface.blit(hs, (mx + MOD_W // 2 - hs.get_width() // 2,
                      my + MOD_H - 20))


def draw_info_button(surface, x, y, hovered=False):
    """Dibuja el botón [?] que abre el modal."""
    R = 14
    col = NEON_CYAN if hovered else (80, 65, 110)
    bg  = (30, 18, 55) if hovered else (18, 10, 34)
    pygame.draw.circle(surface, bg, (x, y), R)
    pygame.draw.circle(surface, col, (x, y), R, 2)
    f = pygame.font.SysFont("impact", 18)
    s = f.render("?", True, col)
    surface.blit(s, (x - s.get_width() // 2, y - s.get_height() // 2))
    return pygame.Rect(x - R, y - R, R * 2, R * 2)

# ── Data ───────────────────────────────────────────────────────────────────────

def _sf(v, d=0.0):
    try: return float(v)
    except: return d

def load_sessions(md=METRICS_DIR):
    sessions=[]
    if not os.path.isdir(md): return sessions
    for fname in sorted(os.listdir(md)):
        if not fname.endswith(".csv"): continue
        try:
            with open(os.path.join(md,fname),newline="",encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["_file"]=fname; sessions.append(row)
        except Exception as e: print(f"[WARN] {fname}: {e}")
    return sessions

NUMERIC_KEYS=[
    "fps_avg","latency_avg_ms","inference_avg_ms",
    "cpu_avg_pct","gpu_avg_pct","ram_avg_mb",
    "late_frames","dropped_frames","kp_dropout_avg",
    "kp_stability","angular_variability_avg_deg","trajectory_consistency","final_score",
]

def aggregate(sessions):
    g={"cpu":{k:[] for k in NUMERIC_KEYS},"gpu":{k:[] for k in NUMERIC_KEYS}}
    for s in sessions:
        cond=str(s.get("condition","")).lower()
        if cond not in g:
            fn=s.get("_file","").lower()
            cond="cpu" if "cpu" in fn else ("gpu" if "gpu" in fn else None)
            if cond is None: continue
        for k in NUMERIC_KEYS:
            v=_sf(s.get(k,""),d=None)
            if v is not None: g[cond][k].append(v)
    return g

def mean(lst): return sum(lst)/len(lst) if lst else 0.0
def std(lst):
    if len(lst)<2: return 0.0
    m=mean(lst); return math.sqrt(sum((x-m)**2 for x in lst)/(len(lst)-1))
def ic95(lst): return 1.96*std(lst)/math.sqrt(len(lst)) if len(lst)>=2 else 0.0
def build_summary(g): return {c:{k:mean(v) for k,v in m.items()} for c,m in g.items()}

# ── Bar chart ──────────────────────────────────────────────────────────────────

def draw_bar_chart(surface, rect, cpu_vals, gpu_vals, labels,
                   title, unit="", higher=True):
    x,y,w,h=rect
    rr(surface,CARD_BG,rect,16,alpha=235,border=1,bc=(55,38,85))

    tf=pygame.font.SysFont("trebuchetms",17,bold=True)
    surface.blit(tf.render(title,True,NEON_CYAN),(x+18,y+16))

    if not labels:
        nf=pygame.font.SysFont("trebuchetms",16)
        ns=nf.render("Sin datos",True,GRAY)
        surface.blit(ns,(x+w//2-ns.get_width()//2,y+h//2)); return

    PL,PR,PT,PB=62,18,56,50
    cx,cy,cw,ch=x+PL,y+PT,w-PL-PR,h-PT-PB

    all_v=[v for v in cpu_vals+gpu_vals if v is not None]
    if not all_v: return
    mv=max(all_v)*1.18 or 1.0

    def px(v): return cy+ch-int(v/mv*ch)

    sf=pygame.font.SysFont("trebuchetms",13)
    for i in range(5):
        gy=cy+int(i*ch/4)
        pygame.draw.line(surface,(38,28,62),(cx,gy),(cx+cw,gy),1)
        vl=f"{mv*(4-i)/4:.1f}"
        ls=sf.render(vl,True,(100,88,128))
        surface.blit(ls,(x+4,gy-ls.get_height()//2))

    n=len(labels); gw=cw/n; bw=max(8,int(gw*.33)); gap=4
    for i,lbl in enumerate(labels):
        gx=cx+int(i*gw)+int(gw*.12)
        cv_=cpu_vals[i] if i<len(cpu_vals) and cpu_vals[i] is not None else 0.
        gv_=gpu_vals[i] if i<len(gpu_vals) and gpu_vals[i] is not None else 0.

        ct=px(cv_); cb=cy+ch
        if cb>ct:
            rr(surface,CPU_COLOR,(gx,ct,bw,cb-ct),4,alpha=215)
        gt=px(gv_); gb=cy+ch
        if gb>gt:
            rr(surface,GPU_COLOR,(gx+bw+gap,gt,bw,gb-gt),4,alpha=215)

        if cv_>0:
            dp=(gv_-cv_)/cv_*100; better=(dp>0)==higher
            arrow="▲" if dp>0 else "▼"; col=DELTA_POS if better else DELTA_NEG
            df=pygame.font.SysFont("trebuchetms",13,bold=True)
            ds=df.render(f"{arrow}{abs(dp):.0f}%",True,col)
            surface.blit(ds,(gx+bw-ds.get_width()//2,min(ct,gt)-22))

        ls=sf.render(str(lbl)[:8],True,GRAY)
        surface.blit(ls,(gx-ls.get_width()//4,cy+ch+8))

    pygame.draw.line(surface,(55,42,88),(cx,cy),(cx,cy+ch),1)
    pygame.draw.line(surface,(55,42,88),(cx,cy+ch),(cx+cw,cy+ch),1)
    uf=sf.render(unit,True,(100,88,128)); surface.blit(uf,(x+4,y+20))

# ── Summary Cards ──────────────────────────────────────────────────────────────

SUMMARY_ROWS=[
    ("FPS promedio",         "fps_avg",                    True,  "fps"),
    ("Latencia total",       "latency_avg_ms",             False, "ms"),
    ("Inferencia",           "inference_avg_ms",           False, "ms"),
    ("CPU uso",              "cpu_avg_pct",                False, "%"),
    ("GPU uso",              "gpu_avg_pct",                False, "%"),
    ("RAM",                  "ram_avg_mb",                 False, "MB"),
    ("Frames tardíos",       "late_frames",                False, ""),
    ("Frames perdidos",      "dropped_frames",             False, ""),
    ("Dropout keypoints",    "kp_dropout_avg",             False, "%"),
    ("Estabilidad KP",       "kp_stability",               True,  ""),
    ("Var. angular",         "angular_variability_avg_deg",False, "°"),
    ("Consist. tray.",       "trajectory_consistency",     True,  ""),
]

def draw_metric_card(surface, rect, label, cpu_v, gpu_v, unit, higher, pulse):
    x,y,w,h=rect
    rr(surface,CARD_BG2,rect,14,alpha=238,border=1,bc=(52,36,82))

    nf=pygame.font.SysFont("trebuchetms",13,bold=True)
    ns=nf.render(label.upper(),True,GRAY)
    surface.blit(ns,(x+12,y+10))

    vf=pygame.font.SysFont("impact",28)
    def fmtv(v):
        if v is None: return "--"
        if unit in ("ms","fps","%","°"): return f"{v:.1f}"
        if unit=="MB": return f"{v:.0f}"
        return f"{v:.2f}"

    cv_str=fmtv(cpu_v)+(f" {unit}" if unit else "")
    gv_str=fmtv(gpu_v)+(f" {unit}" if unit else "")

    cs=vf.render(cv_str,True,CPU_COLOR)
    gs_=vf.render(gv_str,True,GPU_COLOR)

    half=w//2
    surface.blit(cs,(x+half//2-cs.get_width()//2, y+28))
    surface.blit(gs_,(x+half+half//2-gs_.get_width()//2, y+28))

    lf=pygame.font.SysFont("trebuchetms",11,bold=True)
    cl=lf.render("CPU",True,CPU_COLOR); gl=lf.render("GPU",True,GPU_COLOR)
    surface.blit(cl,(x+half//2-cl.get_width()//2, y+60))
    surface.blit(gl,(x+half+half//2-gl.get_width()//2, y+60))

    pygame.draw.line(surface,(52,36,82),(x+half,y+26),(x+half,y+68),1)

    BAR_Y=y+h-26; BAR_H=10; BAR_X=x+12; BAR_W=w-24
    rr(surface,(32,22,55),(BAR_X,BAR_Y,BAR_W,BAR_H),5,alpha=200)

    if cpu_v is not None and gpu_v is not None and (cpu_v+gpu_v)>0:
        total=cpu_v+gpu_v
        cpu_frac=cpu_v/total
        cpu_px=int(BAR_W*cpu_frac); gpu_px=BAR_W-cpu_px
        if cpu_px>0: rr(surface,CPU_COLOR,(BAR_X,BAR_Y,cpu_px,BAR_H),5,alpha=210)
        if gpu_px>0: rr(surface,GPU_COLOR,(BAR_X+cpu_px,BAR_Y,gpu_px,BAR_H),5,alpha=210)

    if cpu_v is not None and gpu_v is not None and cpu_v!=0:
        delta=(gpu_v-cpu_v)/abs(cpu_v)*100
        better=(delta>0)==higher
        arrow="▲" if delta>0 else "▼"
        col=DELTA_POS if better else DELTA_NEG
        pf=pygame.font.SysFont("trebuchetms",13,bold=True)
        ps=pf.render(f"{arrow}{abs(delta):.1f}%",True,col)
        pill_w=ps.get_width()+16; pill_h=20
        pill_x=x+w-pill_w-8; pill_y=y+h-pill_h-30
        rr(surface,(10,6,22),(pill_x,pill_y,pill_w,pill_h),10,alpha=220,border=1,bc=col)
        surface.blit(ps,(pill_x+8,pill_y+2))


def draw_summary_grid(surface, rect, summary, pulse):
    x,y,w,h=rect
    cpu_d=summary.get("cpu",{}); gpu_d=summary.get("gpu",{})
    COLS=4; rows=math.ceil(len(SUMMARY_ROWS)/COLS)
    gap=10
    card_w=(w-gap*(COLS-1))//COLS
    card_h=(h-gap*(rows-1))//rows

    for i,(label,key,higher,unit) in enumerate(SUMMARY_ROWS):
        col=i%COLS; row=i//COLS
        cx=x+col*(card_w+gap); cy=y+row*(card_h+gap)
        draw_metric_card(surface,(cx,cy,card_w,card_h),
                         label,cpu_d.get(key),gpu_d.get(key),unit,higher,pulse)

# ── Stats table ────────────────────────────────────────────────────────────────

STATS_ROWS=[
    ("FPS promedio",         "fps_avg",                     True,  "fps"),
    ("Latencia total",       "latency_avg_ms",              False, "ms"),
    ("Inferencia",           "inference_avg_ms",            False, "ms"),
    ("CPU uso",              "cpu_avg_pct",                 False, "%"),
    ("GPU uso",              "gpu_avg_pct",                 False, "%"),
    ("RAM",                  "ram_avg_mb",                  False, "MB"),
    ("Frames tardíos",       "late_frames",                 False, ""),
    ("Frames perdidos",      "dropped_frames",              False, ""),
    ("Dropout KP",           "kp_dropout_avg",              False, "%"),
    ("Estabilidad KP",       "kp_stability",                True,  ""),
    ("Var. angular",         "angular_variability_avg_deg", False, "°"),
    ("Consist. tray.",       "trajectory_consistency",      True,  ""),
]

def draw_stats_table(surface, rect, groups, scroll):
    x,y,w,h=rect
    rr(surface,CARD_BG,rect,16,alpha=235,border=1,bc=(52,36,82))

    tf=pygame.font.SysFont("trebuchetms",18,bold=True)
    ts=tf.render("ANÁLISIS ESTADÍSTICO COMPARATIVO  ·  media · σ · IC 95%",True,NEON_CYAN)
    surface.blit(ts,(x+18,y+16))

    COLS=["MÉTRICA","CPU μ","CPU σ","IC 95%","GPU μ","GPU σ","IC 95%","MEJORA %"]
    CW  =[180,     90,    78,    96,      90,    78,    96,     96]

    hf=pygame.font.SysFont("trebuchetms",15,bold=True)
    sf=pygame.font.SysFont("trebuchetms",16)

    HDR_Y=y+46; ROW_H=38; NOTE_H=22

    cx=x+16
    for hdr,cw in zip(COLS,CW):
        col=(CPU_COLOR if "CPU" in hdr else
             GPU_COLOR if "GPU" in hdr else
             NEON_YELLOW if "MEJORA" in hdr else GRAY)
        surface.blit(hf.render(hdr,True,col),(cx,HDR_Y)); cx+=cw
    pygame.draw.line(surface,(55,38,85),(x+10,HDR_Y+22),(x+w-10,HDR_Y+22),1)

    table_top=HDR_Y+28
    vis_h=h-(table_top-y)-NOTE_H-12
    max_scroll=max(0,len(STATS_ROWS)*ROW_H-vis_h)
    scroll=min(scroll,max_scroll)

    clip=pygame.Rect(x+8,table_top,w-20,vis_h)
    surface.set_clip(clip)

    for ri,(label,key,higher,unit) in enumerate(STATS_ROWS):
        ry=table_top+ri*ROW_H-scroll
        if ry+ROW_H<table_top or ry>table_top+vis_h: continue

        cl=groups.get("cpu",{}).get(key,[])
        gl=groups.get("gpu",{}).get(key,[])
        cm=mean(cl); cs=std(cl); ci=ic95(cl)
        gm=mean(gl); gs_=std(gl); gi=ic95(gl)

        row_col=CPU_COLOR if ri%2==0 else GPU_COLOR
        rr(surface,(28,18,50),(x+10,ry,w-20,ROW_H-2),6,alpha=120)
        rr(surface,row_col,(x+10,ry,4,ROW_H-2),2,alpha=180)

        if cm!=0 and (cl or gl):
            mejora=((gm-cm)/abs(cm)*100) if higher else ((cm-gm)/abs(cm)*100)
            better=mejora>0; arrow="▲" if mejora>0 else "▼"
            mstr=f"{arrow}{abs(mejora):.1f}%"; mcol=DELTA_POS if better else DELTA_NEG
        else: mstr="--"; mcol=GRAY

        def fmt(v,u):
            if not v and v!=0: return "--"
            if u in("ms","fps","%","°"): return f"{v:.2f}"
            if u=="MB": return f"{v:.0f}"
            return f"{v:.4f}"

        row_vals=[
            (label,              WHITE),
            (fmt(cm,unit),       CPU_COLOR),
            (f"±{fmt(cs,unit)}", (100,145,225)),
            (f"±{fmt(ci,unit)}", (80, 125,205)),
            (fmt(gm,unit),       GPU_COLOR),
            (f"±{fmt(gs_,unit)}",(225,140, 80)),
            (f"±{fmt(gi,unit)}", (205,120, 60)),
            (mstr,               mcol),
        ]
        cx=x+16
        for (txt,col),cw in zip(row_vals,CW):
            surface.blit(sf.render(str(txt),True,col),(cx,ry+10)); cx+=cw

    surface.set_clip(None)

    if max_scroll>0:
        sbx=x+w-12; sby=table_top; sbh=vis_h
        rr(surface,(30,20,50),(sbx,sby,6,sbh),3,alpha=180)
        th=max(28,int(sbh*vis_h/(len(STATS_ROWS)*ROW_H)))
        ty=sby+int((scroll/max_scroll)*(sbh-th)) if max_scroll else sby
        rr(surface,NEON_CYAN,(sbx,ty,6,th),3,alpha=230)

    nf=pygame.font.SysFont("trebuchetms",12)
    note=nf.render(
        "μ = media  ·  σ = desviación estándar  ·  IC 95% = μ ± 1.96·(σ/√n)  ·  "
        "Mejora % = (CPU−GPU)/CPU×100  (o inverso si mayor=mejor)",True,(75,62,105))
    surface.blit(note,(x+12,y+h-NOTE_H))

# ── Legend ─────────────────────────────────────────────────────────────────────

def draw_legend(surface, x, y):
    lf=pygame.font.SysFont("trebuchetms",15,bold=True)
    rr(surface,CPU_COLOR,(x,y+5,18,13),4); surface.blit(lf.render("CPU",True,CPU_COLOR),(x+24,y))
    rr(surface,GPU_COLOR,(x+82,y+5,18,13),4); surface.blit(lf.render("GPU",True,GPU_COLOR),(x+106,y))

# ── Single metric tab ──────────────────────────────────────────────────────────

def draw_single_metric(surface, x, y, w, h, groups, summary,
                       metric, title, unit, higher, note=""):
    chart_w=int(w*.67); side_w=w-chart_w-18
    cpu_v=groups.get("cpu",{}).get(metric,[])
    gpu_v=groups.get("gpu",{}).get(metric,[])
    n=max(len(cpu_v),len(gpu_v))
    cp=cpu_v+[None]*(n-len(cpu_v)); gp=gpu_v+[None]*(n-len(gpu_v))
    lbls=[f"S{i+1}" for i in range(n)]
    draw_bar_chart(surface,(x,y,chart_w,h),cp,gp,lbls,title,unit,higher)

    sx=x+chart_w+18
    rr(surface,CARD_BG,(sx,y,side_w,h),16,alpha=235,border=1,bc=(52,36,82))
    tf=pygame.font.SysFont("trebuchetms",15,bold=True)
    vf=pygame.font.SysFont("impact",38)
    sf=pygame.font.SysFont("trebuchetms",14)
    oy=y+22
    for cond,col,lbl in[("cpu",CPU_COLOR,"CPU"),("gpu",GPU_COLOR,"GPU")]:
        v=summary.get(cond,{}).get(metric)
        surface.blit(tf.render(f"  {lbl} promedio",True,col),(sx+14,oy)); oy+=24
        vs=f"{v:.1f} {unit}" if v is not None else "--"
        surface.blit(vf.render(vs,True,col),(sx+14,oy)); oy+=50

    cv=summary.get("cpu",{}).get(metric); gv=summary.get("gpu",{}).get(metric)
    if cv and gv and cv!=0:
        delta=(gv-cv)/abs(cv)*100; better=(delta>0)==higher
        arrow="▲" if delta>0 else "▼"; col=DELTA_POS if better else DELTA_NEG
        pygame.draw.line(surface,(52,36,82),(sx+14,oy),(sx+side_w-14,oy),1); oy+=16
        surface.blit(tf.render("DELTA GPU vs CPU",True,GRAY),(sx+14,oy)); oy+=24
        df=pygame.font.SysFont("impact",36)
        surface.blit(df.render(f"{arrow} {abs(delta):.1f}%",True,col),(sx+14,oy)); oy+=50
        surface.blit(sf.render("GPU mejor" if better else "CPU mejor",True,col),(sx+14,oy)); oy+=28

    if note:
        oy+=10
        pygame.draw.line(surface,(52,36,82),(sx+14,oy),(sx+side_w-14,oy),1); oy+=14
        words=note.split(); line=""; mw=side_w-28
        for word in words:
            cand=f"{line} {word}".strip()
            if sf.size(cand)[0]<=mw: line=cand
            else:
                surface.blit(sf.render(line,True,(125,115,155)),(sx+14,oy)); oy+=19; line=word
        if line: surface.blit(sf.render(line,True,(125,115,155)),(sx+14,oy))

def draw_two_metrics(surface, x, y, w, h, groups, summary,
                     metric, title, unit, higher):
    cv=groups.get("cpu",{}).get(metric,[]); gv=groups.get("gpu",{}).get(metric,[])
    n=max(len(cv),len(gv),1)
    cp=cv+[None]*(n-len(cv)); gp=gv+[None]*(n-len(gv)); lbls=[f"S{i+1}" for i in range(n)]
    ch=int(h*.72)
    draw_bar_chart(surface,(x,y,w,ch),cp,gp,lbls,title,unit,higher)
    by=y+ch+10; bh=h-ch-10
    rr(surface,CARD_BG,(x,by,w,bh),12,alpha=228,border=1,bc=(52,36,82))
    vf=pygame.font.SysFont("impact",24); sf=pygame.font.SysFont("trebuchetms",14,bold=True)
    ox=x+14
    for cond,col in[("cpu",CPU_COLOR),("gpu",GPU_COLOR)]:
        v=summary.get(cond,{}).get(metric)
        vs=f"{v:.1f} {unit}" if v is not None else "--"
        surface.blit(vf.render(vs,True,col),(ox,by+10)); ox+=vf.size(vs)[0]+20
    cv2=summary.get("cpu",{}).get(metric); gv2=summary.get("gpu",{}).get(metric)
    if cv2 and gv2 and cv2!=0:
        d=(gv2-cv2)/abs(cv2)*100; b=(d>0)==higher
        arrow="▲" if d>0 else "▼"; col=DELTA_POS if b else DELTA_NEG
        ds=sf.render(f"{arrow}{abs(d):.1f}%",True,col)
        surface.blit(ds,(x+w-ds.get_width()-14,by+12))

# ── Main class ─────────────────────────────────────────────────────────────────

class Analytics:
    TABS=["FPS","LATENCIA","CALIDAD POSE","RESUMEN","ESTADÍSTICAS"]

    def __init__(self,screen_w=1280,screen_h=720):
        self.SW=screen_w; self.SH=screen_h; self.run()

    def run(self):
        SW,SH=self.SW,self.SH
        pygame.init()
        screen=pygame.display.get_surface()
        if screen is None:
            screen=pygame.display.set_mode((SW,SH),pygame.NOFRAME)
            pygame.display.set_caption("Rehabilitacion Fisica - Analisis")
        clock=pygame.time.Clock()

        font_title=pygame.font.SysFont("impact",54)
        font_tab  =pygame.font.SysFont("impact",22)
        font_btn  =pygame.font.SysFont("impact",24)
        font_hint =pygame.font.SysFont("trebuchetms",14)

        particles=[Particle(SW,SH) for _ in range(90)]
        tick=0; fade_alpha=255; stats_scroll=0; view=0
        show_info = False   # ← estado del modal de info

        sessions=load_sessions()
        groups  =aggregate(sessions)
        summary =build_summary(groups)

        n_cpu=sum(1 for s in sessions if "cpu" in str(s.get("condition","")).lower()+" "+s.get("_file","").lower())
        n_gpu=sum(1 for s in sessions if "gpu" in str(s.get("condition","")).lower()+" "+s.get("_file","").lower())

        TAB_Y=72; TAB_H=54; CARD_X=30
        CARD_Y=TAB_Y+TAB_H+18; CARD_W=SW-60; CARD_H=SH-CARD_Y-82
        BTN_Y=SH-64; BTN_H=50

        n_tabs=len(self.TABS)
        tab_w=min(186,(SW-60)//n_tabs-8)
        tabs=[pygame.Rect(CARD_X+i*(tab_w+8),TAB_Y,tab_w,TAB_H) for i in range(n_tabs)]
        BTN_QT=pygame.Rect(SW-148,BTN_Y,130,BTN_H)

        # Botón [?] — a la derecha de los tabs
        INFO_BTN_X = CARD_X + n_tabs*(tab_w+8) + 22
        INFO_BTN_Y = TAB_Y + TAB_H//2

        running=True
        while running:
            clock.tick(60); tick+=1
            mx,my=pygame.mouse.get_pos()
            if fade_alpha>0: fade_alpha=max(0,fade_alpha-14)
            pulse=math.sin(tick*.06)*.5+.5

            for event in pygame.event.get():
                if event.type==pygame.QUIT: pygame.quit(); sys.exit()
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:
                        if show_info:
                            show_info = False
                        else:
                            return
                    if not show_info:
                        if event.key==pygame.K_TAB:
                            view=(view+1)%len(self.TABS); stats_scroll=0
                if event.type==pygame.MOUSEWHEEL and view==4 and not show_info:
                    stats_scroll=max(0,stats_scroll-event.y*36)
                if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    if show_info:
                        # Cualquier click cierra el modal
                        show_info = False
                        continue
                    if BTN_QT.collidepoint(mx,my): return
                    # Botón [?]
                    info_rect = pygame.Rect(INFO_BTN_X-14, INFO_BTN_Y-14, 28, 28)
                    if info_rect.collidepoint(mx,my):
                        show_info = True
                        continue
                    for i,tr in enumerate(tabs):
                        if tr.collidepoint(mx,my): view=i; stats_scroll=0; break

            # Fondo
            screen.fill(BG_DARK)
            for gx_ in range(0,SW,64): pygame.draw.line(screen,(28,16,50),(gx_,0),(gx_,SH),1)
            for gy_ in range(0,SH,64): pygame.draw.line(screen,(28,16,50),(0,gy_),(SW,gy_),1)
            for p in particles: p.update(); p.draw(screen)

            # Título
            tc=tuple(_clamp(NEON_CYAN[j]*.75+NEON_CYAN[j]*.25*pulse) for j in range(3))
            draw_glow_text(screen,"ANÁLISIS DE SESIONES",font_title,tc,SW//2,36,glow=12)
            ns=pygame.font.SysFont("trebuchetms",14).render(
                f"  {n_cpu} sesiones CPU  ·  {n_gpu} sesiones GPU  ",True,GRAY)
            screen.blit(ns,(SW//2-ns.get_width()//2,62))

            # Tabs
            for i,(tr,lbl) in enumerate(zip(tabs,self.TABS)):
                active=(i==view); hov=tr.collidepoint(mx,my)
                if active:
                    gs=pygame.Surface((tr.width+16,tr.height+16),pygame.SRCALPHA)
                    pygame.draw.rect(gs,(*NEON_CYAN,int(22+16*pulse)),(0,0,tr.width+16,tr.height+16),border_radius=16)
                    screen.blit(gs,(tr.x-8,tr.y-8))
                rr(screen,(42,22,72) if active else (18,10,36),tr,14,alpha=240,
                   border=2,bc=NEON_CYAN if active else ((80,55,110) if hov else (45,30,72)))
                col=WHITE if active else (GRAY if not hov else (200,190,220))
                draw_glow_text(screen,lbl,font_tab,col,tr.centerx,tr.centery,glow=5 if active else 1)

            # Botón [?]
            info_hov = pygame.Rect(INFO_BTN_X-14, INFO_BTN_Y-14, 28, 28).collidepoint(mx, my)
            draw_info_button(screen, INFO_BTN_X, INFO_BTN_Y, hovered=info_hov)

            # Leyenda
            draw_legend(screen,CARD_X,TAB_Y+TAB_H+20)

            content_y=CARD_Y+20

            if view==0:
                draw_single_metric(screen,CARD_X,content_y,CARD_W,CARD_H-20,
                    groups,summary,"fps_avg","FPS PROMEDIO POR SESIÓN","fps",True,
                    "Mayor FPS → animación más fluida. GPU debería superar a CPU con MoveNet.")
            elif view==1:
                hw=CARD_W//2-10
                draw_two_metrics(screen,CARD_X,content_y,hw,CARD_H-20,
                    groups,summary,"latency_avg_ms","LATENCIA TOTAL (ms)","ms",False)
                draw_two_metrics(screen,CARD_X+hw+20,content_y,hw,CARD_H-20,
                    groups,summary,"inference_avg_ms","INFERENCIA (ms)","ms",False)
            elif view==2:
                tw=(CARD_W-40)//3
                for mi,(mk,mt,mu,mh) in enumerate([
                    ("kp_dropout_avg","DROPOUT KEYPOINTS (%)","%" ,False),
                    ("kp_stability","ESTABILIDAD KP","",True),
                    ("angular_variability_avg_deg","VAR. ANGULAR (°)","°",False)]):
                    draw_two_metrics(screen,CARD_X+mi*(tw+20),content_y,tw,CARD_H-20,
                        groups,summary,mk,mt,mu,mh)
            elif view==3:
                draw_summary_grid(screen,(CARD_X,content_y,CARD_W,CARD_H-20),summary,pulse)
            elif view==4:
                draw_stats_table(screen,(CARD_X,content_y,CARD_W,CARD_H-20),groups,stats_scroll)

            if not sessions:
                mf=pygame.font.SysFont("trebuchetms",22)
                ms_=mf.render("No se encontraron archivos CSV en metrics/",True,NEON_YELLOW)
                screen.blit(ms_,(SW//2-ms_.get_width()//2,SH//2))
                sf=pygame.font.SysFont("trebuchetms",16)
                ss=sf.render("Ejecuta al menos una sesión para ver análisis.",True,GRAY)
                screen.blit(ss,(SW//2-ss.get_width()//2,SH//2+36))

            # Botón SALIR
            hov_q=BTN_QT.collidepoint(mx,my)
            if hov_q:
                gs=pygame.Surface((BTN_QT.width+16,BTN_QT.height+16),pygame.SRCALPHA)
                pygame.draw.rect(gs,(220,60,60,40),(0,0,BTN_QT.width+16,BTN_QT.height+16),border_radius=16)
                screen.blit(gs,(BTN_QT.x-8,BTN_QT.y-8))
            rr(screen,(18,10,36),BTN_QT,14,alpha=230,border=2,
               bc=(230,65,65) if hov_q else (55,38,80))
            draw_glow_text(screen,"SALIR",font_btn,
                           (235,75,75) if hov_q else GRAY,
                           BTN_QT.centerx,BTN_QT.centery,glow=5 if hov_q else 1)

            hint_text = "Click o ESC para cerrar info" if show_info else "TAB: siguiente pestaña  |  ESC: volver al lobby  |  [?] info de esta pestaña"
            hint=font_hint.render(hint_text,True,(55,45,75))
            screen.blit(hint,(SW//2-hint.get_width()//2,SH-20))

            # Modal de info — siempre encima de todo
            if show_info:
                draw_info_modal(screen, self.TABS[view], SW, SH)

            if fade_alpha>0:
                fs=pygame.Surface((SW,SH)); fs.fill((0,0,0)); fs.set_alpha(fade_alpha)
                screen.blit(fs,(0,0))

            pygame.display.flip()


# ── Entry point ────────────────────────────────────────────────────────────────

def run_analytics(W=1280,H=720):
    Analytics(screen_w=W,screen_h=H)

if __name__=="__main__":
    pygame.init()
    pygame.display.set_mode((1280,720),pygame.NOFRAME)
    run_analytics(1280,720)
    pygame.quit()
