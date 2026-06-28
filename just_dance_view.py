from __future__ import annotations
"""
just_dance_view.py — v4 (UI overhaul)
HUD con avatar de paciente, referencia de record e indicador de modo practica.
"""
import math, random, cv2, numpy as np


# ── Sistema de partículas ──────────────────────────────────────────────────────

class EffectParticle:
    __slots__ = ("x","y","vx","vy","color","size","life","decay")
    def __init__(self, x, y, color):
        self.x=x; self.y=y
        self.vx=random.uniform(-9,9); self.vy=random.uniform(-14,-3)
        self.color=color; self.size=random.randint(4,9)
        self.life=1.0; self.decay=random.uniform(0.03,0.07)
    def update(self):
        self.x+=self.vx; self.y+=self.vy
        self.vy+=0.55; self.vx*=0.97; self.life-=self.decay
        return self.life>0
    def draw(self, frame):
        if self.life<=0: return
        c=tuple(int(ch*self.life) for ch in self.color)
        cv2.circle(frame,(int(self.x),int(self.y)),self.size,c,-1)


def spawn_rating_particles(cx, cy, rating, count=40):
    COLOR_MAP={
        "PERFECT":[(255,215,0),(255,255,100),(255,180,0)],
        "GREAT":  [(0,255,80),(80,255,160),(0,200,80)],
        "GOOD":   [(0,255,255),(0,200,220),(80,220,255)],
        "OK":     [(0,165,255),(0,130,200),(80,160,255)],
    }
    colors=COLOR_MAP.get(rating,[(255,255,255)])
    return [EffectParticle(cx+random.randint(-30,30),
                           cy+random.randint(-20,20),
                           random.choice(colors))
            for _ in range(count)]


def update_and_draw_particles(frame, particles):
    alive=[]
    for p in particles:
        if p.update(): p.draw(frame); alive.append(p)
    return alive


# ── Helpers internos ───────────────────────────────────────────────────────────

def _draw_star(frame, cx, cy, r_outer, color, filled=True):
    r_inner=int(r_outer*0.42)
    pts=[]
    for i in range(10):
        angle=math.radians(-90+i*36)
        r=r_outer if i%2==0 else r_inner
        pts.append([int(cx+r*math.cos(angle)), int(cy+r*math.sin(angle))])
    pts=np.array([pts],dtype=np.int32)
    if filled: cv2.fillPoly(frame,pts,color)
    else:       cv2.polylines(frame,pts,True,color,1)


def _draw_avatar_cv2(frame, cx, cy, r, color_bgr, initials="?"):
    """Dibuja un círculo de avatar con iniciales en OpenCV."""
    cv2.circle(frame,(cx,cy),r,color_bgr,-1)
    cv2.circle(frame,(cx,cy),r,(255,255,255),2)
    fsz=max(0.4, r/20)
    tsz=cv2.getTextSize(initials,cv2.FONT_HERSHEY_DUPLEX,fsz,1)[0]
    tx=cx-tsz[0]//2; ty=cy+tsz[1]//2
    cv2.putText(frame,initials,(tx,ty),cv2.FONT_HERSHEY_DUPLEX,fsz,(15,8,30),1)


# ── Clase principal ─────────────────────────────────────────────────────────────

class JustDanceView:

    SKELETON_CONNECTIONS=[
        (0,1),(0,2),(1,3),(2,4),(5,6),
        (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]

    RATING_COLORS={
        "PERFECT":(255,215,0),"GREAT":(0,255,80),
        "GOOD":(0,255,255),"OK":(0,165,255),"MISS":(60,60,255),
    }

    DIFFICULTY_COLORS={
        "EASY":(0,220,100),"NORMAL":(0,200,255),
        "HARD":(60,60,255),"EXTREME":(200,0,255),
    }

    # Avatar colors in BGR for OpenCV
    AVATAR_COLORS_BGR=[
        (255,215,0),(130,30,255),(140,255,0),(0,220,255),
        (255,0,200),(30,100,255),(255,140,100),(80,80,255),
    ]

    def __init__(self, model):
        self.model=model

    @staticmethod
    def display_frame(frame, window_name):
        cv2.imshow(window_name,frame); cv2.waitKey(10)

    # ── Esqueleto ──────────────────────────────────────────────────────────────

    @staticmethod
    def draw_skeleton(frame, key_points, color=(0,255,0), threshold=0.3, thickness=2):
        if key_points is None:
            return frame
        height,width,_=frame.shape
        shaped=np.squeeze(np.multiply(key_points,[height,width,1]))
        for kp in shaped:
            y,x,confidence=kp
            if confidence>threshold:
                cv2.circle(frame,(int(x),int(y)),thickness+2,color,-1)
        for s,e in JustDanceView.SKELETON_CONNECTIONS:
            y1,x1,c1=shaped[s]; y2,x2,c2=shaped[e]
            if c1>threshold and c2>threshold:
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,thickness)
        return frame

    @staticmethod
    def draw_game_hud(frame, score=0, combo=0, multiplier=1,
                      sync=0, time_left=0, song_name="Rehabilitación Física",
                      difficulty="NORMAL", max_score=12500,
                      player_name="", avatar_color_idx=0,
                      record_score=0,
                      practice_mode=False, practice_rep=0,
                      repetitions=5, current_rep=0, rep_results=None):
        """HUD clínico para sesión de rehabilitación."""
        h, w, _ = frame.shape
        import time

        BG = (40, 20, 12)          # BGR de (12,20,40)
        CARD = (65, 35, 20)        # BGR de (20,35,65)
        TEAL = (180, 180, 0)       # BGR de (0,180,180)
        GREEN = (120, 200, 40)     # BGR de (40,200,120)
        AMBER = (40, 160, 255)     # BGR de (255,160,40)
        TEXT = (250, 245, 235)
        MUTED = (185, 170, 150)
        BORDER = (95, 60, 35)

        rep_results = rep_results or []
        current_display = min(int(current_rep) + 1, int(repetitions)) if repetitions else int(current_rep) + 1

        # Header principal
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 76), BG, -1)
        cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)
        cv2.line(frame, (0, 76), (w, 76), TEAL, 2)

        # Marca y ejercicio
        cv2.putText(frame, "REHABILITACION FISICA", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEAL, 1, cv2.LINE_AA)
        safe_name = str(song_name or "Ejercicio")[:44]
        cv2.putText(frame, safe_name.upper(), (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)

        # Contador grande central
        rep_text = f"REPETICION {current_display} / {repetitions}"
        rep_sz = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_DUPLEX, 0.82, 2)[0]
        cx = w // 2
        cv2.putText(frame, rep_text, (cx - rep_sz[0] // 2, 32), cv2.FONT_HERSHEY_DUPLEX, 0.82, GREEN, 2, cv2.LINE_AA)

        # Progreso por repeticiones
        dot_y = 58
        spacing = 24
        total_w = max(1, repetitions) * spacing
        start_x = cx - total_w // 2 + spacing // 2
        for i in range(int(repetitions)):
            dx = start_x + i * spacing
            if i < len(rep_results):
                cv2.circle(frame, (dx, dot_y), 7, GREEN, -1, cv2.LINE_AA)
                cv2.line(frame, (dx - 3, dot_y), (dx - 1, dot_y + 3), TEXT, 1, cv2.LINE_AA)
                cv2.line(frame, (dx - 1, dot_y + 3), (dx + 4, dot_y - 4), TEXT, 1, cv2.LINE_AA)
            elif i == int(current_rep):
                pulse = int(8 + 2 * math.sin(time.time() * 5))
                cv2.circle(frame, (dx, dot_y), pulse, TEAL, 2, cv2.LINE_AA)
                cv2.circle(frame, (dx, dot_y), 3, TEAL, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (dx, dot_y), 6, BORDER, 1, cv2.LINE_AA)

        # Info derecha: nivel y tiempo
        level = f"NIVEL: {str(difficulty).upper()}"
        mins, secs = int(time_left) // 60, int(time_left) % 60
        cv2.putText(frame, level, (w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"TIEMPO {mins}:{secs:02d}", (w - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT, 1, cv2.LINE_AA)

        # Badge estado/sincronía clínico
        badge = f"CONTROL {int(sync)}%" if sync else "SEGUIMIENTO ACTIVO"
        bsz = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
        bx = w - 240
        by = 82
        cv2.rectangle(frame, (bx, by), (bx + bsz[0] + 22, by + 26), CARD, -1)
        cv2.rectangle(frame, (bx, by), (bx + bsz[0] + 22, by + 26), TEAL, 1)
        cv2.putText(frame, badge, (bx + 10, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, TEAL, 1, cv2.LINE_AA)

        # Contador de aciertos y fallas
        valid_count = sum(1 for r in rep_results if str(r.get("status", "")).upper() not in ("MISS", "OMITIDA", "SKIPPED"))
        invalid_count = sum(1 for r in rep_results if str(r.get("status", "")).upper() in ("MISS", "OMITIDA", "SKIPPED"))
        panel_x, panel_y = 16, 84
        panel_h = 22
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 200, panel_y + panel_h), CARD, -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 200, panel_y + panel_h), BORDER, 1)
        cv2.putText(frame, f"ACIERTOS: {valid_count}", (panel_x + 8, panel_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FALLAS: {invalid_count}", (panel_x + 110, panel_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, AMBER if invalid_count > 0 else MUTED, 1, cv2.LINE_AA)

        return frame

    @staticmethod

    def draw_combo(frame, combo):
        h,w,_=frame.shape
        overlay=frame.copy()
        bx,by,bw,bh=10,h-148,165,92
        cv2.rectangle(overlay,(bx,by),(bx+bw,by+bh),(8,4,18),-1)
        cv2.addWeighted(overlay,0.70,frame,0.30,0,frame)
        cv2.putText(frame,"COMBO",(bx+10,by+22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,90,130),1)
        combo_color=(0,215,255) if combo<10 else (0,255,80) if combo<25 else (255,215,0)
        cv2.putText(frame,f"x{combo}",(bx+10,by+80),cv2.FONT_HERSHEY_DUPLEX,1.8,combo_color,3)

    @staticmethod
    def draw_star_bar(frame, combo):
        h,w,_=frame.shape
        THRESHOLDS=[5,10,15,20,25]
        stars_filled=sum(1 for t in THRESHOLDS if combo>=t)
        STAR_R=14; GAP=36; total_w=len(THRESHOLDS)*GAP
        start_x=w//2-total_w//2+GAP//2; star_y=h-30
        for i in range(5):
            cx=start_x+i*GAP
            if i<stars_filled:
                _draw_star(frame,cx,star_y,STAR_R,(255,215,0),filled=True)
                _draw_star(frame,cx,star_y,STAR_R,(255,255,150),filled=False)
            else:
                _draw_star(frame,cx,star_y,STAR_R,(50,40,70),filled=True)
                _draw_star(frame,cx,star_y,STAR_R,(80,70,110),filled=False)

    # ── Ratings ────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_pose_rating(frame, rating, frames_left, total_frames):
        """Rating discreto, sin estética arcade."""
        if not rating or rating == "MISS":
            return frame
        h, w, _ = frame.shape
        alpha = max(0.0, min(1.0, frames_left / max(total_frames, 1)))
        msg_map = {
            "PERFECT": "Movimiento excelente",
            "GREAT": "Buen control",
            "GOOD": "Movimiento correcto",
            "OK": "Sigue con control",
        }
        msg = msg_map.get(str(rating).upper(), "Movimiento registrado")
        color = (120, 200, 40) if str(rating).upper() in ("PERFECT", "GREAT") else (40, 160, 255)
        overlay = frame.copy()
        box_w, box_h = 430, 54
        x1 = w // 2 - box_w // 2
        y1 = 92
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (65, 35, 20), -1)
        cv2.addWeighted(overlay, 0.55 * alpha, frame, 1 - 0.55 * alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)[0]
        cv2.putText(frame, msg, (w // 2 - sz[0] // 2, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.70, (250,245,235), 2, cv2.LINE_AA)
        return frame

    @staticmethod

    def draw_pregame_countdown(frame, value, progress=1.0):
        h,w,_=frame.shape
        overlay=frame.copy()
        cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
        text=str(value); is_go=(text in ("BAILA!", "INICIA"))
        color=(0,255,100) if is_go else (0,215,255)
        scale=4.0+(1.0-progress)*1.5
        if is_go: scale=3.0+(1.0-progress)*0.8
        font=cv2.FONT_HERSHEY_DUPLEX; thickness=int(8*max(0.3,progress))
        tsz=cv2.getTextSize(text,font,scale,thickness)[0]
        tx=(w-tsz[0])//2; ty=(h+tsz[1])//2
        alpha=max(0.0,progress)
        overlay2=frame.copy()
        cx2,cy2=w//2,h//2; radius=int(min(w,h)*0.22)
        ring_c=tuple(int(c*0.3) for c in color)
        cv2.circle(overlay2,(cx2,cy2),radius,ring_c,-1)
        cv2.circle(overlay2,(cx2,cy2),radius,color,4)
        cv2.circle(overlay2,(cx2,cy2),radius-10,color,1)
        cv2.putText(overlay2,text,(tx+6,ty+6),font,scale,(0,0,0),thickness+4)
        cv2.putText(overlay2,text,(tx,ty),font,scale,color,thickness)
        cv2.addWeighted(overlay2,alpha,frame,1-alpha,0,frame)
        return frame

    # ── Indicadores varios ──────────────────────────────────────────────────────

    @staticmethod
    def draw_score_overlay(frame,score,label="Score"):
        overlay=frame.copy()
        cv2.rectangle(overlay,(10,10),(220,70),(0,0,0),-1)
        cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
        cv2.putText(frame,f"{label}: {int(score)}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
        return frame

    @staticmethod
    def draw_match_indicator(frame,match_percentage):
        h,w,_=frame.shape
        bar_width=int((match_percentage/100)*200)
        cv2.rectangle(frame,(10,h-40),(210,h-15),(50,50,50),-1)
        color=(0,255,0) if match_percentage>70 else (0,255,255) if match_percentage>40 else (0,0,255)
        if bar_width>0:
            cv2.rectangle(frame,(10,h-40),(10+bar_width,h-15),color,-1)
        cv2.putText(frame,"Sync",(10,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        return frame

    @staticmethod
    def draw_accumulated_score(frame,total_points):
        h,w,_=frame.shape
        overlay=frame.copy()
        cv2.rectangle(overlay,(w-210,10),(w-10,72),(0,0,0),-1)
        cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
        cv2.putText(frame,f"PTS: {total_points:,}",(w-200,55),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,215,255),2)
        return frame

    @staticmethod
    def draw_ghost_silhouette(frame,key_points_reference,threshold=0.3):
        h,w,_=frame.shape
        shaped=np.squeeze(np.multiply(key_points_reference,[h,w,1]))
        ghost=(220,220,220); overlay=frame.copy()
        for kp in shaped:
            y,x,confidence=kp
            if confidence>threshold:
                cv2.circle(overlay,(int(x),int(y)),9,ghost,-1)
                cv2.circle(overlay,(int(x),int(y)),9,(150,150,150),2)
        for s,e in JustDanceView.SKELETON_CONNECTIONS:
            y1,x1,c1=shaped[s]; y2,x2,c2=shaped[e]
            if c1>threshold and c2>threshold:
                cv2.line(overlay,(int(x1),int(y1)),(int(x2),int(y2)),ghost,4)
        cv2.addWeighted(overlay,0.45,frame,0.55,0,frame)
        return frame

    @staticmethod
    def draw_countdown(frame,number):
        h,w,_=frame.shape
        font,fs,t=cv2.FONT_HERSHEY_DUPLEX,5.0,8; text=str(number); color=(0,215,255)
        tsz=cv2.getTextSize(text,font,fs,t)[0]
        tx=(w-tsz[0])//2; ty=h//2-60
        overlay=frame.copy()
        cx2,cy2=w//2,h//2-40
        cv2.circle(overlay,(cx2,cy2),80,(0,0,0),-1)
        cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
        cv2.putText(frame,text,(tx+4,ty+4),font,fs,(0,0,0),t+2)
        cv2.putText(frame,text,(tx,ty),font,fs,color,t)
        return frame

    @staticmethod
    def draw_body_warning(frame, subtitle=None):
        """Advertencia clinica de encuadre."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        amber = (40, 160, 255)
        bg = (65, 35, 20)
        text = (250, 245, 235)
        rect_h = 62
        y1 = h // 2 - rect_h // 2
        x1 = int(w * 0.52)
        x2 = w - 32
        cv2.rectangle(overlay, (x1, y1), (x2, y1 + rect_h), bg, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + rect_h), amber, 2)
        msg = "AJUSTA ENCUADRE"
        sub = subtitle or "Ajusta la camara para ver el movimiento"
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)[0]
        cv2.putText(frame, msg, (x1 + (x2 - x1 - sz[0]) // 2, y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.72, text, 2, cv2.LINE_AA)
        s2 = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(frame, sub, (x1 + (x2 - x1 - s2[0]) // 2, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185,170,150), 1, cv2.LINE_AA)
        return frame

    @staticmethod

    def draw_pose_preview(frame,pose_img,alpha=1.0,size=(160,160),margin=12):
        if pose_img is None or alpha<=0.0: return frame
        hf,wf,_=frame.shape
        x1=wf-size[0]-margin; y1=hf-size[1]-margin-50
        x2,y2=x1+size[0],y1+size[1]
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(wf,x2),min(hf,y2)
        aw,ah=x2-x1,y2-y1
        if aw<=0 or ah<=0: return frame
        pose_crop=cv2.resize(pose_img,(aw,ah)); overlay=frame.copy()
        pad=6
        cv2.rectangle(overlay,(x1-pad,y1-pad),(x2+pad,y2+pad),(0,0,0),-1)
        cv2.addWeighted(overlay,0.6*alpha,frame,1-0.6*alpha,0,frame)
        cv2.rectangle(frame,(x1-pad,y1-pad),(x2+pad,y2+pad),(0,215,255),2)
        roi=frame[y1:y2,x1:x2]
        cv2.addWeighted(pose_crop,alpha,roi,1-alpha,0,roi)
        frame[y1:y2,x1:x2]=roi
        return frame

    @staticmethod
    def draw_pose_conveyor(frame,upcoming_poses,video_time,eval_zone_x_ratio=0.20):
        if not upcoming_poses: return frame
        h,w,_=frame.shape
        STRIP_H=160; POSE_SIZE=140; STRIP_Y=h-STRIP_H-5
        EVAL_X=int(w*eval_zone_x_ratio); SECONDS_PER_PX=0.022
        BLUE_LIGHT=(255,140,30); floor_y=STRIP_Y+STRIP_H-15
        overlay=frame.copy()
        cv2.rectangle(overlay,(0,STRIP_Y),(w,h),(5,4,12),-1)
        cv2.addWeighted(overlay,0.30,frame,0.70,0,frame)
        cv2.line(frame,(0,floor_y),(w,floor_y),BLUE_LIGHT,2)
        cv2.line(frame,(EVAL_X,STRIP_Y-5),(EVAL_X,floor_y),BLUE_LIGHT,1)
        for beat_time,pose_data in upcoming_poses:
            if pose_data is None: continue
            time_diff=beat_time-video_time
            pose_x=EVAL_X+int(time_diff/SECONDS_PER_PX)
            is_active=abs(time_diff)<1.5
            img_size=POSE_SIZE if is_active else int(POSE_SIZE*0.72)
            half_img=img_size//2
            if pose_x+half_img<0 or pose_x-half_img>w: continue
            img_alpha=float(np.clip(1.0-abs(time_diff)*0.09,0.25,1.0))
            x1=max(0,pose_x-half_img); y1=max(0,floor_y-img_size)
            x2=min(w,pose_x+half_img); y2=min(h,floor_y)
            aw=x2-x1; ah=y2-y1
            if aw<4 or ah<4: continue
            try:
                if isinstance(pose_data,dict):
                    key="active" if is_active else "small"
                    pose_img,mask_f_orig=pose_data[key]
                elif isinstance(pose_data,tuple):
                    pose_img,mask_f_orig=pose_data
                else: continue
                pose_resized=cv2.resize(pose_img,(aw,ah))
                mask_resized=cv2.resize(mask_f_orig,(aw,ah))
                alpha_mask=(mask_resized*img_alpha)[...,np.newaxis]
                roi=frame[y1:y2,x1:x2].astype(np.float32)
                blended=pose_resized.astype(np.float32)*alpha_mask+roi*(1.0-alpha_mask)
                frame[y1:y2,x1:x2]=blended.astype(np.uint8)
            except Exception: continue
            if is_active:
                cv2.circle(frame,(pose_x,floor_y),6,BLUE_LIGHT,-1)
        return frame
