"""
just_dance_controller.py — v6 (sync fix + beat angles)
"""
from __future__ import annotations
from typing import Optional
import json, os, queue, threading, time, math
import cv2, numpy as np, pygame

from just_dance_view      import JustDanceView
from just_dance_telemetry import TelemetryMonitor
from pose_json_loader     import PoseJsonLoader
from just_dance_score     import load_profile, get_best_score, get_avatar_color
from duo_pose_utils       import DuoPoseTracker, PERSON_IDS, normalize_keypoints

# ── Capa clínica separada (rehab_core) ───────────────────────────────────────
try:
    from rehab_core.controller_bridge import RehabControllerBridge
    from rehab_core.angles import canonical_angle_name
    REHAB_BRIDGE_AVAILABLE = True
    from rehab_core.rep_error_detector import RepErrorDetector
except Exception as _rehab_bridge_error:
    print(f"[WARN] Rehab bridge no disponible: {_rehab_bridge_error}")
    RehabControllerBridge = None
    RepErrorDetector = None
    canonical_angle_name = lambda x: x
    REHAB_BRIDGE_AVAILABLE = False


class JustDanceController:

    POSE_EVAL_EVERY_SECONDS   = 0.5
    POSE_PREVIEW_SECONDS      = 1.5
    MIN_EVAL_SEPARATION_SECONDS = 0.25
    REQUIRED_KEYPOINTS        = [5,6,11,12,13,14,15,16]
    BODY_CONFIDENCE_THRESHOLD = 0.18
    MIN_BEAT_SEPARATION       = 0.5
    SILENT_EVAL_FRAMES        = 15
    _INFER_QUEUE_SIZE         = 2
    TRACKED_KEYPOINT_NAMES = {
        0: "Nariz", 1: "Ojo Izq.", 2: "Ojo Der.", 3: "Oreja Izq.",
        4: "Oreja Der.", 5: "Hombro Izq.", 6: "Hombro Der.",
        7: "Codo Izq.", 8: "Codo Der.", 9: "Muneca Izq.",
        10: "Muneca Der.", 11: "Cadera Izq.", 12: "Cadera Der.",
        13: "Rodilla Izq.", 14: "Rodilla Der.", 15: "Tobillo Izq.",
        16: "Tobillo Der.",
    }

    def __init__(self, model, video_path, camera_index=0, song_key=None, poses_data=None,
                 beats=None, char_info=None, screen_w=1280, screen_h=720,
                 difficulty="NORMAL", volume=0.8,
                 metrics=None, expected_players=1, repetitions=5):
        self.model        = model
        self.song_key     = song_key
        from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS as _rehab_cfgs
        self.exercise_name = (
            _rehab_cfgs.get(song_key or "", {}).get("name")
            or song_key
            or "Rehabilitacion Fisica"
        )
        self.video_path   = video_path
        self.camera_index = camera_index
        self.char_info    = char_info
        self.screen_w     = screen_w
        self.screen_h     = screen_h
        self.volume       = volume
        self.metrics      = metrics
        self.expected_players = max(1, int(expected_players or 1))
        self._duo_mode    = self.expected_players >= 2
        self._duo_tracker = DuoPoseTracker(carry_frames=12) if self._duo_mode else None
        self._duo_video_tracker = DuoPoseTracker(carry_frames=12) if self._duo_mode else None
        self._reference_person_id = self._get_reference_person_id(char_info)

        # Parámetros de rehabilitación clínica
        self.repetitions = repetitions
        self.current_rep = 0
        self.rep_results = []
        self.error_count = 0
        self._rep_error_detector = None
        self.error_attempts = []
        self._chrono_idx = 0

        from just_dance_rehab_config import REHAB_EXERCISE_CONFIGS
        self.rehab_cfg = REHAB_EXERCISE_CONFIGS.get(song_key, {})
        if self.rehab_cfg:
            self.rep_duration = self.rehab_cfg.get("rep_duration", 10.0)
            self.key_pose_offset = self.rehab_cfg.get("key_pose_offset", 4.0)
            if RepErrorDetector is not None:
                self._rep_error_detector = RepErrorDetector(self.rehab_cfg)
        else:
            self.rep_duration = 10.0
            self.key_pose_offset = 4.0

        # ── Bridge clínico separado ──────────────────────────────────────────
        self.rehab_bridge = None
        if REHAB_BRIDGE_AVAILABLE and self.rehab_cfg and self.rehab_cfg.get("tracking_type") not in ("pose33", "hand21"):
            try:
                self.rehab_bridge = RehabControllerBridge(
                    exercise_config=self.rehab_cfg,
                    exercise_id=song_key or "unknown",
                    target_repetitions=self.repetitions,
                )
                self.rehab_bridge.start()
                print(f"[INFO] Rehab bridge activo: {self.rehab_cfg.get('name', song_key)}")
            except Exception as e:
                print(f"[WARN] No se pudo iniciar RehabControllerBridge: {e}")
                self.rehab_bridge = None
        self.loop_start = 0.0
        self.loop_end = self.rep_duration
        self.key_pose_time = self.key_pose_offset
        self.rep_evaluated = False
        self.rep_best_similarity = 0.0
        self.rep_best_rating = "MISS"

        if poses_data is not None: self.poses_data = poses_data
        if beats      is not None: self.beats      = beats

        self.model.set_difficulty(difficulty)
        self.cap1 = cv2.VideoCapture(video_path)

        self.cap2 = None
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: self.cap2 = cap; break
            cap.release()
        if self.cap2 is None:
            self.cap2 = cv2.VideoCapture(camera_index)

        self.frame1_rate       = min(self.cap1.get(cv2.CAP_PROP_FPS) or 30, 30)
        self._video_actual_fps = self.cap1.get(cv2.CAP_PROP_FPS) or 30
        # Cámara optimizada para modo trajectory
        _is_trajectory_mode = (
            isinstance(getattr(self, "rehab_cfg", None), dict)
            and self.rehab_cfg.get("evaluation_mode") == "trajectory"
        )

        if _is_trajectory_mode:
            # Mejor calidad visual sin subir demasiado el costo.
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap2.set(cv2.CAP_PROP_FPS, 24)
            print("[INFO] Cámara trajectory: 640x480 @24fps")
        else:
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap2.set(cv2.CAP_PROP_FPS, 30)
        

        self._video_queue    = queue.Queue(maxsize=3)
        self._video_stop     = threading.Event()
        self._video_seek_req = threading.Event()
        self._video_seek_pos = 0
        threading.Thread(target=self._video_reader_thread, daemon=True).start()

        self._camera_queue = queue.Queue(maxsize=2)
        self._camera_stop  = threading.Event()
        threading.Thread(target=self._camera_reader_thread, daemon=True).start()

        # ── CORRECCIÓN: claves alineadas con KEYPOINT_DICT ────────────
        self.angles_video  = {j:[] for j in [
            "head",
            "left_arm", "right_arm", "left_elbow", "right_elbow",
            "left_trunk", "right_trunk",
            "left_knee", "right_knee", "left_ankle", "right_ankle"]}
        self.angles_camera = {j:[] for j in self.angles_video}
        self.angles_camera_duo = {
            pid: {j: [] for j in self.angles_video}
            for pid in PERSON_IDS
        }

        self.joint_stats = {"count": 0}
        for j in self.angles_video:
            self.joint_stats[j] = 0.0

        self.performance_stats = {}
        self._pose_raw_scores = []
        self._pose_adjusted_scores = []
        self._rating_counts = {r: 0 for r in ("PERFECT", "GREAT", "GOOD", "OK", "MISS")}
        self._tracking_frames = 0
        self._tracking_lost_counts = {i: 0 for i in range(17)}

        self.total_points     = 0
        self._song_start_time = None

        from pose_session_logger import PoseSessionLogger
        self._session_logger = PoseSessionLogger(
            song_key   = song_key  or "unknown",
            difficulty = difficulty,
            fps        = 30.0,
            condition  = model._actual_condition,
        )

        self._beat_angles = {}
        self.beats        = self._load_beats(song_key)
        # ──────────────────────────────────────────────────────────────

        self.using_beats = len(self.beats) > 0
        self.pose_images = self._load_pose_images(song_key)

        self._pose_loader = PoseJsonLoader(song_key)

        self._beat_pose_map   = self._build_beat_pose_map()
        self._processed_poses = {}
        self._poses_ready     = threading.Event()

        self._infer_input_q  = queue.Queue(maxsize=self._INFER_QUEUE_SIZE)
        self._infer_output_q = queue.Queue(maxsize=self._INFER_QUEUE_SIZE)
        self._infer_stop     = threading.Event()

        threading.Thread(target=self._preprocess_poses_thread, daemon=True).start()
        threading.Thread(target=self._inference_thread,        daemon=True).start()

        _profile              = load_profile() or {}
        self._player_name     = _profile.get("name","PACIENTE")
        self._avatar_color_idx= _profile.get("color_index",0)
        self._record_score    = get_best_score(song_key or "", difficulty)

    def _get_reference_person_id(self, char_info):
        if not isinstance(char_info, dict):
            return "left"
        info = char_info.get("char_info") or {}
        if info.get("id") in PERSON_IDS:
            return info["id"]
        x_start = float(info.get("x_start", 0.0) or 0.0)
        x_end = float(info.get("x_end", 1.0) or 1.0)
        return "right" if (x_start + x_end) / 2.0 >= 0.5 else "left"

    # ═══════════════════════════════════════════════════════════════════
    # HILOS BACKGROUND
    # ═══════════════════════════════════════════════════════════════════

    def _video_reader_thread(self):
        actual_fps     = self._video_actual_fps
        frame_interval = 1.0 / actual_fps
        frame_index    = 0

        while not self._video_stop.is_set():
            t0 = time.perf_counter()

            if self._video_seek_req.is_set():
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self._video_seek_pos)
                frame_index = int(self._video_seek_pos)
                while not self._video_queue.empty():
                    try: self._video_queue.get_nowait()
                    except queue.Empty: break
                self._video_seek_req.clear()

            ret, frame = self.cap1.read()
            if not ret:
                self._video_queue.put((False, None, 0.0))
                time.sleep(0.05)
                continue

            frame_time = frame_index / actual_fps
            frame_index += 1

            try:
                self._video_queue.put_nowait((True, frame, frame_time))
            except queue.Full:
                if self.metrics is not None:
                    self.metrics.record_video_drop()

            elapsed  = time.perf_counter() - t0
            to_sleep = frame_interval - elapsed
            if to_sleep > 0.001:
                time.sleep(to_sleep)

    def _camera_reader_thread(self):
        _interval = 1.0/30.0
        while not self._camera_stop.is_set():
            t0 = time.perf_counter()
            if self.cap2 and self.cap2.isOpened():
                ret, frame = self.cap2.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame,1)
                    if self._camera_queue.full():
                        if self.metrics is not None:
                            self.metrics.record_camera_drop()
                        try: self._camera_queue.get_nowait()
                        except queue.Empty: pass
                    self._camera_queue.put(frame)
            elapsed = time.perf_counter()-t0
            time.sleep(max(0.0,_interval-elapsed))

    def _preprocess_poses_thread(self):
        ACTIVE=140; SMALL=int(140*0.72)
        for beat_t,img in self._beat_pose_map.items():
            if img is None: self._processed_poses[beat_t]=None; continue
            result={}
            for key,size in (("active",ACTIVE),("small",SMALL)):
                resized=cv2.resize(img,(size,size))
                gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
                _,mask=cv2.threshold(gray,25,255,cv2.THRESH_BINARY)
                mask=cv2.GaussianBlur(mask,(3,3),0)
                mask_f=(mask/255.0).astype(np.float32)
                result[key]=(resized,mask_f)
            self._processed_poses[beat_t]=result
        count=sum(1 for v in self._processed_poses.values() if v is not None)
        print(f"[INFO] Poses pre-procesadas: {count} listas.")
        self._poses_ready.set()

    def _inference_thread(self):
        while not self._infer_stop.is_set():
            try:
                frame1_crop, frame2 = self._infer_input_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._duo_mode:
                camera_candidates = self._run_multi_inference_on(frame2, self.expected_players)
                kp_camera = self._duo_tracker.assign(camera_candidates)
                for pid in PERSON_IDS:
                    person_kp = kp_camera.get(pid, {}).get("keypoints")
                    if person_kp is not None:
                        self.model.store_angles(self.angles_camera_duo[pid], frame2, person_kp)

                if self._pose_loader.available and self._pose_loader.multi_person:
                    kp_video = None
                else:
                    video_candidates = self._run_multi_inference_on(frame1_crop, self.expected_players)
                    kp_video = self._duo_video_tracker.assign(video_candidates)
            else:
                kp_camera = self._run_inference_on(frame2)
                self.model.store_angles(self.angles_camera, frame2, kp_camera)

                # En modo trajectory, el video de referencia es solo guía visual.
                # No necesitamos inferir pose del video porque eso duplica el costo.
                _trajectory_mode = (
                    isinstance(getattr(self, "rehab_cfg", None), dict)
                    and self.rehab_cfg.get("evaluation_mode") == "trajectory"
                )

                if _trajectory_mode:
                    kp_video = None
                elif self._pose_loader.available:
                    kp_video = kp_camera
                else:
                    kp_video = self._run_inference_on(frame1_crop)
                    self.model.store_angles(self.angles_video, frame1_crop, kp_video)

            if self._infer_output_q.full():
                try: self._infer_output_q.get_nowait()
                except queue.Empty: pass
            self._infer_output_q.put((kp_video, kp_camera, frame1_crop, frame2))

    def _run_inference_on(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.metrics is not None:
            with self.metrics.inference_timer():
                result = self.model.run_inference(img)
        else:
            result = self.model.run_inference(img)
        return result

    def _run_multi_inference_on(self, frame, max_poses=2):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.metrics is not None:
            with self.metrics.inference_timer():
                result = self.model.run_multi_inference(img, max_poses=max_poses)
        else:
            result = self.model.run_multi_inference(img, max_poses=max_poses)
        return result


    # ═══════════════════════════════════════════════════════════════════
    # CARGA DE DATOS
    # ═══════════════════════════════════════════════════════════════════

    def _load_beats(self, song_key):
        if not song_key: return []
        path = os.path.join("songs_beats",f"{song_key}_beats.json")
        if not os.path.exists(path): return []
        try:
            with open(path,"r",encoding="utf-8") as f: data=json.load(f)
            raw_beats = data.get("beats", [])

            angles_loaded = 0
            for b in raw_beats:
                if "angles" in b and b["angles"]:
                    self._beat_angles[b["time"]] = b["angles"]
                    angles_loaded += 1

            raw  = sorted(set(b["time"] for b in raw_beats))
            filt, last = [], -999
            for t in raw:
                if t-last >= self.MIN_BEAT_SEPARATION: filt.append(t); last=t

            print(f"[INFO] Beats: {len(raw)} raw → {len(filt)} filtrados  "
                  f"| ángulos pre-calculados: {angles_loaded}")
            return filt
        except Exception as e:
            print(f"[WARNING] Error beats: {e}"); return []

    def _get_beat_angles(self, beat_time: float) -> Optional[dict]:
        if not self._beat_angles:
            return None
        if beat_time in self._beat_angles:
            return self._beat_angles[beat_time]
        best_t, best_d = None, float("inf")
        for t in self._beat_angles:
            d = abs(t - beat_time)
            if d < best_d:
                best_d, best_t = d, t
        if best_d <= 0.15:
            return self._beat_angles[best_t]
        return None

    def _parse_pose_time(self, filename):
        try:
            name=os.path.splitext(filename)[0]; parts=name.strip().split(" ")
            if len(parts)<2: return None
            m,s=parts[-1].split("-"); return float(int(m)*60+int(s))
        except Exception: return None

    def _load_pose_images(self, song_key):
        if not song_key: return []
        folder=os.path.join("songs_poses",song_key)
        if not os.path.exists(folder): return []
        images=[]
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".png"): continue
            t=self._parse_pose_time(fname)
            if t is None: continue
            img=cv2.imread(os.path.join(folder,fname),cv2.IMREAD_COLOR)
            if img is not None: images.append((t,img))
        images.sort(key=lambda x:x[0])
        return images

    def _build_beat_pose_map(self):
        return {bt:self._get_pose_image_for_beat(bt) for bt in self.beats}

    def _get_pose_image_for_beat(self, beat_time):
        if not self.pose_images: return None
        best,best_d=None,float("inf")
        for t,img in self.pose_images:
            d=abs(t-beat_time)
            if d<best_d and d<=3.0: best_d,best=d,img
        return best

    def _get_upcoming_poses(self, video_time, window=8.0):
        return [(bt,self._processed_poses.get(bt))
                for bt in self.beats if -1.0<=bt-video_time<=window]

    # ═══════════════════════════════════════════════════════════════════
    # AUDIO
    # ═══════════════════════════════════════════════════════════════════

    def play_sound(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            self._song_start_time = time.time()
            return
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(self.volume)
        self._song_start_time = time.time()

    def _seek_to(self, target_secs: float, audio_path: Optional[str] = None):
        self._video_seek_pos = int(max(0, target_secs) * self._video_actual_fps)
        self._video_seek_req.set()
        while self._video_seek_req.is_set():
            time.sleep(0.001)
        if audio_path and pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.set_pos(target_secs)
            except Exception:
                pass
        self._song_start_time = time.time() - target_secs

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS DE POSE
    # ═══════════════════════════════════════════════════════════════════

    def is_body_visible(self, key_points, frame):
        if key_points is None:
            return False
        h,w,_=frame.shape
        normalized = normalize_keypoints(key_points)
        if normalized is None:
            return False
        shaped=np.squeeze(np.multiply(normalized,[h,w,1]))
        required_visible = sum(
            1 for idx in self.REQUIRED_KEYPOINTS
            if shaped[idx][2] > self.BODY_CONFIDENCE_THRESHOLD
        )
        torso_visible = sum(
            1 for idx in (5, 6, 11, 12)
            if shaped[idx][2] > self.BODY_CONFIDENCE_THRESHOLD
        )
        leg_visible = sum(
            1 for idx in (13, 14, 15, 16)
            if shaped[idx][2] > self.BODY_CONFIDENCE_THRESHOLD
        )
        return required_visible >= 3 and (torso_visible >= 2 or leg_visible >= 2)

    def _crop_to_character(self, frame, char_info):
        if char_info is None: return frame
        h,w,_=frame.shape
        x1=max(0,int(char_info.get("x_start",0.0)*w))
        x2=min(w,int(char_info.get("x_end",1.0)*w))
        return frame[:,x1:x2] if x2>x1 else frame

    def _calculate_visible_angle(self, frame, key_points, start, middle, end, min_conf=0.12):
        if frame is None or key_points is None:
            return None
        kp = np.squeeze(np.array(key_points))
        if kp.ndim != 2 or kp.shape[0] <= max(start, middle, end):
            return None
        if min(kp[start][2], kp[middle][2], kp[end][2]) < min_conf:
            return None
        from just_dance_model import JustDanceModel
        return JustDanceModel.calculate_angle(frame, key_points, start, middle, end)

    def _get_head_angle(self, frame, key_points):
        for start, middle, end in ((3, 0, 4), (1, 0, 2), (5, 0, 6)):
            angle = self._calculate_visible_angle(frame, key_points, start, middle, end)
            if angle is not None:
                return angle
        return None

    def _with_head_angle(self, angles, frame, key_points):
        if not isinstance(angles, dict):
            return angles
        enriched = dict(angles)
        if "head" not in enriched:
            head_angle = self._get_head_angle(frame, key_points)
            if head_angle is not None:
                enriched["head"] = head_angle
        for name, triplet in {
            "left_trunk": (5, 11, 13),
            "right_trunk": (6, 12, 14),
        }.items():
            if name not in enriched:
                angle = self._calculate_visible_angle(frame, key_points, *triplet)
                if angle is not None:
                    enriched[name] = angle
        return enriched

    def _get_pose_angles(self, frame, key_points):
        """
        Calcula ángulos usando la capa separada rehab_core.

        Prioridad:
        1) RehabControllerBridge, si está disponible.
        2) Respaldo local con tripletas clínicas corregidas.
        """
        if getattr(self, "rehab_bridge", None) is not None:
            try:
                angles = self.rehab_bridge.calculate_angles_for_exercise(key_points)
                if angles:
                    return angles
            except Exception as e:
                print(f"[WARN] Error rehab_bridge.calculate_angles_for_exercise: {e}")

        triplets = {
            "left_elbow": (5, 7, 9),
            "right_elbow": (6, 8, 10),
            "left_shoulder": (7, 5, 11),
            "right_shoulder": (8, 6, 12),
            "left_trunk": (5, 11, 13),
            "right_trunk": (6, 12, 14),
            "left_hip": (5, 11, 13),
            "right_hip": (6, 12, 14),
            "left_knee": (11, 13, 15),
            "right_knee": (12, 14, 16),
            # Compatibilidad antigua
            "left_arm": (5, 7, 9),
            "right_arm": (6, 8, 10),
        }

        active_joints = self.rehab_cfg.get("active_joints") if self.rehab_cfg else None
        active_angles = self.rehab_cfg.get("active_angles") if self.rehab_cfg else None
        active = active_angles or active_joints

        if active:
            wanted = set()
            for name in active:
                wanted.add(name)
                wanted.add(canonical_angle_name(name))

            triplets = {
                k: v
                for k, v in triplets.items()
                if k in wanted or canonical_angle_name(k) in wanted
            }

        angles = {}
        for j, (s, m, e) in triplets.items():
            angle = self._calculate_visible_angle(frame, key_points, s, m, e)
            if angle is not None:
                angles[j] = angle

        head_angle = self._get_head_angle(frame, key_points)
        if head_angle is not None:
            angles["head"] = head_angle

        return angles


    def _compare_pose_angles(self, angles_target, angles_user):
        """
        Compara ángulos de referencia vs usuario.

        Mantiene el score anterior, pero filtra por articulaciones activas
        y normaliza nombres con canonical_angle_name.
        """
        if not isinstance(angles_target, dict) or not isinstance(angles_user, dict):
            return 0

        threshold = self.model.angle_threshold
        soft_limit = threshold * 1.5

        active_joints = self.rehab_cfg.get("active_joints") if self.rehab_cfg else None
        active_angles = self.rehab_cfg.get("active_angles") if self.rehab_cfg else None
        active = active_angles or active_joints

        if active:
            wanted = set()
            for name in active:
                wanted.add(name)
                wanted.add(canonical_angle_name(name))

            angles_target = {
                canonical_angle_name(k): v
                for k, v in angles_target.items()
                if k in wanted or canonical_angle_name(k) in wanted
            }
            angles_user = {
                canonical_angle_name(k): v
                for k, v in angles_user.items()
                if k in wanted or canonical_angle_name(k) in wanted
            }
        else:
            angles_target = {canonical_angle_name(k): v for k, v in angles_target.items()}
            angles_user = {canonical_angle_name(k): v for k, v in angles_user.items()}

        weights = {
            "head": 0.7,
            "left_shoulder": 2.0, "right_shoulder": 2.0,
            "left_elbow": 1.5, "right_elbow": 1.5,
            "left_trunk": 1.6, "right_trunk": 1.6,
            "left_hip": 1.3, "right_hip": 1.3,
            "left_knee": 1.4, "right_knee": 1.4,
            "left_arm": 1.5, "right_arm": 1.5,
        }

        ws = 0.0
        tw = 0.0
        for joint in angles_target:
            if joint in angles_user:
                w = weights.get(joint, 1.0)
                diff = abs(float(angles_target[joint]) - float(angles_user[joint]))
                ws += max(0.0, 1.0 - (diff / soft_limit) ** 2) * w
                tw += w

        return int((ws / tw) * 100) if tw >= 1.0 else 0


    def _record_tracking_quality(self, key_points):
        kp = np.squeeze(np.array(key_points))
        if kp.ndim != 2 or kp.shape[0] < 17:
            return
        self._tracking_frames += 1
        for idx in range(17):
            if kp[idx][2] < self.BODY_CONFIDENCE_THRESHOLD:
                self._tracking_lost_counts[idx] += 1

    def _record_pose_evaluation(self, raw_score, adjusted_score, rating):
        self._pose_raw_scores.append(int(np.clip(raw_score, 0, 100)))
        self._pose_adjusted_scores.append(int(np.clip(adjusted_score, 0, 100)))
        self._rating_counts[rating] = self._rating_counts.get(rating, 0) + 1

    def _build_performance_stats(self):
        adjusted  = self._pose_adjusted_scores
        raw       = self._pose_raw_scores
        eval_count = len(adjusted)

        if self._tracking_frames > 0:
            total_slots  = self._tracking_frames * 17
            lost_total   = sum(self._tracking_lost_counts.values())
            keypoint_loss_pct = lost_total / max(total_slots, 1) * 100.0
            worst_keypoints = sorted(
                (
                    {
                        "name": self.TRACKED_KEYPOINT_NAMES.get(idx, str(idx)),
                        "loss_pct": round(count / self._tracking_frames * 100.0, 1),
                    }
                    for idx, count in self._tracking_lost_counts.items()
                ),
                key=lambda item: item["loss_pct"],
                reverse=True,
            )[:3]
        else:
            keypoint_loss_pct = 0.0
            worst_keypoints   = []

        joint_count  = self.joint_stats.get("count", 0)
        weakest_joint = None
        if joint_count > 0:
            # ── CORRECCIÓN: claves y etiquetas alineadas con KEYPOINT_DICT ──
            joint_labels = {
                "head":         "Cabeza",
                "left_arm":     "Brazo Izq.",   "right_arm":    "Brazo Der.",
                "left_elbow":   "Codo Izq.",    "right_elbow":  "Codo Der.",
                "left_knee":    "Rodilla Izq.", "right_knee":   "Rodilla Der.",
                "left_ankle":   "Pie Izq.",     "right_ankle":  "Pie Der.",
            }
            # ──────────────────────────────────────────────────────────────
            joint_scores = [
                (joint_labels[j], self.joint_stats.get(j, 0.0) / joint_count * 100.0)
                for j in joint_labels
            ]
            weakest_joint = min(joint_scores, key=lambda item: item[1])

        avg_similarity    = float(np.mean(adjusted)) if adjusted else 0.0
        tracking_quality  = max(0.0, 100.0 - keypoint_loss_pct)

        if keypoint_loss_pct >= 35.0:
            advice = "Mejora iluminacion y encuadre para que la camara no pierda el cuerpo."
        elif weakest_joint and weakest_joint[1] < 55.0:
            advice = f"Enfocate en {weakest_joint[0]}: fue la articulacion mas baja."
        elif avg_similarity >= 80.0:
            advice = "Gran control: mantuviste poses muy cercanas al objetivo."
        elif eval_count > 0:
            advice = "Buen intento: busca marcar mas los brazos en cada pose clave."
        else:
            advice = "No hubo evaluaciones suficientes para dar un consejo."

        perfect_pct = (
            self._rating_counts.get("PERFECT", 0) / eval_count * 100.0
            if eval_count else 0.0
        )

        return {
            "evaluation_count":    eval_count,
            "avg_similarity":      round(avg_similarity, 1),
            "avg_raw_similarity":  round(float(np.mean(raw)) if raw else 0.0, 1),
            "best_similarity":     int(max(adjusted)) if adjusted else 0,
            "worst_similarity":    int(min(adjusted)) if adjusted else 0,
            "perfect_pct":         round(perfect_pct, 1),
            "rating_counts":       dict(self._rating_counts),
            "tracking_frames":     self._tracking_frames,
            "tracking_quality_pct":round(tracking_quality, 1),
            "keypoint_loss_pct":   round(keypoint_loss_pct, 1),
            "worst_keypoints":     worst_keypoints,
            "advice":              advice,
        }

    def _is_song_finished(self, video_time, song_length):
        return video_time >= song_length - 0.1

    # ═══════════════════════════════════════════════════════════════════
    # UI HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _draw_pause_menu(self, frame, practice_mode=False):
        h,w,_=frame.shape
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
        pw,ph=720,485; px,py=(w-pw)//2,(h-ph)//2
        cv2.rectangle(frame,(px+8,py+8),(px+pw+8,py+ph+8),(0,0,0),-1)
        cv2.rectangle(frame,(px,py),(px+pw,py+ph),(12,12,28),-1)
        cv2.rectangle(frame,(px,py),(px+pw,py+ph),(0,220,255),3)
        cv2.line(frame,(px+5,py+5),(px+pw-5,py+5),(255,255,255),1)
        cv2.putText(frame,"PAUSED",(px+205,py+82),cv2.FONT_HERSHEY_DUPLEX,1.9,(255,255,255),4)
        cv2.putText(frame,"SESSION INTERRUPTED",(px+210,py+112),cv2.FONT_HERSHEY_SIMPLEX,0.55,(140,140,140),1)
        dc={"EASY":(0,255,140),"NORMAL":(255,220,0),"HARD":(60,60,255)}.get(self.model.difficulty,(200,200,200))
        cv2.putText(frame,f"DIFFICULTY: {self.model.difficulty}",(px+240,py+148),cv2.FONT_HERSHEY_SIMPLEX,0.6,dc,2)
        pm_label="[ M ]  SALIR PRACTICA" if practice_mode else "[ M ]  MODO PRACTICA"
        pm_color=(0,100,255) if practice_mode else (200,180,255)
        opts=[
            ("[ C ]  CONTINUAR",       (0,255,140)),
            ("[ R ]  REINICIAR",       (255,220,0)),
            ("[ Q ]  VOLVER AL INICIO",(0,180,255)),
            (pm_label,                  pm_color),
            ("[ ESC ] REANUDAR",       (255,255,255)),
        ]
        y=py+205
        for text,color in opts:
            cv2.rectangle(frame,(px+70,y-28),(px+pw-70,y+14),(20,20,35),-1)
            cv2.putText(frame,text,(px+95,y),cv2.FONT_HERSHEY_SIMPLEX,0.85,color,2)
            y+=50
        cv2.line(frame,(px+40,py+ph-55),(px+pw-40,py+ph-55),(45,45,65),1)
        cv2.putText(frame,"Sistema de Evaluacion de Movimiento",(px+180,py+ph-22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(110,110,110),1)
        return frame

    def _draw_resume_countdown(self, frame, number):
        h,w,_=frame.shape
        ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(ov,0.4,frame,0.6,0,frame)
        text=str(number); font,fs,t=cv2.FONT_HERSHEY_DUPLEX,8.0,12; color=(0,215,255)
        tsz=cv2.getTextSize(text,font,fs,t)[0]
        tx=(w-tsz[0])//2; ty=(h+tsz[1])//2
        cv2.putText(frame,text,(tx+5,ty+5),font,fs,(0,0,0),t+4)
        cv2.putText(frame,text,(tx,ty),font,fs,color,t)
        return frame

    def _draw_silent_points(self, frame, points):
        if points<=0: return frame
        h,w,_=frame.shape
        cv2.putText(frame,f"+{points}",(w-80,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,255,180),2)
        return frame

    def _draw_beat_indicator(self, frame, beat_index, total_beats):
        if not self.using_beats: return frame
        h,w,_=frame.shape
        cv2.putText(frame,f"Beat {beat_index}/{total_beats}",(w-180,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,215,255),2)
        return frame

    # ═══════════════════════════════════════════════════════════════════
    # COMPOSICIÓN
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _square_letterbox_frame(frame, target_w, target_h):
        fh,fw=frame.shape[:2]; side=max(fh,fw)
        square_frame=np.zeros((side,side,3),dtype=np.uint8)
        c_yo=(side-fh)//2; c_xo=(side-fw)//2
        square_frame[c_yo:c_yo+fh,c_xo:c_xo+fw]=frame
        scale=min(target_w/side,target_h/side)*0.95
        new_side=int(side*scale)
        resized=cv2.resize(square_frame,(new_side,new_side),interpolation=cv2.INTER_LINEAR)
        canvas=np.zeros((target_h,target_w,3),dtype=np.uint8)
        xo=(target_w-new_side)//2; yo=(target_h-new_side)//2
        canvas[yo:yo+new_side,xo:xo+new_side]=resized
        return canvas,xo,yo,new_side,new_side

    def _compose_letterbox(self, frame_video, frame_cam, out_w, out_h):
        half_w=out_w//2; half_h=out_h
        DIFF_COLORS={"EASY":(0,220,120),"NORMAL":(255,220,0),"HARD":(60,60,255),"EXTREME":(200,0,255)}
        accent=DIFF_COLORS.get(self.model.difficulty,(0,215,255))
        left_canvas,lx,ly,lw,lh=JustDanceController._square_letterbox_frame(frame_video,half_w,half_h)
        right_canvas,rx,ry,rw,rh=JustDanceController._square_letterbox_frame(frame_cam,half_w,half_h)
        cv2.rectangle(left_canvas,(lx,ly),(lx+lw-1,ly+lh-1),(40,30,60),1)
        cv2.putText(left_canvas,"REFERENCIA",(lx+6,ly+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,70,110),1)
        cv2.rectangle(right_canvas,(rx,ry),(rx+rw-1,ry+rh-1),accent,2)
        cv2.putText(right_canvas,"TU",(rx+6,ry+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,accent,1)
        combined=np.concatenate((left_canvas,right_canvas),axis=1)
        if combined.shape[1]!=out_w or combined.shape[0]!=out_h:
            combined=cv2.resize(combined,(out_w,out_h),interpolation=cv2.INTER_NEAREST)
        cx2=out_w//2
        cv2.line(combined,(cx2-1,0),(cx2-1,out_h),(20,15,40),3)
        cv2.line(combined,(cx2,0),(cx2,out_h),accent,1)
        cv2.line(combined,(cx2+1,0),(cx2+1,out_h),(20,15,40),3)
        return combined

    # ═══════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════

    def _duo_entry(self, persons, person_id):
        if not isinstance(persons, dict):
            return {}
        return persons.get(person_id) or {}

    def _duo_keypoints(self, persons, person_id):
        entry = self._duo_entry(persons, person_id)
        return normalize_keypoints(entry.get("keypoints"))

    def _duo_angles(self, persons, person_id, frame):
        entry = self._duo_entry(persons, person_id)
        angles = entry.get("angles")
        if isinstance(angles, dict) and angles:
            return angles
        keypoints = self._duo_keypoints(persons, person_id)
        if keypoints is None:
            return None
        return self._get_pose_angles(frame, keypoints)

    def _draw_duo_skeletons(self, frame, persons, colors):
        for pid in PERSON_IDS:
            keypoints = self._duo_keypoints(persons, pid)
            if keypoints is not None:
                frame = JustDanceView.draw_skeleton(frame, keypoints, color=colors[pid])
        return frame

    def _draw_duo_scoreboard(self, frame, team_score):
        text = f"EQUIPO: {int(team_score)}%"
        cv2.putText(frame, text, (18, 84), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 0), 4)
        cv2.putText(frame, text, (18, 84), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 215, 255), 2)
        return frame

    def _score_single_pairing(self, target_pid, camera_pid, reference_persons, camera_persons, frame_video, frame_camera):
        from just_dance_model import JustDanceModel

        target_kp = self._duo_keypoints(reference_persons, target_pid)
        user_kp = self._duo_keypoints(camera_persons, camera_pid)
        target_visible = target_kp is not None and self.is_body_visible(target_kp, frame_video)
        user_visible = user_kp is not None and self.is_body_visible(user_kp, frame_camera)

        if not target_visible or not user_visible:
            return 0, 0

        target_angles = self._duo_angles(reference_persons, target_pid, frame_video)
        user_angles = self._get_pose_angles(frame_camera, user_kp)
        if target_angles is None:
            return 0, 0

        target_angles = self._with_head_angle(target_angles, frame_video, target_kp)
        raw_score = self._compare_pose_angles(target_angles, user_angles)
        adjusted = JustDanceModel.adjusted_pose_score(
            raw_score=raw_score,
            pose_angles_target=target_angles,
            angles_camera=self.angles_camera_duo[camera_pid],
            frame_video=frame_video,
            kp_video=target_kp,
            frame_camera=frame_camera,
            kp_camera=user_kp,
            movement_threshold=self.model.movement_threshold,
            distinctiveness_threshold=self.model.distinctiveness_threshold,
            angle_weight=self.model.angle_weight,
            vector_weight=self.model.vector_weight,
        )
        return adjusted, raw_score

    def _score_duo_pose(self, reference_persons, camera_persons, frame_video, frame_camera):
        # Opción A: Directa (left->left, right->right)
        adj_left_A, raw_left_A = self._score_single_pairing("left", "left", reference_persons, camera_persons, frame_video, frame_camera)
        adj_right_A, raw_right_A = self._score_single_pairing("right", "right", reference_persons, camera_persons, frame_video, frame_camera)
        score_A = adj_left_A + adj_right_A

        # Opción B: Cruzada (left->right, right->left)
        adj_left_B, raw_left_B = self._score_single_pairing("right", "left", reference_persons, camera_persons, frame_video, frame_camera)
        adj_right_B, raw_right_B = self._score_single_pairing("left", "right", reference_persons, camera_persons, frame_video, frame_camera)
        score_B = adj_left_B + adj_right_B

        return frame

    def _draw_resume_countdown(self, frame, number):
        h,w,_=frame.shape
        ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(ov,0.4,frame,0.6,0,frame)
        text=str(number); font,fs,t=cv2.FONT_HERSHEY_DUPLEX,8.0,12; color=(0,215,255)
        tsz=cv2.getTextSize(text,font,fs,t)[0]
        tx=(w-tsz[0])//2; ty=(h+tsz[1])//2
        cv2.putText(frame,text,(tx+5,ty+5),font,fs,(0,0,0),t+4)
        cv2.putText(frame,text,(tx,ty),font,fs,color,t)
        return frame

    def _draw_silent_points(self, frame, points):
        if points<=0: return frame
        h,w,_=frame.shape
        cv2.putText(frame,f"+{points}",(w-80,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,255,180),2)
        return frame

    def _draw_beat_indicator(self, frame, beat_index, total_beats):
        if not self.using_beats: return frame
        h,w,_=frame.shape
        cv2.putText(frame,f"Beat {beat_index}/{total_beats}",(w-180,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,215,255),2)
        return frame

    # ═══════════════════════════════════════════════════════════════════
    # COMPOSICIÓN
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _square_letterbox_frame(frame, target_w, target_h):
        fh,fw=frame.shape[:2]; side=max(fh,fw)
        square_frame=np.zeros((side,side,3),dtype=np.uint8)
        c_yo=(side-fh)//2; c_xo=(side-fw)//2
        square_frame[c_yo:c_yo+fh,c_xo:c_xo+fw]=frame
        scale=min(target_w/side,target_h/side)*0.95
        new_side=int(side*scale)
        resized=cv2.resize(square_frame,(new_side,new_side),interpolation=cv2.INTER_LINEAR)
        canvas=np.zeros((target_h,target_w,3),dtype=np.uint8)
        xo=(target_w-new_side)//2; yo=(target_h-new_side)//2
        canvas[yo:yo+new_side,xo:xo+new_side]=resized
        return canvas,xo,yo,new_side,new_side

    def _compose_letterbox(self, frame_video, frame_cam, out_w, out_h):
        half_w=out_w//2; half_h=out_h
        DIFF_COLORS={"EASY":(0,220,120),"NORMAL":(255,220,0),"HARD":(60,60,255),"EXTREME":(200,0,255)}
        accent=DIFF_COLORS.get(self.model.difficulty,(0,215,255))
        left_canvas,lx,ly,lw,lh=JustDanceController._square_letterbox_frame(frame_video,half_w,half_h)
        right_canvas,rx,ry,rw,rh=JustDanceController._square_letterbox_frame(frame_cam,half_w,half_h)
        cv2.rectangle(left_canvas,(lx,ly),(lx+lw-1,ly+lh-1),(40,30,60),1)
        cv2.putText(left_canvas,"REFERENCIA",(lx+6,ly+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,70,110),1)
        cv2.rectangle(right_canvas,(rx,ry),(rx+rw-1,ry+rh-1),accent,2)
        cv2.putText(right_canvas,"TU",(rx+6,ry+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,accent,1)
        combined=np.concatenate((left_canvas,right_canvas),axis=1)
        if combined.shape[1]!=out_w or combined.shape[0]!=out_h:
            combined=cv2.resize(combined,(out_w,out_h),interpolation=cv2.INTER_NEAREST)
        cx2=out_w//2
        cv2.line(combined,(cx2-1,0),(cx2-1,out_h),(20,15,40),3)
        cv2.line(combined,(cx2,0),(cx2,out_h),accent,1)
        cv2.line(combined,(cx2+1,0),(cx2+1,out_h),(20,15,40),3)
        return combined

    # ═══════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════

    def _duo_entry(self, persons, person_id):
        if not isinstance(persons, dict):
            return {}
        return persons.get(person_id) or {}

    def _duo_keypoints(self, persons, person_id):
        entry = self._duo_entry(persons, person_id)
        return normalize_keypoints(entry.get("keypoints"))

    def _duo_angles(self, persons, person_id, frame):
        entry = self._duo_entry(persons, person_id)
        angles = entry.get("angles")
        if isinstance(angles, dict) and angles:
            return angles
        keypoints = self._duo_keypoints(persons, person_id)
        if keypoints is None:
            return None
        return self._get_pose_angles(frame, keypoints)

    def _draw_duo_skeletons(self, frame, persons, colors):
        for pid in PERSON_IDS:
            keypoints = self._duo_keypoints(persons, pid)
            if keypoints is not None:
                frame = JustDanceView.draw_skeleton(frame, keypoints, color=colors[pid])
        return frame

    def _draw_duo_scoreboard(self, frame, team_score):
        text = f"EQUIPO: {int(team_score)}%"
        cv2.putText(frame, text, (18, 84), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 0), 4)
        cv2.putText(frame, text, (18, 84), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 215, 255), 2)
        return frame

    def _score_single_pairing(self, target_pid, camera_pid, reference_persons, camera_persons, frame_video, frame_camera):
        from just_dance_model import JustDanceModel

        target_kp = self._duo_keypoints(reference_persons, target_pid)
        user_kp   = self._duo_keypoints(camera_persons, camera_pid)
        target_visible = target_kp is not None and self.is_body_visible(target_kp, frame_video)
        user_visible   = user_kp   is not None and self.is_body_visible(user_kp,   frame_camera)

        if not target_visible or not user_visible:
            return 0, 0

        target_angles = self._duo_angles(reference_persons, target_pid, frame_video)
        user_angles   = self._get_pose_angles(frame_camera, user_kp)
        if target_angles is None:
            return 0, 0

        target_angles = self._with_head_angle(target_angles, frame_video, target_kp)
        raw_score = self._compare_pose_angles(target_angles, user_angles)

        active_joints = self.rehab_cfg.get("active_joints") if self.rehab_cfg else None
        _target_filtered = {k: v for k, v in target_angles.items()
                            if not active_joints or k in active_joints}
        _camera_filtered = {k: v for k, v in self.angles_camera_duo[camera_pid].items()
                            if not active_joints or k in active_joints}

        adjusted = JustDanceModel.adjusted_pose_score(
            raw_score=raw_score,
            pose_angles_target=_target_filtered,
            angles_camera=_camera_filtered,
            frame_video=frame_video,
            kp_video=target_kp,
            frame_camera=frame_camera,
            kp_camera=user_kp,
            movement_threshold=self.model.movement_threshold,
            distinctiveness_threshold=self.model.distinctiveness_threshold,
            angle_weight=self.model.angle_weight,
            vector_weight=self.model.vector_weight,
        )
        return adjusted, raw_score

    def _score_duo_pose(self, reference_persons, camera_persons, frame_video, frame_camera):
        # Opción A: Directa (left->left, right->right)
        adj_left_A, raw_left_A = self._score_single_pairing("left", "left", reference_persons, camera_persons, frame_video, frame_camera)
        adj_right_A, raw_right_A = self._score_single_pairing("right", "right", reference_persons, camera_persons, frame_video, frame_camera)
        score_A = adj_left_A + adj_right_A

        # Opción B: Cruzada (left->right, right->left)
        adj_left_B, raw_left_B = self._score_single_pairing("right", "left", reference_persons, camera_persons, frame_video, frame_camera)
        adj_right_B, raw_right_B = self._score_single_pairing("left", "right", reference_persons, camera_persons, frame_video, frame_camera)
        score_B = adj_left_B + adj_right_B

        # Seleccionar la mejor opción
        if score_B > score_A:
            adjusted_scores = [adj_left_B, adj_right_B]
            raw_scores = [raw_left_B, raw_right_B]
            person_scores = {"left": adj_left_B, "right": adj_right_B}
            best_target_mapping = {"left": "right", "right": "left"}
        else:
            adjusted_scores = [adj_left_A, adj_right_A]
            raw_scores = [raw_left_A, raw_right_A]
            person_scores = {"left": adj_left_A, "right": adj_right_A}
            best_target_mapping = {"left": "left", "right": "right"}

        # Actualizar joint_stats con el promedio de ambos pacientes
        if "count" in self.joint_stats:
            self.joint_stats["count"] += 1
            soft_limit = self.model.angle_threshold * 1.5
            
            # Para cada articulación, promediamos la precisión de P1 y P2
            for j_key in self.angles_video:
                precision_sum = 0.0
                pair_count = 0
                for camera_pid in PERSON_IDS:
                    target_pid = best_target_mapping[camera_pid]
                    target_kp = self._duo_keypoints(reference_persons, target_pid)
                    user_kp = self._duo_keypoints(camera_persons, camera_pid)
                    target_visible = target_kp is not None and self.is_body_visible(target_kp, frame_video)
                    user_visible = user_kp is not None and self.is_body_visible(user_kp, frame_camera)
                    
                    if target_visible and user_visible:
                        target_angles = self._duo_angles(reference_persons, target_pid, frame_video)
                        user_angles = self._get_pose_angles(frame_camera, user_kp)
                        if target_angles and j_key in target_angles and j_key in user_angles:
                            diff = abs(target_angles[j_key] - user_angles[j_key])
                            precision_sum += max(0.0, 1.0 - (diff / soft_limit) ** 2)
                            pair_count += 1
                
                if pair_count > 0:
                    self.joint_stats[j_key] += (precision_sum / pair_count)

        visible_users = 0
        for pid in PERSON_IDS:
            user_kp = self._duo_keypoints(camera_persons, pid)
            if user_kp is not None and self.is_body_visible(user_kp, frame_camera):
                visible_users += 1

        team_score = int(np.mean(adjusted_scores)) if adjusted_scores else 0
        raw_team = int(np.mean(raw_scores)) if raw_scores else 0
        return team_score, raw_team, person_scores, visible_users

    def _process_frames_duo(self, audio_path=None):
        from just_dance_model import JustDanceModel

        total_frames = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap1.get(cv2.CAP_PROP_FPS) or 30
        song_length = total_frames / fps if total_frames > 0 else 999
        audio_enabled = bool(audio_path and os.path.exists(audio_path))
        if audio_enabled:
            try:
                from mutagen.mp3 import MP3
                song_length = MP3(audio_path).info.length
            except Exception:
                pass

        max_score = self.model.max_score
        print(f"[INFO] Modo DUO Clínico: {self.repetitions} repeticiones.")

        game_clock = pygame.time.Clock()
        telemetry = TelemetryMonitor()
        screen = pygame.display.get_surface()
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.NOFRAME)

        if self.metrics is not None:
            self.metrics.start()

        frame_counter = 0
        last_kp_video = None
        last_kp_camera = None
        last_frame1 = None
        last_frame_video_time = 0.0
        last_frame1_crop = None
        _last_cam_frame = None
        video_ended = False
        _audio_started = False
        _audio_started_time = 0.0
        current_score = 0
        current_rating = None
        rating_frames_left = 0
        warning_frames = 0
        combo = best_combo = 0
        multiplier = 1
        person_scores = {"left": 0, "right": 0}
        self.total_points = 0
        SEND_EVERY = 8
        RATING_DISPLAY_FRAMES = 80
        colors_ref = {"left": (0, 255, 0), "right": (255, 0, 220)}
        colors_cam = {"left": (0, 215, 255), "right": (255, 80, 180)}
        
        force_exit_reps = False
        _fx_particles = []

        # Inicializar repetición clínica
        self.current_rep = 0
        self.rep_results = []
        self.error_count = 0
        self.loop_start = 0.0
        self.loop_end = self.rep_duration
        self.key_pose_time = self.loop_start + self.key_pose_offset
        self.rep_evaluated = False
        self.rep_best_similarity = 0.0
        self.rep_best_rating = "MISS"

        while self.cap1.isOpened() and not force_exit_reps:
            game_clock.tick(30)
            telemetry.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._cleanup()
                    return "lobby"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self._cleanup()
                        return "lobby"
                    elif event.key == pygame.K_SPACE:
                        # Saltar repetición manualmente
                        self._play_success_chime()
                        self._add_rep_result({
                            "rep_idx": self.current_rep,
                            "status": "OMITIDA",
                            "similarity": float(self.rep_best_similarity)
                        })
                        self._record_pose_evaluation(0, int(self.rep_best_similarity), "MISS")
                        self.current_rep += 1
                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True
                        else:
                            self.loop_start = self.current_rep * self.rep_duration
                            self.loop_end = (self.current_rep + 1) * self.rep_duration
                            self.key_pose_time = self.loop_start + self.key_pose_offset
                            self.rep_evaluated = False
                            self.rep_best_similarity = 0.0
                            self.rep_best_rating = "MISS"
                            self._seek_to(self.loop_start, audio_path)

            if force_exit_reps:
                break

            _frame_ctx = self.metrics.frame_timer() if self.metrics else _NullCtx()
            with _frame_ctx:
                if not video_ended:
                    try:
                        ret1, frame1, frame_video_time = self._video_queue.get(timeout=0.033)
                        if not ret1 or frame1 is None or frame1.size == 0:
                            video_ended = True
                            frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                            frame_video_time = last_frame_video_time
                        else:
                            last_frame1 = frame1.copy()
                            last_frame_video_time = frame_video_time
                    except queue.Empty:
                        frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                        frame_video_time = last_frame_video_time
                else:
                    frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    frame_video_time = last_frame_video_time

                try:
                    _last_cam_frame = self._camera_queue.get_nowait()
                except queue.Empty:
                    pass
                frame2 = _last_cam_frame if _last_cam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                video_time = frame_video_time if _audio_started else 0.0

                # Detección de fin de video/ejercicio
                if video_ended and not audio_enabled:
                    break
                if audio_enabled and _audio_started and not pygame.mixer.music.get_busy() and (time.time() - _audio_started_time > 2.0):
                    break

                # Segment looping si no se completó la repetición
                if video_time >= self.loop_end or video_ended:
                    if not self.rep_evaluated:
                        self.rep_best_similarity = 0.0
                        self.rep_best_rating = "MISS"
                        self._seek_to(self.loop_start, audio_path)
                        video_ended = False
                        continue

                frame_counter += 1
                frame1_crop = frame1
                if not video_ended and frame_counter % SEND_EVERY == 0 and not self._infer_input_q.full():
                    self._infer_input_q.put_nowait((frame1_crop.copy(), frame2.copy()))

                try:
                    kpv, kpc, f1c, _ = self._infer_output_q.get_nowait()
                    last_kp_video = kpv
                    last_kp_camera = kpc
                    last_frame1_crop = f1c
                except queue.Empty:
                    pass

                # ── Arrancar audio / countdown ────────────────────────
                if not _audio_started:
                    self._video_seek_pos = 0; self._video_seek_req.set()
                    while self._video_seek_req.is_set(): time.sleep(0.001)
                    frame_counter = 0
                    _ref_panel = np.zeros((self.screen_h, self.screen_w // 2, 3), dtype=np.uint8)
                    _label = "PREPARATE PARA EL EJERCICIO DUO"
                    _lsz = cv2.getTextSize(_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                    cv2.putText(_ref_panel, _label, (self.screen_w // 4 - _lsz[0] // 2, self.screen_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 50, 90), 2)
                    _cd_cam = frame2.copy()
                    for countdown_val, dur in [(3, 1.0), (2, 1.0), (1, 1.0), ("INICIA", 0.65)]:
                        t_start = time.time()
                        while True:
                            elapsed = time.time() - t_start
                            if elapsed >= dur: break
                            progress = max(0.0, 1.0 - elapsed / dur)
                            try: _cd_cam = self._camera_queue.get_nowait()
                            except Exception: pass
                            disp = self._compose_letterbox(_ref_panel, _cd_cam, self.screen_w, self.screen_h)
                            disp = JustDanceView.draw_pregame_countdown(disp, countdown_val, progress)
                            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                            surf = pygame.image.frombuffer(rgb.flatten(), (self.screen_w, self.screen_h), 'RGB')
                            screen.blit(surf, (0, 0)); pygame.display.flip(); game_clock.tick(30)
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT: self._cleanup(); return "lobby"
                    self._video_seek_pos = 0; self._video_seek_req.set()
                    while self._video_seek_req.is_set(): time.sleep(0.001)
                    frame_counter = 0; frame_video_time = 0.0
                    if audio_enabled:
                        self.play_sound(audio_path)
                    else:
                        self._song_start_time = time.time()
                    _audio_started = True; _audio_started_time = time.time()

                # Keypoints
                if self._pose_loader.available and self._pose_loader.multi_person:
                    reference_persons = self._pose_loader.get_multi_at(video_time)
                else:
                    reference_persons = last_kp_video or {}
                camera_persons = last_kp_camera or {}

                # ── Evaluación ─────────────────────────────────────────
                in_eval_window = (self.key_pose_time - 1.5 <= video_time <= self.loop_end)
                
                if self._pose_loader.available and self._pose_loader.multi_person:
                    ref_pose_data = self._pose_loader.get_multi_at(self.key_pose_time)
                else:
                    ref_pose_data = reference_persons

                if in_eval_window and camera_persons and ref_pose_data:
                    team_score, raw_team, person_scores_step, visible_users = self._score_duo_pose(
                        ref_pose_data, camera_persons, frame1, frame2
                    )
                    
                    if team_score > self.rep_best_similarity:
                        self.rep_best_similarity = team_score
                        self.rep_best_rating = JustDanceModel.get_pose_rating(team_score)
                        
                    if team_score >= 55: # Éxito!
                        self._play_success_chime()
                        self._add_rep_result({
                            "rep_idx": self.current_rep,
                            "status": self.rep_best_rating,
                            "similarity": float(self.rep_best_similarity)
                        })
                        self._record_pose_evaluation(raw_team, team_score, self.rep_best_rating)
                        
                        from just_dance_view import spawn_rating_particles
                        count = 50 if self.rep_best_rating == "PERFECT" else (35 if self.rep_best_rating == "GREAT" else 20)
                        _fx_particles += spawn_rating_particles(self.screen_w // 2, self.screen_h // 2, self.rep_best_rating, count=count)
                        
                        self.current_rep += 1
                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True
                        else:
                            self.loop_start = self.current_rep * self.rep_duration
                            self.loop_end = (self.current_rep + 1) * self.rep_duration
                            self.key_pose_time = self.loop_start + self.key_pose_offset
                            self.rep_evaluated = False
                            self.rep_best_similarity = 0.0
                            self.rep_best_rating = "MISS"
                            self._seek_to(self.loop_start, audio_path)
                            rating_frames_left = RATING_DISPLAY_FRAMES
                            continue

                # ── Dibujar ────────────────────────────────────────────
                frame1 = self._draw_duo_skeletons(frame1, reference_persons, colors_ref)
                frame2 = self._draw_duo_skeletons(frame2, camera_persons, colors_cam)
                # Parada dura: evita que el ejercicio siga si ya alcanzó las repeticiones objetivo.
                if self.current_rep >= self.repetitions or evaluator.state.completed_reps >= self.repetitions:
                    self.current_rep = min(max(self.current_rep, evaluator.state.completed_reps), self.repetitions)
                    force_exit_reps = True
                    continue

                combined_frame = self._compose_letterbox(frame1, frame2, self.screen_w, self.screen_h)
                
                combined_frame = JustDanceView.draw_game_hud(
                    combined_frame,
                    score=self.total_points, combo=combo, multiplier=multiplier,
                    sync=int(self.rep_best_similarity), time_left=max(0, int(song_length - video_time)),
                    song_name=self.exercise_name + " DUO",
                    difficulty=self.model.difficulty, max_score=max_score,
                    player_name=self._player_name, avatar_color_idx=self._avatar_color_idx,
                    repetitions=self.repetitions, current_rep=self.current_rep, rep_results=self.rep_results
                )

                if self.rep_best_rating and self.rep_best_rating != "MISS" and rating_frames_left > 0:
                    combined_frame = JustDanceView.draw_pose_rating(
                        combined_frame, self.rep_best_rating, rating_frames_left, RATING_DISPLAY_FRAMES
                    )
                    rating_frames_left -= 1

                combined_frame = telemetry.draw_overlay(combined_frame)
                if self.metrics is not None:
                    combined_frame = self.metrics.draw_overlay(combined_frame)

                if _fx_particles:
                    from just_dance_view import update_and_draw_particles
                    _fx_particles = update_and_draw_particles(combined_frame, _fx_particles)

                rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                surf = pygame.image.frombuffer(rgb.flatten(), (self.screen_w, self.screen_h), "RGB")
                screen.blit(surf, (0, 0))
                pygame.display.flip()

        self._best_combo = best_combo
        self.performance_stats = self._build_performance_stats()

        self.rehab_summary = None
        if getattr(self, "rehab_bridge", None) is not None:
            try:
                self.rehab_summary = self.rehab_bridge.finish()
                if isinstance(self.performance_stats, dict):
                    self.performance_stats["rehab_summary"] = self.rehab_summary
            except Exception as e:
                print(f"[WARN] No se pudo cerrar rehab_bridge: {e}")
        self._game_duration = song_length
        self._cleanup()
        return "lobby"

    def _add_rep_result(self, entry: dict) -> None:
        self._chrono_idx += 1
        entry["_chrono_idx"] = self._chrono_idx
        self.rep_results.append(entry)

    def _update_error_detector(self, eval_result: dict) -> None:
        if self._rep_error_detector is not None and eval_result:
            try:
                error_info = self._rep_error_detector.evaluate(eval_result)
                self.error_count = error_info.get("error_count", self.error_count)
                if error_info.get("error_just_counted") and error_info.get("error_type") == "aborted":
                    self._chrono_idx += 1
                    self.error_attempts.append({
                        "rep_idx": self._chrono_idx,
                        "_chrono_idx": self._chrono_idx,
                        "status": "INCORRECTO",
                        "similarity": 0.0,
                        "duration_s": 0.0,
                        "error_type": "aborted",
                    })
            except Exception as e:
                pass

    def process_frames(self, audio_path=None):
        from just_dance_model import JustDanceModel

        if isinstance(getattr(self, "rehab_cfg", None), dict) and self.rehab_cfg.get("tracking_type") == "pose33":
            return self._process_frames_pose33(audio_path)

        if isinstance(getattr(self, "rehab_cfg", None), dict) and self.rehab_cfg.get("tracking_type") == "hand21":
            return self._process_frames_hand21(audio_path)

        if self._duo_mode:
            return self._process_frames_duo(audio_path)

        RATING_DISPLAY_FRAMES = 80
        total_frames = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap1.get(cv2.CAP_PROP_FPS) or 30
        song_length = total_frames / fps if total_frames > 0 else 999

        audio_enabled = bool(audio_path and os.path.exists(audio_path))
        if audio_enabled:
            try:
                from mutagen.mp3 import MP3
                song_length = MP3(audio_path).info.length
                print(f"[INFO] Duracion audio: {song_length:.1f}s")
            except Exception as e:
                print(f"[WARN] Usando duracion video: {e}")
        else:
            print("[INFO] Audio opcional: usando duracion del video.")

        max_score = self.model.max_score

        game_clock = pygame.time.Clock()
        telemetry = TelemetryMonitor()
        screen = pygame.display.get_surface()
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.NOFRAME)

        if self.metrics is not None:
            self.metrics.start()

        frame_counter = 0
        last_kp_video = None
        last_kp_camera = None
        last_frame1 = None
        last_frame_video_time = 0.0
        last_frame1_crop = None
        _last_cam_frame = None
        video_ended = False
        _audio_started = False
        _audio_started_time = 0.0
        current_score = 0
        current_rating = None
        rating_frames_left = 0
        warning_frames = 0
        combo = best_combo = 0
        multiplier = 1
        self.total_points = 0
        SEND_EVERY = 8
        _fx_particles = []

        # ── Modo trayectoria clínica ──────────────────────────────────────────
        # En este modo el video de referencia es guía visual y se reproduce en loop.
        # La repetición se cuenta por recorrido: start -> target -> return.
        rehab_loop_mode = (
            getattr(self, "rehab_bridge", None) is not None
            and isinstance(self.rehab_cfg, dict)
            and self.rehab_cfg.get("evaluation_mode") == "trajectory"
        )
        if rehab_loop_mode:
            print("[INFO] Modo rehab trajectory activo: video en loop hasta completar repeticiones.")

        # Inicializar repetición clínica
        self.current_rep = 0
        self.rep_results = []
        self.error_count = 0
        self.loop_start = 0.0
        self.loop_end = self.rep_duration
        self.key_pose_time = self.loop_start + self.key_pose_offset
        self.rep_evaluated = False
        self.rep_best_similarity = 0.0
        self.rep_best_rating = "MISS"

        force_exit_reps = False

        while self.cap1.isOpened() and not force_exit_reps:
            game_clock.tick(30)
            telemetry.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._cleanup()
                    return "lobby"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self._cleanup()
                        return "lobby"
                    elif event.key == pygame.K_SPACE:
                        # Saltar repetición manualmente
                        self._play_success_chime()
                        self._add_rep_result({
                            "rep_idx": self.current_rep,
                            "status": "OMITIDA",
                            "similarity": float(self.rep_best_similarity)
                        })
                        self._record_pose_evaluation(0, int(self.rep_best_similarity), "MISS")
                        self.current_rep += 1
                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True
                        else:
                            self.loop_start = self.current_rep * self.rep_duration
                            self.loop_end = (self.current_rep + 1) * self.rep_duration
                            self.key_pose_time = self.loop_start + self.key_pose_offset
                            self.rep_evaluated = False
                            self.rep_best_similarity = 0.0
                            self.rep_best_rating = "MISS"
                            self._seek_to(self.loop_start, audio_path)

            if force_exit_reps:
                break

            _frame_ctx = self.metrics.frame_timer() if self.metrics else _NullCtx()
            with _frame_ctx:
                if not video_ended:
                    try:
                        ret1, frame1, frame_video_time = self._video_queue.get(timeout=0.033)
                        if not ret1 or frame1 is None or frame1.size == 0:
                            video_ended = True
                            frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                            frame_video_time = last_frame_video_time
                        else:
                            last_frame1 = frame1.copy()
                            last_frame_video_time = frame_video_time
                    except queue.Empty:
                        frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                        frame_video_time = last_frame_video_time
                else:
                    frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    frame_video_time = last_frame_video_time

                try:
                    _last_cam_frame = self._camera_queue.get_nowait()
                except queue.Empty:
                    pass
                frame2 = _last_cam_frame if _last_cam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                video_time = frame_video_time if _audio_started else 0.0

                # Fin del ejercicio / loop de referencia
                if rehab_loop_mode:
                    # En modo trajectory, el video NO termina la sesión.
                    # Si el video de referencia acaba, se reinicia como guía visual.
                    if video_ended:
                        self._seek_to(0.0, audio_path if audio_enabled else None)
                        video_ended = False
                        last_frame1 = None
                        last_kp_video = None
                        last_frame1_crop = None
                        last_frame_video_time = 0.0

                        # Reset de sincronización visual para evitar acumulación de tiempo.
                        frame_counter = 0
                        self._song_start_time = time.time()
                        _audio_started_time = time.time()

                        continue    

                    # Si hubiera audio, también se puede reiniciar.
                    if audio_enabled and _audio_started and not pygame.mixer.music.get_busy() and (time.time() - _audio_started_time > 2.0):
                        self._seek_to(0.0, audio_path)
                        video_ended = False
                        continue
                else:
                    # Comportamiento viejo: cuando acaba video/audio, termina sesión.
                    if video_ended and not audio_enabled:
                        break
                    if audio_enabled and _audio_started and not pygame.mixer.music.get_busy() and (time.time() - _audio_started_time > 2.0):
                        break

                # Segment looping
                if (not rehab_loop_mode) and (video_time >= self.loop_end or video_ended):
                    if not self.rep_evaluated:
                        self.rep_best_similarity = 0.0
                        self.rep_best_rating = "MISS"
                        self._seek_to(self.loop_start, audio_path)
                        video_ended = False
                        continue

                frame_counter += 1
                char_region = self.char_info.get("char_info") if self.char_info else None
                frame1_crop = self._crop_to_character(frame1, char_region)

                if not video_ended and frame_counter % SEND_EVERY == 0 and not self._infer_input_q.full():
                    self._infer_input_q.put_nowait((frame1_crop.copy(), frame2.copy()))

                try:
                    kpv, kpc, f1c, _ = self._infer_output_q.get_nowait()
                    last_kp_video = kpv
                    last_kp_camera = kpc
                    last_frame1_crop = f1c
                except queue.Empty:
                    pass

                # ── Arrancar audio / countdown ────────────────────────
                if not _audio_started:
                    self._video_seek_pos = 0; self._video_seek_req.set()
                    while self._video_seek_req.is_set(): time.sleep(0.001)
                    frame_counter = 0
                    _ref_panel = np.zeros((self.screen_h, self.screen_w // 2, 3), dtype=np.uint8)
                    _label = "PREPARATE PARA EL EJERCICIO"
                    _lsz = cv2.getTextSize(_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                    cv2.putText(_ref_panel, _label, (self.screen_w // 4 - _lsz[0] // 2, self.screen_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 50, 90), 2)
                    _cd_cam = frame2.copy()
                    for countdown_val, dur in [(3, 1.0), (2, 1.0), (1, 1.0), ("INICIA", 0.65)]:
                        t_start = time.time()
                        while True:
                            elapsed = time.time() - t_start
                            if elapsed >= dur: break
                            progress = max(0.0, 1.0 - elapsed / dur)
                            try: _cd_cam = self._camera_queue.get_nowait()
                            except Exception: pass
                            disp = self._compose_letterbox(_ref_panel, _cd_cam, self.screen_w, self.screen_h)
                            disp = JustDanceView.draw_pregame_countdown(disp, countdown_val, progress)
                            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                            surf = pygame.image.frombuffer(rgb.flatten(), (self.screen_w, self.screen_h), 'RGB')
                            screen.blit(surf, (0, 0)); pygame.display.flip(); game_clock.tick(30)
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT: self._cleanup(); return "lobby"
                    self._video_seek_pos = 0; self._video_seek_req.set()
                    while self._video_seek_req.is_set(): time.sleep(0.001)
                    frame_counter = 0; frame_video_time = 0.0
                    if audio_enabled:
                        self.play_sound(audio_path)
                    else:
                        self._song_start_time = time.time()
                    _audio_started = True; _audio_started_time = time.time()

                # Keypoints
                if self._pose_loader.available:
                    json_data = self._pose_loader.get_at(video_time, self._reference_person_id)
                    key_points_video = json_data["keypoints"]
                else:
                    key_points_video = last_kp_video
                key_points_camera = last_kp_camera
                user_body_visible = self.is_body_visible(key_points_camera, frame2)

                if key_points_camera is not None:
                    self._record_tracking_quality(key_points_camera)
                    _cam_angles = self._get_pose_angles(frame2, key_points_camera) if user_body_visible else {}

                # ── Evaluación clínica separada por frame ───────────────────
                rehab_result = None
                if user_body_visible and key_points_camera is not None and getattr(self, "rehab_bridge", None) is not None:
                    try:
                        rehab_result = self.rehab_bridge.evaluate(
                            keypoints=key_points_camera,
                            angles=_cam_angles,
                            frame_index=frame_counter,
                            timestamp_s=video_time,
                        )
                        self._update_error_detector(rehab_result)
                    except Exception as e:
                        print(f"[WARN] Error evaluando rehab_bridge: {e}")
                        rehab_result = None

                # ── Conteo clínico de repeticiones por trayectoria ───────────
                if rehab_loop_mode and rehab_result and rehab_result.get("rep_completed"):
                    try:
                        last_rep = rehab_result.get("last_rep") or {}
                        best_score_rep = float(last_rep.get("best_score", rehab_result.get("score", 0.0)))
                        rehab_rating = JustDanceModel.get_pose_rating(best_score_rep)

                        if getattr(self, "rehab_bridge", None) is not None:
                            rep_result = self.rehab_bridge.build_rep_result(
                                rep_idx=int(rehab_result.get("completed_reps", self.current_rep + 1)),
                                rating=rehab_rating,
                                similarity=best_score_rep,
                            )
                        else:
                            rep_result = {
                                "rep_idx": int(rehab_result.get("completed_reps", self.current_rep + 1)),
                                "status": rehab_rating,
                                "similarity": best_score_rep,
                            }

                        self._add_rep_result(rep_result)
                        self.current_rep = int(rehab_result.get("completed_reps", self.current_rep + 1))
                        self.rep_best_similarity = best_score_rep
                        self.rep_best_rating = rehab_rating
                        self.total_points += int(best_score_rep * 10)

                        print(f"[REHAB] Repetición {self.current_rep}/{self.repetitions} completada | score={best_score_rep:.1f}")

                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True
                            continue

                    except Exception as e:
                        print(f"[WARN] Error contando repetición rehab: {e}")

                # Esto debe ir FUERA del if de rep_completed
                if key_points_camera is not None:
                    self._session_logger.log_frame(
                        video_time_s=video_time,
                        keypoints_raw=key_points_camera,
                        angles=_cam_angles if _cam_angles else None,
                    )

                    if self.metrics is not None:
                        self.metrics.log_pose_frame(
                            keypoints_raw=key_points_camera,
                            angles=_cam_angles if _cam_angles else None,
                        )


                # Poses de referencia para la pose clave
                if self._pose_loader.available:
                    ref_pose_data = self._pose_loader.get_at(self.key_pose_time, self._reference_person_id)
                    pose_target_angles = ref_pose_data["angles"]
                    pose_target_kp_raw = ref_pose_data["keypoints"]
                else:
                    pose_target_angles = self._get_beat_angles(self.key_pose_time)
                    pose_target_kp_raw = None

                # ── Evaluación ─────────────────────────────────────────
                in_eval_window = (self.key_pose_time - 1.5 <= video_time <= self.loop_end)

                active_joints = self.rehab_cfg.get("active_joints") if self.rehab_cfg else None

                if (not rehab_loop_mode) and in_eval_window and user_body_visible and pose_target_angles is not None:
                    ua = self._get_pose_angles(frame2, key_points_camera)
                    raw_score = self._compare_pose_angles(pose_target_angles, ua)

                    _target_angles_filtered = {k: v for k, v in pose_target_angles.items()
                                                if not active_joints or k in active_joints}
                    _camera_angles_filtered = {k: v for k, v in self.angles_camera.items()
                                                if not active_joints or k in active_joints}

                    current_score = JustDanceModel.adjusted_pose_score(
                        raw_score=raw_score,
                        pose_angles_target=_target_angles_filtered,
                        angles_camera=_camera_angles_filtered,
                        frame_video=frame1_crop,
                        kp_video=pose_target_kp_raw,
                        frame_camera=frame2,
                        kp_camera=key_points_camera,
                        movement_threshold=self.model.movement_threshold,
                        distinctiveness_threshold=self.model.distinctiveness_threshold,
                        angle_weight=self.model.angle_weight,
                        vector_weight=self.model.vector_weight,
                    )

                    if current_score > self.rep_best_similarity:
                        self.rep_best_similarity = current_score
                        self.rep_best_rating = JustDanceModel.get_pose_rating(current_score)

                    if current_score >= 55:
                        self._play_success_chime()
                        self._add_rep_result({
                            "rep_idx": self.current_rep,
                            "status": self.rep_best_rating,
                            "similarity": float(self.rep_best_similarity)
                        })
                        self._record_pose_evaluation(raw_score, current_score, self.rep_best_rating)

                        if "count" in self.joint_stats:
                            self.joint_stats["count"] += 1
                            soft_limit = self.model.angle_threshold * 1.5
                            for jt in _target_angles_filtered:
                                if jt in ua:
                                    diff = abs(_target_angles_filtered[jt] - ua[jt])
                                    self.joint_stats[jt] += max(0.0, 1.0 - (diff / soft_limit) ** 2)

                        from just_dance_view import spawn_rating_particles
                        count = 50 if self.rep_best_rating == "PERFECT" else (35 if self.rep_best_rating == "GREAT" else 20)
                        _fx_particles += spawn_rating_particles(self.screen_w // 2, self.screen_h // 2, self.rep_best_rating, count=count)

                        self.current_rep += 1
                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True
                        else:
                            self.loop_start = self.current_rep * self.rep_duration
                            self.loop_end = (self.current_rep + 1) * self.rep_duration
                            self.key_pose_time = self.loop_start + self.key_pose_offset
                            self.rep_evaluated = False
                            self.rep_best_similarity = 0.0
                            self.rep_best_rating = "MISS"
                            self._seek_to(self.loop_start, audio_path)
                            rating_frames_left = RATING_DISPLAY_FRAMES
                            continue

                # ── Dibujar ────────────────────────────────────────────
                if not self._pose_loader.available:
                    if key_points_video is not None:
                        frame1 = JustDanceView.draw_skeleton(frame1, key_points_video, color=(0, 255, 0))

                    frame2 = JustDanceView.draw_skeleton(frame2, key_points_camera, color=(0, 165, 255))
                
                combined_frame = self._compose_letterbox(frame1, frame2, self.screen_w, self.screen_h)
                combined_frame = JustDanceView.draw_game_hud(
                    combined_frame,
                    score=self.total_points, combo=combo, multiplier=multiplier,
                    sync=int(self.rep_best_similarity), time_left=max(0, int(song_length - video_time)),
                    song_name=self.exercise_name,
                    difficulty=self.model.difficulty, max_score=max_score,
                    player_name=self._player_name, avatar_color_idx=self._avatar_color_idx,
                    repetitions=self.repetitions, current_rep=self.current_rep, rep_results=self.rep_results,
                    error_state="correct" if self.error_count == 0 else "incorrect",
                    error_count=self.error_count,
                )

                if self.rep_best_rating and self.rep_best_rating != "MISS" and rating_frames_left > 0:
                    combined_frame = JustDanceView.draw_pose_rating(
                        combined_frame, self.rep_best_rating, rating_frames_left, RATING_DISPLAY_FRAMES
                    )
                    rating_frames_left -= 1

                if not user_body_visible:
                    combined_frame = JustDanceView.draw_body_warning(combined_frame)
                if getattr(self, "rehab_bridge", None) is not None:
                    combined_frame = self.rehab_bridge.draw_feedback_overlay(combined_frame)

                combined_frame = telemetry.draw_overlay(combined_frame)
                if self.metrics is not None:
                    combined_frame = self.metrics.draw_overlay(combined_frame)

                if _fx_particles:
                    from just_dance_view import update_and_draw_particles
                    _fx_particles = update_and_draw_particles(combined_frame, _fx_particles)

                rgb_combined = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                surf = pygame.image.frombuffer(rgb_combined.flatten(), (self.screen_w, self.screen_h), 'RGB')
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                # En modo trajectory NO sincronizamos por tiempo de canción/video.
                # El video es solo referencia visual en loop y el paciente va a su ritmo.
                if self._song_start_time and not rehab_loop_mode:
                    expected_time = frame_counter / self.frame1_rate
                    surplus = expected_time - (time.time() - self._song_start_time)
                    if surplus > 0.002:
                        time.sleep(surplus * 0.9)

            if self.metrics is not None:
                self.metrics.record_frame()

        # ── Fin del loop ───────────────────────────────────────────────
        self._infer_stop.set(); self._video_stop.set()
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
        self.performance_stats = self._build_performance_stats()

        self._session_logger.fps = self.frame1_rate
        self._session_path = self._session_logger.save(
            final_score = self.total_points,
            extra_metadata = {
                "best_combo": best_combo,
                "joint_stats": self.joint_stats,
                "performance_stats": self.performance_stats,
                "rehab_summary": getattr(self, "rehab_summary", None),
                "rep_results": self.rep_results,
            }
        )

        if self.metrics is not None:
            self.metrics.finish(final_score=self.total_points)

        self._game_duration = song_length
        self._best_combo = best_combo
        return None



    def _process_frames_hand21(self, audio_path=None):
        """Loop para ejercicios tracking_type='hand21'."""
        from just_dance_model import JustDanceModel
        from pose_backends.mediapipe_hand21_backend import MediaPipeHand21Backend
        from rehab_core.hand21_evaluator import Hand21OpenCloseEvaluator
        from rehab_core.hand21_static_wrist_evaluator import Hand21StaticWristAlignmentEvaluator
        from rehab_core.pose33_trajectory_evaluator import Pose33TrajectoryEvaluator

        total_frames = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap1.get(cv2.CAP_PROP_FPS) or 30
        song_length = total_frames / fps if total_frames > 0 else 999

        game_clock = pygame.time.Clock()
        telemetry = TelemetryMonitor()
        screen = pygame.display.get_surface()
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.NOFRAME)

        if self.metrics is not None:
            self.metrics.start()

        print('[INFO] process_frames Hand21 activo.')

        config = {
            'open_range': self.rehab_cfg.get('open_range', [145.0, 180.0]),
            'closed_range': self.rehab_cfg.get('closed_range', [0.0, 105.0]),
            'min_open_fingers': self.rehab_cfg.get('min_open_fingers', 4),
            'min_closed_fingers': self.rehab_cfg.get('min_closed_fingers', 3),
            'min_stable_frames': self.rehab_cfg.get('min_stable_frames', 5),
            'cooldown_frames': self.rehab_cfg.get('cooldown_frames', 10),
            'target_repetitions': self.repetitions,
            'stop_at_target': True,
            'min_tracking_quality': self.rehab_cfg.get('min_tracking_quality', 0.55),
            'require_thumb_open': self.rehab_cfg.get('require_thumb_open', True),
            'require_thumb_closed': self.rehab_cfg.get('require_thumb_closed', True),
            'thumb_open_range': self.rehab_cfg.get('thumb_open_range', [145.0, 180.0]),
            'thumb_closed_range': self.rehab_cfg.get('thumb_closed_range', [0.0, 165.0]),
        }

        # Resetear contadores de tracking para esta ejecucion
        self._hand21_tracking_failures = 0

        # Reutilizar backend entre ejecuciones para evitar agotamiento de recursos MediaPipe
        if not hasattr(self, '_hand_backend') or self._hand_backend is None:
            try:
                import backend_preloader
                self._hand_backend = backend_preloader.get_hand21()
            except Exception:
                self._hand_backend = None
            if self._hand_backend is None:
                self._hand_backend = MediaPipeHand21Backend(
                    'model/hand_landmarker.task',
                    num_hands=1,
                    min_hand_detection_confidence=self.rehab_cfg.get('min_hand_detection_confidence', 0.35),
                    min_hand_presence_confidence=self.rehab_cfg.get('min_hand_presence_confidence', 0.35),
                    min_tracking_confidence=self.rehab_cfg.get('min_tracking_confidence', 0.35),
                )
        hand_backend = self._hand_backend
        if self.rehab_cfg.get('evaluation_mode') == 'trajectory':
            evaluator = Pose33TrajectoryEvaluator(self.rehab_cfg)
        elif self.rehab_cfg.get('evaluation_mode') == 'static_wrist_alignment':
            evaluator = Hand21StaticWristAlignmentEvaluator(config)
        else:
            evaluator = Hand21OpenCloseEvaluator(config)

        self.current_rep = 0
        self.rep_results = []
        self.error_count = 0
        self.rep_best_similarity = 0.0
        self.rep_best_rating = 'MISS'
        self.total_points = 0

        last_frame1 = None
        last_frame_video_time = 0.0
        last_cam_frame = None
        video_ended = False
        frame_counter = 0
        force_exit_reps = False
        combo = best_combo = 0
        multiplier = 1
        max_score = self.model.max_score
        start_time = time.time()

        connections = [
            ('wrist', 'thumb_cmc'), ('thumb_cmc', 'thumb_mcp'), ('thumb_mcp', 'thumb_ip'), ('thumb_ip', 'thumb_tip'),
            ('wrist', 'index_mcp'), ('index_mcp', 'index_pip'), ('index_pip', 'index_dip'), ('index_dip', 'index_tip'),
            ('wrist', 'middle_mcp'), ('middle_mcp', 'middle_pip'), ('middle_pip', 'middle_dip'), ('middle_dip', 'middle_tip'),
            ('wrist', 'ring_mcp'), ('ring_mcp', 'ring_pip'), ('ring_pip', 'ring_dip'), ('ring_dip', 'ring_tip'),
            ('wrist', 'pinky_mcp'), ('pinky_mcp', 'pinky_pip'), ('pinky_pip', 'pinky_dip'), ('pinky_dip', 'pinky_tip'),
        ]

        def choose_hand(result):
            if not result.detected or not result.hands:
                return None
            return max(result.hands, key=lambda h: h.tracking_quality)

        def px(point, w, h):
            return int(point.x * w), int(point.y * h)

        def draw_hand(frame, hand):
            h, w = frame.shape[:2]
            landmarks = hand.landmarks
            for a, b in connections:
                if a in landmarks and b in landmarks:
                    cv2.line(frame, px(landmarks[a], w, h), px(landmarks[b], w, h), (40, 220, 120), 2, cv2.LINE_AA)
            for point in landmarks.values():
                cv2.circle(frame, px(point, w, h), 4, (0, 220, 220), -1, cv2.LINE_AA)
            return frame

        def draw_feedback(frame, eval_result):
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (18, h - 92), (w - 18, h - 22), (20, 35, 65), -1)
            cv2.rectangle(frame, (18, h - 92), (w - 18, h - 22), (180, 120, 255), 2)
            feedback = str(eval_result.get('feedback', ''))[:110]
            cv2.putText(frame, feedback, (34, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 245, 250), 2, cv2.LINE_AA)
            if self.rehab_cfg.get('evaluation_mode') == 'trajectory':
                primary = self.rehab_cfg.get('primary_angle', '')
                raw_val = hand.angles.get(primary) if (hand is not None and isinstance(hand.angles, dict)) else None
                ema_val = eval_result.get('primary_value')
                raw_txt = '--' if raw_val is None else f'{raw_val:.1f}'
                ema_txt = '--' if ema_val is None else f'{ema_val:.1f}'
                phase = str(eval_result.get('phase', ''))
                in_s_txt = 'SI' if eval_result.get('in_start') else 'no'
                in_t_txt = 'SI' if eval_result.get('in_target') else 'no'
                cv2.putText(frame, f'{primary}: crudo={raw_txt} EMA={ema_txt} | fase={phase} | inicio={in_s_txt} obj={in_t_txt}', (34, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 170, 185), 1, cv2.LINE_AA)
            else:
                avg = eval_result.get('avg_pip_flexion')
                avg_txt = '--' if avg is None else f'{avg:.1f}'
                state = eval_result.get('hand_state', '--')
                open_count = eval_result.get('open_count', 0)
                closed_count = eval_result.get('closed_count', 0)
                thumb_value = eval_result.get('thumb_value')
                thumb_txt = '--' if thumb_value is None else f'{thumb_value:.1f}'
                thumb_closed = eval_result.get('thumb_closed', False)
                cv2.putText(frame, f'estado: {state} | avg: {avg_txt} | dedos: {closed_count}/4 | pulgar: {thumb_txt} cerrado={thumb_closed}', (34, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 170, 185), 1, cv2.LINE_AA)
            return frame

        try:
            while self.cap1.isOpened() and not force_exit_reps:
                game_clock.tick(30)
                telemetry.tick()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._cleanup()
                        return 'lobby'
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            self._cleanup()
                            return 'lobby'
                        elif event.key == pygame.K_SPACE:
                            self._add_rep_result({'rep_idx': self.current_rep + 1, 'status': 'OMITIDA', 'similarity': float(self.rep_best_similarity)})
                            self.current_rep += 1
                            if self.current_rep >= self.repetitions:
                                force_exit_reps = True

                if force_exit_reps:
                    break

                try:
                    ret1, frame1, frame_video_time = self._video_queue.get(timeout=0.033)
                    if not ret1 or frame1 is None or frame1.size == 0:
                        video_ended = True
                        frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                        frame_video_time = last_frame_video_time
                    else:
                        last_frame1 = frame1.copy()
                        last_frame_video_time = frame_video_time
                except queue.Empty:
                    frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    frame_video_time = last_frame_video_time

                if video_ended:
                    self._seek_to(0.0, audio_path if audio_path and os.path.exists(audio_path) else None)
                    video_ended = False
                    last_frame1 = None
                    last_frame_video_time = 0.0
                    frame_counter = 0
                    continue

                try:
                    last_cam_frame = self._camera_queue.get_nowait()
                except queue.Empty:
                    pass
                frame2 = last_cam_frame if last_cam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

                frame_counter += 1
                video_time = frame_video_time
                timestamp_s = time.time() - start_time

                result = hand_backend.detect_bgr(frame2)
                hand = choose_hand(result)
                user_body_visible = False
                eval_result = {
                    'ok': False,
                    'phase': evaluator.state.phase,
                    'completed_reps': evaluator.state.completed_reps,
                    'feedback': 'Muestra una mano a la camara.',
                    'tracking_quality': 0.0,
                    'hand_state': 'sin mano',
                    'open_count': 0,
                    'closed_count': 0,
                    'thumb_value': None,
                    'thumb_open': False,
                    'thumb_closed': False,
                }

                if hand is not None:
                    user_body_visible = True
                    frame2 = draw_hand(frame2, hand)
                    if self.rehab_cfg.get('evaluation_mode') == 'trajectory':
                        # Check required landmarks (wrist + MCP) first; fall back to quality heuristic
                        required = self.rehab_cfg.get('required_landmarks', [])
                        has_required = hand.required_visible(required) if required else True
                        min_q = self.rehab_cfg.get('min_tracking_confidence', 0.35)
                        quality_ok = hand.tracking_quality >= min_q
                        tq_ok = has_required or quality_ok
                        # Grace period: tolera pérdidas breves de tracking
                        if not hasattr(self, '_hand21_tracking_failures'):
                            self._hand21_tracking_failures = 0
                        if tq_ok:
                            self._hand21_tracking_failures = 0
                        else:
                            self._hand21_tracking_failures += 1
                            if self._hand21_tracking_failures <= 3:
                                tq_ok = True
                        tq = {'ok': tq_ok}
                        eval_result = evaluator.evaluate(
                            angles=hand.angles,
                            pose33_result=None,
                            tracking_quality=tq,
                            frame_index=frame_counter,
                            timestamp_s=timestamp_s
                        )
                    else:
                        eval_result = evaluator.evaluate(hand_result=hand, frame_index=frame_counter, timestamp_s=timestamp_s)
                    self._update_error_detector(eval_result)
                    self.rep_best_similarity = max(self.rep_best_similarity, float(eval_result.get('best_score_current_rep', 0.0)))

                    if eval_result.get('rep_completed'):
                        last_rep = eval_result.get('last_rep') or {}
                        best_score_rep = float(last_rep.get('best_score', eval_result.get('score', 0.0)))

                        # HAND21 REP0 MISS FIX: ignorar repeticiones fantasma con score bajo.
                        valid_rep_score = float(self.rehab_cfg.get('valid_rep_score', 70.0))
                        if best_score_rep < valid_rep_score:
                            print(f'[HAND21] Rep ignorada por score bajo: {best_score_rep:.1f}')
                            continue

                        rehab_rating = JustDanceModel.get_pose_rating(best_score_rep)
                        next_rep_idx = int(getattr(self, 'current_rep', 0)) + 1
                        if next_rep_idx > int(self.repetitions):
                            force_exit_reps = True
                            continue

                        self.current_rep = next_rep_idx
                        self.rep_best_similarity = best_score_rep
                        self.rep_best_rating = rehab_rating
                        self.total_points += int(best_score_rep * 10)
                        rep_entry = {
                            'rep_idx': self.current_rep,
                            'status': rehab_rating,
                            'similarity': best_score_rep,
                            'duration_s': last_rep.get('duration_s'),
                            'too_fast': bool(last_rep.get('too_fast', False)),
                        }
                        if 'compensations' in last_rep:
                            rep_entry['compensations'] = last_rep['compensations']
                            rep_entry['compensation_penalty'] = last_rep.get('compensation_penalty', 0.0)
                        self._add_rep_result(rep_entry)
                        self._record_pose_evaluation(0, int(best_score_rep), rehab_rating)
                        try:
                            self._play_success_chime()
                        except Exception:
                            pass
                        print(f'[HAND21] Repeticion {self.current_rep}/{self.repetitions} completada | score={best_score_rep:.1f}')
                        if self.current_rep >= int(self.repetitions):
                            force_exit_reps = True
                            continue

                # HAND21 FINISH FIX: parada robusta al alcanzar el objetivo de repeticiones.
                try:
                    _completed_now = max(
                        int(getattr(evaluator.state, "completed_reps", 0)),
                        int(eval_result.get("completed_reps", 0)),
                        int(getattr(self, "current_rep", 0)),
                    )
                    if str(eval_result.get("phase", "")) == "done" or _completed_now >= int(self.repetitions):
                        self.current_rep = min(_completed_now, int(self.repetitions))
                        force_exit_reps = True
                        print(f"[HAND21] Objetivo completado: {self.current_rep}/{self.repetitions}. Finalizando ejercicio.")
                        break
                except Exception as _finish_err:
                    print(f"[WARN] Error en parada hand21: {_finish_err}")

                combined_frame = self._compose_letterbox(frame1, frame2, self.screen_w, self.screen_h)
                combined_frame = JustDanceView.draw_game_hud(
                    combined_frame, score=self.total_points, combo=combo, multiplier=multiplier,
                    sync=int(self.rep_best_similarity), time_left=max(0, int(song_length - video_time)),
                    song_name=self.exercise_name, difficulty=self.model.difficulty, max_score=max_score,
                    player_name=self._player_name, avatar_color_idx=self._avatar_color_idx,
                    repetitions=self.repetitions, current_rep=self.current_rep, rep_results=self.rep_results,
                    error_state="correct" if self.error_count == 0 else "incorrect",
                    error_count=self.error_count,
                )
                if not user_body_visible:
                    combined_frame = JustDanceView.draw_body_warning(combined_frame)
                combined_frame = draw_feedback(combined_frame, eval_result)
                combined_frame = telemetry.draw_overlay(combined_frame)
                if self.metrics is not None:
                    combined_frame = self.metrics.draw_overlay(combined_frame)

                rgb_combined = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                surf = pygame.image.frombuffer(rgb_combined.flatten(), (self.screen_w, self.screen_h), 'RGB')
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                if self.metrics is not None:
                    self.metrics.record_frame()
        finally:
            pass  # backend se reutiliza entre ejecuciones; no cerrar aqui

        self._infer_stop.set(); self._video_stop.set()
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        
        # HAND21 LAST REP FIX: asegurar que la ultima repeticion quede registrada
        try:
            completed_final = max(
                int(getattr(evaluator.state, "completed_reps", 0)),
                int(getattr(self, "current_rep", 0)),
            )

            if completed_final >= int(self.repetitions) and len(self.rep_results) < int(self.repetitions):
                missing = int(self.repetitions) - len(self.rep_results)

                last_rep = getattr(evaluator.state, "last_rep", {}) or {}
                score = float(
                    last_rep.get(
                        "best_score",
                        getattr(self, "rep_best_similarity", 100.0) or 100.0
                    )
                )

                rating = JustDanceModel.get_pose_rating(score)

                for _ in range(missing):
                    rep_idx = len(self.rep_results) + 1
                    self._add_rep_result({
                        "rep_idx": rep_idx,
                        "status": rating,
                        "similarity": score,
                    })

                self.current_rep = int(self.repetitions)
                print(f"[HAND21] rep_results sincronizado: {len(self.rep_results)}/{self.repetitions}")

        except Exception as sync_err:
            print(f"[WARN] No se pudo sincronizar rep_results hand21: {sync_err}")
        self.performance_stats = self._build_performance_stats()
        self._game_duration = song_length
        self._best_combo = best_combo
        return None

    def _process_frames_pose33(self, audio_path=None):
        from just_dance_model import JustDanceModel
        from pose_backends.mediapipe33_backend import MediaPipe33Backend
        from rehab_core.angles_33 import compute_angles_33, tracking_quality_for
        from rehab_core.pose33_trajectory_evaluator import Pose33TrajectoryEvaluator

        total_frames = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap1.get(cv2.CAP_PROP_FPS) or 30
        song_length = total_frames / fps if total_frames > 0 else 999

        game_clock = pygame.time.Clock()
        telemetry = TelemetryMonitor()
        screen = pygame.display.get_surface()
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.NOFRAME)

        if self.metrics is not None:
            self.metrics.start()

        print('[INFO] process_frames Pose33 activo.')

        if getattr(self, 'pose33_backend', None) is None:
            try:
                import backend_preloader
                self.pose33_backend = backend_preloader.get_pose33()
            except Exception:
                self.pose33_backend = None
            if self.pose33_backend is None:
                self.pose33_backend = MediaPipe33Backend('model/pose_landmarker_lite.task')
        if getattr(self, 'pose33_evaluator', None) is None:
            self.pose33_evaluator = Pose33TrajectoryEvaluator(self.rehab_cfg)

        self.current_rep = 0
        self.rep_results = []
        self.error_count = 0
        self.rep_best_similarity = 0.0
        self.rep_best_rating = 'MISS'
        self.total_points = 0
        self.pose33_last_feedback = self.rehab_cfg.get('feedback', {}).get('waiting_start', 'Coloca el brazo abajo en posición inicial.')

        last_frame1 = None
        last_frame_video_time = 0.0
        last_cam_frame = None
        video_ended = False
        frame_counter = 0
        force_exit_reps = False
        combo = best_combo = 0
        multiplier = 1
        max_score = self.model.max_score
        start_time = time.time()

        def draw_pose33(frame, landmarks):
            if not isinstance(landmarks, dict):
                return frame
            h, w = frame.shape[:2]
            pairs = [
                ('left_shoulder','right_shoulder'), ('left_shoulder','left_elbow'), ('left_elbow','left_wrist'),
                ('right_shoulder','right_elbow'), ('right_elbow','right_wrist'),
                ('left_shoulder','left_hip'), ('right_shoulder','right_hip'), ('left_hip','right_hip'),
                ('left_hip','left_knee'), ('left_knee','left_ankle'), ('right_hip','right_knee'), ('right_knee','right_ankle'),
            ]
            def px(p):
                return int(p.x * w), int(p.y * h)
            for a, b in pairs:
                pa = landmarks.get(a); pb = landmarks.get(b)
                if pa is not None and pb is not None and pa.confidence >= 0.25 and pb.confidence >= 0.25:
                    cv2.line(frame, px(pa), px(pb), (40, 200, 120), 2, cv2.LINE_AA)
            for p in landmarks.values():
                if getattr(p, 'confidence', 0.0) >= 0.25:
                    cv2.circle(frame, px(p), 3, (0, 180, 180), -1, cv2.LINE_AA)
            return frame

        def draw_feedback(frame, feedback, angles, eval_result, tq=None):
            h, w = frame.shape[:2]

            # ── Bottom bar ──────────────────────────────────────────────────
            cv2.rectangle(frame, (18, h - 88), (w - 18, h - 22), (20, 35, 65), -1)
            cv2.rectangle(frame, (18, h - 88), (w - 18, h - 22), (0, 180, 180), 2)
            cv2.putText(frame, str(feedback or '')[:115], (34, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 245, 250), 2, cv2.LINE_AA)
            primary = self.rehab_cfg.get('primary_angle', '')
            raw_val = angles.get(primary)
            ema_val = eval_result.get('primary_value')
            raw_txt = '--' if raw_val is None else f'{raw_val:.1f}'
            ema_txt = '--' if ema_val is None else f'{ema_val:.1f}'
            phase = str(eval_result.get('phase', ''))
            in_s_txt = 'SI' if eval_result.get('in_start') else 'no'
            in_t_txt = 'SI' if eval_result.get('in_target') else 'no'
            cv2.putText(frame,
                        f'{primary}: crudo={raw_txt} EMA={ema_txt} | fase={phase} | inicio={in_s_txt} obj={in_t_txt}',
                        (34, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 170, 185), 1, cv2.LINE_AA)

            # ── Top diagnostic panel ─────────────────────────────────────────
            px, py, pw, panel_h = 8, 8, min(w - 16, 470), 122
            overlay = frame.copy()
            cv2.rectangle(overlay, (px, py), (px + pw, py + panel_h), (8, 18, 30), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            cv2.rectangle(frame, (px, py), (px + pw, py + panel_h), (55, 65, 85), 1)

            fnt = cv2.FONT_HERSHEY_SIMPLEX
            lh = 22

            ev_state = self.pose33_evaluator.state
            calibrated = ev_state.calibrated
            neutral = ev_state.neutral_value
            if not self.pose33_evaluator.auto_calibrate:
                _pk = self.rehab_cfg.get('primary_angle', '')
                _ss = self.rehab_cfg.get('start_ranges', {}).get(_pk)
                _ts = self.rehab_cfg.get('target_ranges', {}).get(_pk)
                _sr_s2 = f'[{_ss[0]:.0f},{_ss[1]:.0f}]' if _ss else '?'
                _tr_s2 = f'[{_ts[0]:.0f},{_ts[1]:.0f}]' if _ts else '?'
                cal_txt = f'RANGOS FIJOS  inicio={_sr_s2}  objetivo={_tr_s2}'
                cal_col = (160, 160, 200)
            elif calibrated and neutral is not None:
                sr = ev_state.dyn_start_range
                tr = ev_state.dyn_target_range
                sr_s = f'[{sr[0]:.0f},{sr[1]:.0f}]' if sr else '?'
                tr_s = f'[{tr[0]:.0f},{tr[1]:.0f}]' if tr else '?'
                cal_txt = f'CALIB OK  neutro={neutral:.1f}  inicio={sr_s}  objetivo={tr_s}'
                cal_col = (70, 210, 70)
            else:
                n_cal = len(ev_state.calib_samples)
                n_tot = self.pose33_evaluator.calib_frames
                cal_txt = f'CALIBRANDO {n_cal}/{n_tot} frames - mantente quieto'
                cal_col = (80, 155, 215)
            cv2.putText(frame, cal_txt, (px + 6, py + lh), fnt, 0.36, cal_col, 1, cv2.LINE_AA)

            fs = eval_result.get('frames_in_start', 0)
            ft = eval_result.get('frames_in_target', 0)
            in_safe = eval_result.get('in_safe', True)
            safe_s = 'seguro' if in_safe else '!FUERA_RANGO!'
            min_s = self.pose33_evaluator.min_frames_in_start
            min_t = self.pose33_evaluator.min_frames_in_target
            ph_col = (220, 180, 60) if phase == 'going_target' else (60, 200, 220) if phase == 'returning' else (160, 160, 165)
            cv2.putText(frame,
                        f'FASE: {phase}  fs={fs}/{min_s}  ft={ft}/{min_t}  [{safe_s}]',
                        (px + 6, py + lh * 2), fnt, 0.36, ph_col, 1, cv2.LINE_AA)

            cv2.putText(frame,
                        f'crudo={raw_txt}  EMA={ema_txt}  inicio={"SI" if eval_result.get("in_start") else "no"}  objetivo={"SI" if eval_result.get("in_target") else "no"}',
                        (px + 6, py + lh * 3), fnt, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

            if tq and isinstance(tq, dict):
                details = tq.get('details', {})
                parts = []
                for lm_name, lm_info in list(details.items())[:7]:
                    conf = lm_info.get('confidence', 0.0)
                    vis = lm_info.get('visible', False)
                    short = lm_name.replace('right_', 'r').replace('left_', 'l').replace('_', '')
                    parts.append(f'{short}:{conf:.2f}{"+" if vis else "-"}')
                lm_col = (70, 210, 70) if tq.get('ok') else (120, 100, 210)
                cv2.putText(frame, '  '.join(parts), (px + 6, py + lh * 4), fnt, 0.34, lm_col, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'landmarks: sin datos', (px + 6, py + lh * 4),
                            fnt, 0.34, (110, 110, 130), 1, cv2.LINE_AA)

            if not calibrated and self.pose33_evaluator.auto_calibrate:
                n_cal2 = len(ev_state.calib_samples)
                n_tot2 = self.pose33_evaluator.calib_frames
                bar_x, bar_y = px + 10, py + lh * 5 - 4
                bar_w = pw - 20
                filled = int(bar_w * n_cal2 / max(n_tot2, 1))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 7), (35, 45, 55), -1)
                if filled > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 7), (80, 155, 215), -1)

            return frame

        while self.cap1.isOpened() and not force_exit_reps:
            game_clock.tick(30)
            telemetry.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._cleanup()
                    return 'lobby'
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self._cleanup()
                        return 'lobby'
                    elif event.key == pygame.K_SPACE:
                        self._add_rep_result({'rep_idx': self.current_rep + 1, 'status': 'OMITIDA', 'similarity': float(self.rep_best_similarity)})
                        self.current_rep += 1
                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True

            if force_exit_reps:
                break

            try:
                ret1, frame1, frame_video_time = self._video_queue.get(timeout=0.033)
                if not ret1 or frame1 is None or frame1.size == 0:
                    video_ended = True
                    frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    frame_video_time = last_frame_video_time
                else:
                    last_frame1 = frame1.copy()
                    last_frame_video_time = frame_video_time
            except queue.Empty:
                frame1 = last_frame1 if last_frame1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                frame_video_time = last_frame_video_time

            if video_ended:
                self._seek_to(0.0, audio_path if audio_path and os.path.exists(audio_path) else None)
                video_ended = False
                last_frame1 = None
                last_frame_video_time = 0.0
                frame_counter = 0
                continue

            try:
                last_cam_frame = self._camera_queue.get_nowait()
            except queue.Empty:
                pass
            frame2 = last_cam_frame if last_cam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            frame_counter += 1
            video_time = frame_video_time
            timestamp_s = time.time() - start_time
            eval_result = {'ok': False, 'phase': getattr(self.pose33_evaluator.state, 'phase', 'waiting_start'), 'completed_reps': self.current_rep, 'feedback': self.pose33_last_feedback, 'compensations': {}}
            angles = {}
            user_body_visible = False
            tracking_quality = None

            try:
                pose33_result = self.pose33_backend.detect_bgr(frame2)
                if pose33_result.detected:
                    angles = compute_angles_33(pose33_result.image_landmarks, pose33_result.world_landmarks, min_confidence=self.rehab_cfg.get('min_tracking_confidence', 0.35), include_3d=True)
                    tracking_quality = tracking_quality_for(pose33_result.image_landmarks, self.rehab_cfg.get('required_landmarks', []), min_confidence=self.rehab_cfg.get('min_tracking_confidence', 0.35))
                    user_body_visible = bool(tracking_quality.get('ok', False))
                    eval_result = self.pose33_evaluator.evaluate(angles=angles, pose33_result=pose33_result, tracking_quality=tracking_quality, frame_index=frame_counter, timestamp_s=timestamp_s)
                    self._update_error_detector(eval_result)
                    self.pose33_last_feedback = eval_result.get('feedback', self.pose33_last_feedback)
                    if frame_counter % 30 == 0:
                        _pa = self.rehab_cfg.get('primary_angle', '')
                        _rv = angles.get(_pa)
                        _ev = eval_result.get('primary_value')
                        _st = self.pose33_evaluator.state
                        _sr = _st.dyn_start_range
                        _tr = _st.dyn_target_range
                        _calib = _st.calibrated
                        _neu = _st.neutral_value if _st.neutral_value is not None else '?'
                        _det = tracking_quality.get('details', {}) if tracking_quality else {}
                        _rs = f'{_rv:.1f}' if _rv is not None else '--'
                        _es = f'{_ev:.1f}' if _ev is not None else '--'
                        if not self.pose33_evaluator.auto_calibrate:
                            _ss_c = self.rehab_cfg.get('start_ranges', {}).get(_pa)
                            _ts_c = self.rehab_cfg.get('target_ranges', {}).get(_pa)
                            _srs = f'[{_ss_c[0]:.0f},{_ss_c[1]:.0f}]' if _ss_c else '?'
                            _trs = f'[{_ts_c[0]:.0f},{_ts_c[1]:.0f}]' if _ts_c else '?'
                            _cal_s = f'FIJO ini={_srs} obj={_trs}'
                        else:
                            _sr = _st.dyn_start_range
                            _tr = _st.dyn_target_range
                            _srs = f'[{_sr[0]:.0f},{_sr[1]:.0f}]' if _sr else 'pendiente'
                            _trs = f'[{_tr[0]:.0f},{_tr[1]:.0f}]' if _tr else 'pendiente'
                            _cal_s = f'OK neu={_neu} ini={_srs} obj={_trs}' if _calib else f'ESPERA {len(_st.calib_samples)}/{self.pose33_evaluator.calib_frames}'
                        _lms = ' '.join(
                            f'{k.replace("right_","r").replace("left_","l").replace("_","")}:{v.get("confidence",0):.2f}{"+" if v.get("visible") else "-"}'
                            for k, v in _det.items()
                        )
                        print(
                            f'[DIAG f={frame_counter}] {_pa} raw={_rs} EMA={_es} | '
                            f'fase={eval_result.get("phase","?")} '
                            f'fs={eval_result.get("frames_in_start",0)} '
                            f'ft={eval_result.get("frames_in_target",0)} | '
                            f'in_s={eval_result.get("in_start","?")} '
                            f'in_t={eval_result.get("in_target","?")} '
                            f'safe={eval_result.get("in_safe","?")} | '
                            f'calib={_cal_s} | {_lms}'
                        )
                    frame2 = draw_pose33(frame2, pose33_result.image_landmarks)
                    self.rep_best_similarity = max(self.rep_best_similarity, float(eval_result.get('best_score_current_rep', 0.0)))
                    if eval_result.get('rep_completed'):
                        last_rep = eval_result.get('last_rep') or {}
                        best_score_rep = float(last_rep.get('best_score', eval_result.get('score', 0.0)))
                        rehab_rating = JustDanceModel.get_pose_rating(best_score_rep)
                        self.current_rep = int(eval_result.get('completed_reps', self.current_rep + 1))
                        self.rep_best_similarity = best_score_rep
                        self.rep_best_rating = rehab_rating
                        self.total_points += int(best_score_rep * 10)
                        self._add_rep_result({
                            'rep_idx': self.current_rep,
                            'status': rehab_rating,
                            'similarity': best_score_rep,
                            'duration_s': last_rep.get('duration_s'),
                            'too_fast': bool(last_rep.get('too_fast', False)),
                            'compensations': last_rep.get('compensations', {}),
                            'compensation_penalty': last_rep.get('compensation_penalty', 0.0),
                        })
                        self._record_pose_evaluation(0, int(best_score_rep), rehab_rating)
                        try:
                            self._play_success_chime()
                        except Exception:
                            pass
                        print(f'[POSE33] Repetición {self.current_rep}/{self.repetitions} completada | score={best_score_rep:.1f}')
                        if self.current_rep >= self.repetitions:
                            force_exit_reps = True
                            continue
            except Exception as e:
                print(f'[WARN] Error Pose33 en juego: {e}')

            combined_frame = self._compose_letterbox(frame1, frame2, self.screen_w, self.screen_h)
            combined_frame = JustDanceView.draw_game_hud(
                combined_frame,
                score=self.total_points, combo=combo, multiplier=multiplier,
                sync=int(self.rep_best_similarity), time_left=max(0, int(song_length - video_time)),
                song_name=self.exercise_name,
                difficulty=self.model.difficulty, max_score=max_score,
                player_name=self._player_name, avatar_color_idx=self._avatar_color_idx,
                repetitions=self.repetitions, current_rep=self.current_rep, rep_results=self.rep_results,
                error_state=eval_result.get("error_state", "neutral"),
                error_count=eval_result.get("error_count", 0),
            )
            if not user_body_visible:
                lost_msg = self.rehab_cfg.get('feedback', {}).get('lost_tracking', None)
                combined_frame = JustDanceView.draw_body_warning(combined_frame, subtitle=lost_msg)
            combined_frame = draw_feedback(combined_frame, self.pose33_last_feedback, angles, eval_result, tracking_quality)
            combined_frame = telemetry.draw_overlay(combined_frame)
            if self.metrics is not None:
                combined_frame = self.metrics.draw_overlay(combined_frame)
            rgb_combined = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb_combined.flatten(), (self.screen_w, self.screen_h), 'RGB')
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            if self.metrics is not None:
                self.metrics.record_frame()

        self._infer_stop.set(); self._video_stop.set()
        if getattr(self, 'pose33_backend', None) is not None:
            try:
                import backend_preloader
                if not backend_preloader.is_preloaded_pose33(self.pose33_backend):
                    self.pose33_backend.close()
            except Exception:
                try:
                    self.pose33_backend.close()
                except Exception:
                    pass
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.performance_stats = self._build_performance_stats()
        self._game_duration = song_length
        self._best_combo = best_combo
        return None

    def _cleanup(self):
        self._infer_stop.set(); self._video_stop.set()
        if getattr(self, "pose33_backend", None) is not None:
            try:
                import backend_preloader
                if not backend_preloader.is_preloaded_pose33(self.pose33_backend):
                    self.pose33_backend.close()
            except Exception:
                try: self.pose33_backend.close()
                except Exception: pass
        if self.metrics is not None:
            self.metrics.finish(final_score=self.total_points)
        self.cap1.release()
        if self.cap2: self.cap2.release()
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()

    def release_capture(self):
        self._infer_stop.set(); self._video_stop.set(); self._camera_stop.set()
        self.cap1.release()
        if self.cap2: self.cap2.release()
        if not hasattr(self, "_session_path"):
            self._session_path = None

    def close_windows(self): pass

    def _play_success_chime(self):
        try:
            import array
            import math
            sample_rate = 44100
            duration = 0.3
            n_samples = int(sample_rate * duration)
            buf = array.array('h', [0]*n_samples)
            for i in range(n_samples):
                t = i / sample_rate
                val = int(16383 * math.sin(2 * math.pi * 880 * t) * math.exp(-10 * t))
                buf[i] = val
            sound = pygame.mixer.Sound(buffer=buf)
            sound.play()
        except Exception as e:
            print(f"[WARN] No chime: {e}")



# ── Helper: context manager nulo ─────────────────────────────────────────────

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *args): pass
