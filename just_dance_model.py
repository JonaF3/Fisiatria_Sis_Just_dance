"""Core pose/scoring model for the physical rehabilitation app."""

from pathlib import Path
import os

import os
# Asegurar que CUDA y cuDNN sean encontrados en Windows
_cuda_paths = [
    r"C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
]
for _p in _cuda_paths:
    if os.path.isdir(_p):
        os.add_dll_directory(_p)

import cv2
import numpy as np

from just_dance_score import DifficultyManager


class JustDanceModel:

    RATING_THRESHOLDS = {"PERFECT": 88, "GREAT": 75, "GOOD": 55, "OK": 35}
    COCO_TO_MEDIAPIPE = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    ONNX_INPUT_SIZE_LIGHTNING = 192
    ONNX_INPUT_SIZE_THUNDER   = 256
    ONNX_INPUT_SIZE           = 192  # se sobreescribe en _create_onnx_session
    ONNX_MODEL_CANDIDATES = (
        Path("model/movenet_singlepose_thunder.onnx"),   # Thunder primero
        Path("model/movenet_singlepose_lightning.onnx"),
        Path("model/movenet.onnx"),
        Path("movenet_singlepose_lightning.onnx"),
    )
    POSE_MODEL_CANDIDATES = (
        Path("model/pose_landmarker_lite.task"),
        Path("model/pose_landmarker_full.task"),
        Path("model/pose_landmarker_heavy.task"),
        Path("pose_landmarker_lite.task"),
        Path("pose_landmarker_full.task"),
        Path("pose_landmarker_heavy.task"),
    )

    NEUTRAL_ANGLES = {
        "left_arm":    170.0,
        "right_arm":   170.0,
        "left_elbow":  170.0,
        "right_elbow": 170.0,
        "left_trunk":  175.0,
        "right_trunk": 175.0,
        "left_knee":   175.0,
        "right_knee":  175.0,
        "left_ankle":  175.0,
        "right_ankle": 175.0,
    }

    def __init__(self, model_path=None, difficulty="NORMAL", use_gpu: bool = False,
                 num_poses: int = 1):
        self.model_path   = model_path
        self.use_gpu      = use_gpu
        self.num_poses    = max(1, int(num_poses or 1))
        self.difficulty_manager = DifficultyManager(difficulty)
        self._sync_difficulty()

        self._timestamp_ms = 0
        self._onnx_input_name = None
        self._onnx_output_name = None
        self._mp = None
        self._pose_api = self._select_pose_api(model_path)
        self.pose_detector = self._create_pose_detector(model_path)
        print(f"[INFO] Pose backend: {self._pose_api.upper()} | Condicion: {self._actual_condition}")

    def _select_pose_api(self, model_path):
        suffix = Path(model_path).suffix.lower() if model_path else ""
        if suffix == ".task":
            return "tasks"
        if self._resolve_onnx_model_path(model_path, required=False):
            return "onnx"
        mp = self._get_mediapipe()
        return "solutions" if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose") else "tasks"

    def _get_mediapipe(self):
        if self._mp is None:
            import mediapipe as mp
            self._mp = mp
        return self._mp

    def _create_pose_detector(self, model_path):
        if self._pose_api == "onnx":
            return self._create_onnx_session(model_path)

        mp = self._get_mediapipe()
        if self._pose_api == "solutions":
            self._actual_condition = "CPU"
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        pose_model_path = self._resolve_pose_model_path(model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._actual_condition = "CPU"
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _create_onnx_session(self, model_path):

        onnx_model_path = self._resolve_onnx_model_path(model_path, required=True)
        print(f"[DEBUG] Modelo cargado: {onnx_model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime no esta instalado. Instala onnxruntime-directml "
                "para usar CPU/GPU con DirectML."
            ) from exc

        onnx_model_path = self._resolve_onnx_model_path(model_path, required=True)
        available = ort.get_available_providers()
        session_options = ort.SessionOptions()

        if self.use_gpu:
            if "CUDAExecutionProvider" in available:
                device_id = int(os.environ.get("JUST_DANCE_CUDA_DEVICE_ID", "0"))
                providers = [("CUDAExecutionProvider", {"device_id": device_id})]
                expected_provider = "CUDAExecutionProvider"
            else:
                print("[WARN] CUDA no disponible, cayendo a CPU.")
                self.use_gpu = False
                providers = ["CPUExecutionProvider"]
                expected_provider = "CPUExecutionProvider"
        else:
            if "CPUExecutionProvider" not in available:
                raise RuntimeError(
                    "CPUExecutionProvider no esta disponible en ONNX Runtime."
                )
            providers = ["CPUExecutionProvider"]
            expected_provider = "CPUExecutionProvider"

        session = ort.InferenceSession(
            onnx_model_path,
            sess_options=session_options,
            providers=providers,
        )
        active = session.get_providers()
        if not active or active[0] != expected_provider:
            print(f"[WARN] Provider esperado: {expected_provider}, activos: {active}")

        self._onnx_input_name = session.get_inputs()[0].name
        self._onnx_output_name = session.get_outputs()[0].name
        self._actual_condition = "GPU" if self.use_gpu else "CPU"

        # Ajustar input size según el modelo cargado
        if "thunder" in onnx_model_path.lower():
            self.ONNX_INPUT_SIZE = self.ONNX_INPUT_SIZE_THUNDER
            print(f"[INFO] Modelo: Thunder — input {self.ONNX_INPUT_SIZE}x{self.ONNX_INPUT_SIZE}")
        else:
            self.ONNX_INPUT_SIZE = self.ONNX_INPUT_SIZE_LIGHTNING
            print(f"[INFO] Modelo: Lightning — input {self.ONNX_INPUT_SIZE}x{self.ONNX_INPUT_SIZE}")

        return session


    def _resolve_onnx_model_path(self, model_path, required=True):
        candidates = []
        if model_path and Path(model_path).suffix.lower() == ".onnx":
            candidates.append(Path(model_path))
        candidates.extend(self.ONNX_MODEL_CANDIDATES)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        if required:
            raise FileNotFoundError(
                "ONNX Runtime requiere un modelo .onnx. Coloca "
                "movenet_singlepose_lightning.onnx en la carpeta model."
            )
        return None

    def _resolve_pose_model_path(self, model_path):
        candidates = []
        if model_path and Path(model_path).suffix.lower() == ".task":
            candidates.append(Path(model_path))
        candidates.extend(self.POSE_MODEL_CANDIDATES)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "MediaPipe PoseLandmarker requires a .task model bundle. "
            "Place pose_landmarker_lite.task in the model directory."
        )

    # ── Dificultad ────────────────────────────────────────────────────────────

    def _sync_difficulty(self):
        self.difficulty               = self.difficulty_manager.difficulty
        self.angle_threshold          = self.difficulty_manager.angle_threshold
        self.score_multiplier         = self.difficulty_manager.score_multiplier
        self.max_score                = self.difficulty_manager.max_score
        self.movement_threshold       = self.difficulty_manager.movement_threshold
        self.distinctiveness_threshold= self.difficulty_manager.distinctiveness_threshold
        self.vector_weight            = getattr(self.difficulty_manager, "vector_weight", 0.55)
        self.angle_weight             = getattr(self.difficulty_manager, "angle_weight",  0.45)

    def set_difficulty(self, difficulty: str):
        self.difficulty_manager.set_difficulty(difficulty)
        self._sync_difficulty()

    def get_rating_points(self, rating: str) -> int:
        return self.difficulty_manager.get_rating_points(rating)

    # ── Inferencia ────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_keypoints():
        return np.zeros((1, 1, 17, 3), dtype=np.float32)

    def _landmarks_to_keypoints(self, landmarks):
        keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
        if landmarks:
            for coco_idx, mp_idx in enumerate(self.COCO_TO_MEDIAPIPE):
                lm = landmarks[mp_idx]
                visibility = getattr(lm, "visibility", getattr(lm, "presence", 0.0))
                keypoints[0, 0, coco_idx] = [lm.y, lm.x, visibility]
        return keypoints

    def run_inference(self, input_image):
        poses = self.run_multi_inference(input_image, max_poses=1)
        return poses[0] if poses else self._empty_keypoints()

    def run_multi_inference(self, input_image, max_poses=None):
        max_poses = max(1, int(max_poses or self.num_poses))

        if input_image is None or input_image.size == 0:
            return []

        input_image = np.ascontiguousarray(input_image, dtype=np.uint8)

        if self._pose_api == "onnx":
            img = cv2.resize(input_image, (self.ONNX_INPUT_SIZE, self.ONNX_INPUT_SIZE))
            img = img[np.newaxis, ...].astype(np.int32)
            result = self.pose_detector.run(
                [self._onnx_output_name],
                {self._onnx_input_name: img},
            )[0].astype(np.float32)
            return [result]

        if self._pose_api == "solutions":
            results = self.pose_detector.process(input_image)
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            return [self._landmarks_to_keypoints(landmarks)] if landmarks else []
        else:
            mp = self._get_mediapipe()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
            self._timestamp_ms += 33
            results = self.pose_detector.detect_for_video(mp_image, self._timestamp_ms)
            landmarks_list = list(results.pose_landmarks or [])[:max_poses]
            return [self._landmarks_to_keypoints(landmarks) for landmarks in landmarks_list]

    def close(self):
        if self._pose_api == "onnx":
            return
        if hasattr(self.pose_detector, "close"):
            self.pose_detector.close()

    # ── Ángulos ───────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_angle(frame, key_points, start_index, middle_index, end_index):
        y_coordinate, x_coordinate, _ = frame.shape
        shaped = np.squeeze(
            np.multiply(key_points, [y_coordinate, x_coordinate, 1])
        )
        joint_start  = np.array([int(shaped[start_index][0]),  int(shaped[start_index][1])])
        joint_middle = np.array([int(shaped[middle_index][0]), int(shaped[middle_index][1])])
        joint_end    = np.array([int(shaped[end_index][0]),    int(shaped[end_index][1])])

        radians = np.arctan2(
            joint_end[1]   - joint_middle[1], joint_end[0]   - joint_middle[0]
        ) - np.arctan2(
            joint_start[1] - joint_middle[1], joint_start[0] - joint_middle[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    @staticmethod
    def store_angles(all_joint_angles, frame, key_points):
        all_joint_angles["left_arm"].append(
            JustDanceModel.calculate_angle(frame, key_points, 5, 7, 9))
        all_joint_angles["right_arm"].append(
            JustDanceModel.calculate_angle(frame, key_points, 6, 8, 10))
        all_joint_angles["left_elbow"].append(
            JustDanceModel.calculate_angle(frame, key_points, 7, 5, 11))
        all_joint_angles["right_elbow"].append(
            JustDanceModel.calculate_angle(frame, key_points, 8, 6, 12))
        all_joint_angles["left_trunk"].append(
            JustDanceModel.calculate_angle(frame, key_points, 5, 11, 13))
        all_joint_angles["right_trunk"].append(
            JustDanceModel.calculate_angle(frame, key_points, 6, 12, 14))
        all_joint_angles["left_knee"].append(
            JustDanceModel.calculate_angle(frame, key_points, 12, 11, 13))
        all_joint_angles["right_knee"].append(
            JustDanceModel.calculate_angle(frame, key_points, 11, 12, 14))
        all_joint_angles["left_ankle"].append(
            JustDanceModel.calculate_angle(frame, key_points, 11, 13, 15))
        all_joint_angles["right_ankle"].append(
            JustDanceModel.calculate_angle(frame, key_points, 12, 14, 16))

    # ── Vectores de articulación (cosine similarity) ──────────────────────────

    @staticmethod
    def _extract_joint_vectors(frame, key_points, confidence_threshold=0.15):
        h, w, _ = frame.shape
        shaped   = np.squeeze(np.multiply(key_points, [h, w, 1]))

        def get_point(idx):
            y, x, conf = shaped[idx]
            return (np.array([x, y]), conf)

        PAIRS = [
            ("left_upper_arm",   5,  7),
            ("right_upper_arm",  6,  8),
            ("left_forearm",     7,  9),
            ("right_forearm",    8, 10),
            ("left_torso",       5, 11),
            ("right_torso",      6, 12),
            ("left_trunk",      11, 13),
            ("right_trunk",     12, 14),
            ("left_thigh",      11, 13),
            ("right_thigh",     12, 14),
            ("left_shin",       13, 15),
            ("right_shin",      14, 16),
        ]

        vectors = {}
        for name, i, j in PAIRS:
            pt_i, conf_i = get_point(i)
            pt_j, conf_j = get_point(j)
            if conf_i < confidence_threshold or conf_j < confidence_threshold:
                continue
            raw = pt_j - pt_i
            norm = np.linalg.norm(raw)
            if norm < 1e-6:
                continue
            vectors[name] = raw / norm
        return vectors

    @staticmethod
    def cosine_similarity_score(vectors_target, vectors_user):
        WEIGHTS = {
            "left_upper_arm":  2.0, "right_upper_arm": 2.0,
            "left_forearm":    1.5, "right_forearm":   1.5,
            "left_torso":      1.0, "right_torso":     1.0,
            "left_trunk":      1.4, "right_trunk":     1.4,
            "left_thigh":      1.2, "right_thigh":     1.2,
            "left_shin":       0.8, "right_shin":      0.8,
        }
        weighted_sum = total_weight = 0.0
        for name in vectors_target:
            if name not in vectors_user:
                continue
            cos_sim = float(np.dot(vectors_target[name], vectors_user[name]))
            score   = (cos_sim + 1.0) / 2.0 * 100.0
            w = WEIGHTS.get(name, 1.0)
            weighted_sum += score * w
            total_weight += w
        if total_weight < 1e-6:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def combined_pose_score(angle_score, vector_score, angle_weight=0.45, vector_weight=0.55):
        combined = angle_score * angle_weight + vector_score * vector_weight
        return int(np.clip(combined, 0, 100))

    # ── Score y rating ────────────────────────────────────────────────────────

    @staticmethod
    def score_calculator(angle_video, angle_camera, threshold):
        video_array  = np.array(angle_video,  dtype=float)
        camera_array = np.array(angle_camera, dtype=float)
        differences  = np.abs(video_array - camera_array)
        soft_limit   = threshold * 1.5
        raw_scores   = np.maximum(0.0, 1.0 - (differences / soft_limit) ** 2)
        return int(np.mean(raw_scores) * 100)

    @staticmethod
    def score_all_joints(angles_video, angles_camera, threshold, window=30):
        joints = [
            "left_arm", "right_arm", "left_elbow", "right_elbow",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        scores = []
        for joint in joints:
            v = angles_video[joint][-window:]
            c = angles_camera[joint][-window:]
            if len(v) > 0 and len(c) > 0:
                min_len = min(len(v), len(c))
                scores.append(
                    JustDanceModel.score_calculator(
                        v[-min_len:], c[-min_len:], threshold
                    )
                )
        return int(np.mean(scores)) if scores else 0

    @staticmethod
    def get_pose_rating(score):
        if score >= 85: return "PERFECT"
        if score >= 70: return "GREAT"
        if score >= 50: return "GOOD"
        if score >= 30: return "OK"
        return "MISS"

    # ── Métricas de movimiento ────────────────────────────────────────────────

    @staticmethod
    def pose_distance_from_neutral(pose_angles):
        distances = []
        for joint, neutral_val in JustDanceModel.NEUTRAL_ANGLES.items():
            if joint in pose_angles:
                distances.append(abs(pose_angles[joint] - neutral_val))
        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def user_movement_variance(angles_camera, window=20):
        variances = []
        joints = [
            "left_arm", "right_arm", "left_elbow", "right_elbow",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        for joint in joints:
            recent = angles_camera.get(joint, [])[-window:]
            if len(recent) >= 3:
                variances.append(float(np.std(recent)))
        return float(np.mean(variances)) if variances else 0.0

    @staticmethod
    def movement_trajectory_score(angles_camera, window=15):
        joints = ["left_arm", "right_arm", "left_elbow", "right_elbow"]
        trajectory_scores = []
        for joint in joints:
            recent = angles_camera.get(joint, [])[-window:]
            if len(recent) < 5:
                continue
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            direction_changes = sum(
                1 for i in range(len(diffs)-1)
                if diffs[i] * diffs[i+1] < 0
            )
            max_swing = max(recent) - min(recent)
            trajectory_scores.append(
                1.0 if (direction_changes >= 2 and max_swing > 10) else 0.0
            )
        return float(np.mean(trajectory_scores)) if trajectory_scores else 0.0

    # ── Score ajustado ────────────────────────────────────────────────────────

    @staticmethod
    def adjusted_pose_score(
        raw_score,
        pose_angles_target,
        angles_camera,
        frame_video=None,
        kp_video=None,
        frame_camera=None,
        kp_camera=None,
        movement_threshold=8.0,
        distinctiveness_threshold=12.0,
        angle_weight=0.45,
        vector_weight=0.55,
    ):
        user_movement = JustDanceModel.user_movement_variance(angles_camera)
        pose_exigence = JustDanceModel.pose_distance_from_neutral(pose_angles_target)

        vector_score = raw_score
        if frame_video is not None and kp_video is not None \
                and frame_camera is not None and kp_camera is not None:
            vecs_target = JustDanceModel._extract_joint_vectors(frame_video,  kp_video)
            vecs_user   = JustDanceModel._extract_joint_vectors(frame_camera, kp_camera)
            if vecs_target and vecs_user:
                vector_score = JustDanceModel.cosine_similarity_score(vecs_target, vecs_user)

        combined = JustDanceModel.combined_pose_score(
            raw_score, vector_score,
            angle_weight=angle_weight,
            vector_weight=vector_weight,
        )

        if pose_exigence > distinctiveness_threshold:
            if user_movement < movement_threshold:
                penalty = 1.0 - (user_movement / movement_threshold)
                penalty_factor = 0.85 if angle_weight <= 0.30 else 0.75
                combined = int(max(0, combined * (1.0 - penalty * penalty_factor)))
            trajectory = JustDanceModel.movement_trajectory_score(angles_camera)
            traj_threshold = 0.4 if angle_weight <= 0.30 else 0.3
            if trajectory < traj_threshold:
                combined = int(combined * (0.55 if angle_weight <= 0.30 else 0.65))

        return int(np.clip(combined, 0, 100))

    @staticmethod
    def final_score(all_angles_video, all_angles_camera, threshold):
        all_scores = []
        joints = [
            "left_arm", "right_arm", "left_elbow", "right_elbow",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        for joint in joints:
            all_scores.append(
                JustDanceModel.score_calculator(
                    all_angles_video[joint],
                    all_angles_camera[joint],
                    threshold,
                )
            )
        final = np.mean(all_scores) + 20
        return min(final, 100)
