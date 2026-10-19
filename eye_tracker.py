"""
Focus Tracker - Multimodal Data Collection for Focus Research
Author: Research Team
Version: 6.1 (Scientific Review Build - Includes Calibration Debug UI)

Features: 
- EAR (Eye Aspect Ratio) tracking
- Gaze ML Calibration (25-point with quality filtering & median)
- Adaptive EMA (Exponential Moving Average) Smoothing
- Audio RMS (Root Mean Square) volume tracking
- Head Pose estimation with Confidence Filtering
- 11 ML Features extraction
- I-VT/I-DT hybrid algorithm for fixation/saccade classification
- Camera intrinsics support
"""

import csv
import time
import logging
import math
import os
from typing import Tuple, Optional, List, Dict
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# --- CONFIGURATION ---
CONFIG = {
    "CSV_FILENAME": "focus_data_log.csv",
    "WEBCAM_ID": 0,
    "AUDIO_SAMPLERATE": 44100,
    "EAR_THRESHOLD": 0.22,
    "SCREEN_WIDTH": 1280,   
    "SCREEN_HEIGHT": 800,   
    
    # 5x5 Grid = 25 points for robust SVR interpolation
    "CALIBRATION_GRID": [
        (x, y) for x in [0.1, 0.3, 0.5, 0.7, 0.9] 
               for y in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    "CALIBRATION_FRAMES_PER_POINT": 20,

    # Gaze & Cognitive Tracker Settings
    "FIXATION_MAX_VELOCITY": 800,       # Maximum pixels per second (I-VT algorithm)
    "FIXATION_MAX_DISPERSION": 35,      # Maximum spatial dispersion in pixels (I-DT algorithm)
    
    # Quality thresholds for calibration frames (adjust if necessary)
    "CALIB_MAX_POSE_ERROR": 6.0,
    "CALIB_Z_DIST_MIN": 450,
    "CALIB_Z_DIST_MAX": 850
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioTracker:
    """
    Captures and processes audio data to track ambient volume levels using RMS.
    """
    def __init__(self, samplerate: int = 44100):
        """
        Initializes the AudioTracker.

        Args:
            samplerate (int): The audio sampling rate in Hz.
        """
        self.samplerate = samplerate
        self.current_volume = 0.0
        self.stream: Optional[sd.InputStream] = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Internal callback function for the sounddevice input stream to calculate RMS volume."""
        if status:
            logging.warning(f"Audio status: {status}")
        self.current_volume = float(np.sqrt(np.mean(indata**2)))

    def start(self) -> None:
        """Starts the audio input stream."""
        try:
            self.stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.samplerate)
            self.stream.start()
        except Exception as e:
            logging.error(f"Fout bij starten audio: {e}")

    def get_volume(self) -> float:
        """
        Retrieves the current volume level.

        Returns:
            float: The latest calculated RMS volume.
        """
        return self.current_volume

    def stop(self) -> None:
        """Stops and closes the audio input stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()

class FaceTracker:
    """
    Extracts facial landmarks, calculates Eye Aspect Ratio (EAR), estimates head pose, 
    and generates machine learning features for gaze tracking using MediaPipe.
    """
    # MediaPipe landmark indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    L_EYE_INNER = 362
    L_EYE_OUTER = 263
    L_EYE_TOP = 386
    L_EYE_BOTTOM = 374
    
    R_EYE_INNER = 133
    R_EYE_OUTER = 33
    R_EYE_TOP = 159
    R_EYE_BOTTOM = 145

    LEFT_IRIS = 468
    RIGHT_IRIS = 473
    BROW_INNER_LEFT = 107
    BROW_INNER_RIGHT = 336
    FACE_LEFT_EDGE = 234
    FACE_RIGHT_EDGE = 454
    
    # 3D model points for head pose estimation (SolvePnP)
    FACE_3D = np.array([
        (0.0, 0.0, 0.0),            # Nose tip (1)
        (0.0, -330.0, -65.0),       # Chin (152)
        (-225.0, 170.0, -135.0),    # Left eye outer corner (33)
        (225.0, 170.0, -135.0),     # Right eye outer corner (263)
        (-150.0, -150.0, -125.0),   # Mouth left corner (61)
        (150.0, -150.0, -125.0)     # Mouth right corner (291)
    ], dtype=np.float64)
    
    POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

    def __init__(self):
        """Initializes the FaceMesh model and loads camera intrinsics if available."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        
        self.cam_matrix = None
        self.dist_matrix = None
        
        # Load camera calibration parameters if present
        if os.path.exists("camera_intrinsics.yml"):
            try:
                fs = cv2.FileStorage("camera_intrinsics.yml", cv2.FILE_STORAGE_READ)
                self.cam_matrix = fs.getNode("camera_matrix").mat()
                self.dist_matrix = fs.getNode("dist_coeffs").mat()
                fs.release()
                logging.info("Camera intrinsics geladen.")
            except Exception as e:
                logging.warning(f"Fout bij laden camera_intrinsics.yml: {e}")

    def _calculate_ear(self, eye_landmarks: list, landmarks: list, img_w: int, img_h: int) -> float:
        """Calculates the Eye Aspect Ratio (EAR) to detect blinks."""
        pts = np.array([(landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in eye_landmarks])
        h = np.linalg.norm(pts[0] - pts[3])
        if h == 0: return 0.0
        return float((np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / (2.0 * h))

    def _get_head_pose(self, landmarks, img_w, img_h) -> Tuple[float, float, float, float, float]:
        """Estimates 3D head pose (pitch, yaw, roll) and distance (z) using cv2.solvePnP."""
        face_2d = np.array([
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h) 
            for idx in self.POSE_LANDMARKS
        ], dtype=np.float64)

        if self.cam_matrix is None:
            focal_length = img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]], dtype=np.float64)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
        else:
            cam_matrix, dist_matrix = self.cam_matrix, self.dist_matrix

        success, rot_vec, trans_vec = cv2.solvePnP(self.FACE_3D, face_2d, cam_matrix, dist_matrix)
        
        if not success: 
            return 0.0, 0.0, 0.0, 0.0, 999.0

        projected, _ = cv2.projectPoints(self.FACE_3D, rot_vec, trans_vec, cam_matrix, dist_matrix)
        error = np.mean(np.linalg.norm(projected.reshape(-1, 2) - face_2d, axis=1))

        if error > 6:
            return 0.0, 0.0, 0.0, 0.0, float(error)

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        z_distance = trans_vec[2][0]
        
        return pitch, yaw, roll, z_distance, float(error)

    def _get_relative_iris_pos(self, iris_idx: int, inner_idx: int, outer_idx: int, 
                               top_idx: int, bottom_idx: int, landmarks: list, img_w: int, img_h: int) -> Tuple[float, float]:
        """Calculates normalized relative coordinates of the iris within the eye bounding box."""
        inner_x = landmarks[inner_idx].x * img_w
        outer_x = landmarks[outer_idx].x * img_w
        top_y = landmarks[top_idx].y * img_h
        bottom_y = landmarks[bottom_idx].y * img_h
        
        iris_x = landmarks[iris_idx].x * img_w
        iris_y = landmarks[iris_idx].y * img_h
        
        min_x, max_x = min(inner_x, outer_x), max(inner_x, outer_x)
        min_y, max_y = min(top_y, bottom_y), max(top_y, bottom_y)
        
        width = max(max_x - min_x, 1)
        height = max(max_y - min_y, 1)
        
        rel_x = (iris_x - min_x) / width
        rel_y = (iris_y - min_y) / height
        
        return rel_x, rel_y

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Processes a single video frame to extract all facial tracking metrics.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.

        Returns:
            dict: Extracted metrics including head pose, EAR, ML features, and coordinates.
        """
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        data = {
            "face_detected": False, "ear_avg": 0.0, "blink": 0, 
            "l_iris": (0,0), "r_iris": (0,0), "ml_features": [],
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "z_dist": 0.0, "pose_error": 999.0,
            "face_width": 0.0, "brow_contraction_index": 0.0, "convergence": 0.0,
            "inter_eye": 0.0
        }

        if not results.multi_face_landmarks: return data
        data["face_detected"] = True
        landmarks = results.multi_face_landmarks[0].landmark

        ear_l = self._calculate_ear(self.LEFT_EYE, landmarks, img_w, img_h)
        ear_r = self._calculate_ear(self.RIGHT_EYE, landmarks, img_w, img_h)
        data["ear_avg"] = (ear_l + ear_r) / 2.0
        if data["ear_avg"] < CONFIG["EAR_THRESHOLD"]: data["blink"] = 1

        l_x, l_y = landmarks[self.LEFT_IRIS].x * img_w, landmarks[self.LEFT_IRIS].y * img_h
        r_x, r_y = landmarks[self.RIGHT_IRIS].x * img_w, landmarks[self.RIGHT_IRIS].y * img_h
        data["l_iris"] = (int(l_x), int(l_y))
        data["r_iris"] = (int(r_x), int(r_y))

        data["pitch"], data["yaw"], data["roll"], data["z_dist"], data["pose_error"] = self._get_head_pose(landmarks, img_w, img_h)

        l_rel_x, l_rel_y = self._get_relative_iris_pos(
            self.LEFT_IRIS, self.L_EYE_INNER, self.L_EYE_OUTER, self.L_EYE_TOP, self.L_EYE_BOTTOM, landmarks, img_w, img_h)
        r_rel_x, r_rel_y = self._get_relative_iris_pos(
            self.RIGHT_IRIS, self.R_EYE_INNER, self.R_EYE_OUTER, self.R_EYE_TOP, self.R_EYE_BOTTOM, landmarks, img_w, img_h)

        left_edge = np.array([landmarks[self.FACE_LEFT_EDGE].x * img_w, landmarks[self.FACE_LEFT_EDGE].y * img_h])
        right_edge = np.array([landmarks[self.FACE_RIGHT_EDGE].x * img_w, landmarks[self.FACE_RIGHT_EDGE].y * img_h])
        data["face_width"] = float(np.linalg.norm(left_edge - right_edge))

        data["inter_eye"] = float(np.linalg.norm(np.array([l_x, l_y]) - np.array([r_x, r_y])))

        # Pack features for machine learning calibration
        data["ml_features"] = [
            l_rel_x, l_rel_y, r_rel_x, r_rel_y,
            data["pitch"], data["yaw"], data["roll"], data["z_dist"],
            data["ear_avg"], data["inter_eye"], data["face_width"]
        ]

        brow_l = np.array([landmarks[self.BROW_INNER_LEFT].x * img_w, landmarks[self.BROW_INNER_LEFT].y * img_h])
        brow_r = np.array([landmarks[self.BROW_INNER_RIGHT].x * img_w, landmarks[self.BROW_INNER_RIGHT].y * img_h])
        data["brow_contraction_index"] = float(np.linalg.norm(brow_l - brow_r) / data["face_width"])
        
        iris_l_arr = np.array([l_x, l_y])
        iris_r_arr = np.array([r_x, r_y])
        data["convergence"] = float(np.linalg.norm(iris_l_arr - iris_r_arr) / data["face_width"])

        return data

class CognitiveAnalyzer:
    """
    Analyzes eye movement patterns using a hybrid I-VT (Velocity-Threshold) 
    and I-DT (Dispersion-Threshold) algorithm to classify gaze as Fixation or Saccade.
    """
    def __init__(self):
        self.history = deque(maxlen=15)
        self.velocity_threshold = CONFIG["FIXATION_MAX_VELOCITY"]
        self.dispersion_threshold = CONFIG["FIXATION_MAX_DISPERSION"]

    def update(self, x: int, y: int, t: float) -> Tuple[str, float, float]:
        """
        Updates the tracker history and classifies current gaze state.

        Args:
            x (int): Current predicted X gaze coordinate.
            y (int): Current predicted Y gaze coordinate.
            t (float): Current timestamp.

        Returns:
            Tuple[str, float, float]: State ("Fixatie", "Saccade", or "Onbekend"), 
                                      calculated velocity, and calculated dispersion.
        """
        self.history.append((x, y, t))

        if len(self.history) < 3:
            return "Onbekend", 0.0, 0.0

        x1, y1, t1 = self.history[-2]
        x2, y2, t2 = self.history[-1]

        dt = max(t2 - t1, 1e-6)
        velocity = math.hypot(x2 - x1, y2 - y1) / dt

        xs = [p[0] for p in self.history]
        ys = [p[1] for p in self.history]
        dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))

        if velocity < self.velocity_threshold and dispersion < self.dispersion_threshold:
            return "Fixatie", velocity, dispersion

        return "Saccade", velocity, dispersion

class GazeCalibrator:
    """
    Handles the on-screen calibration process and predicts gaze coordinates
    using trained Support Vector Regression (SVR) models.
    """
    def __init__(self):
        self.model_x = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
        self.model_y = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
        self.is_calibrated = False
        self.ema_x = None
        self.ema_y = None

    def run_calibration(self, cap: cv2.VideoCapture, tracker: FaceTracker) -> None:
        """
        Executes the calibration loop, displaying targets and recording user gaze.

        Args:
            cap (cv2.VideoCapture): Open CV2 video capture instance.
            tracker (FaceTracker): Instantiated FaceTracker for frame analysis.
        """
        logging.info("Start kalibratieprocedure...")
        window_name = "Kalibratie - Kijk naar de rode stip en druk op SPATIE"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window_name, 0, 0)

        features_list, targets_x, targets_y = [], [], []
        frames_to_collect = CONFIG["CALIBRATION_FRAMES_PER_POINT"]

        for point in CONFIG["CALIBRATION_GRID"]:
            target_x = int(point[0] * CONFIG["SCREEN_WIDTH"])
            target_y = int(point[1] * CONFIG["SCREEN_HEIGHT"])
            
            point_collected = False
            is_recording = False
            collected_frames = 0
            features_buffer = [] 
            
            while not point_collected:
                success, frame = cap.read()
                if not success: continue
                frame = cv2.flip(frame, 1)
                
                calib_screen = np.zeros((CONFIG["SCREEN_HEIGHT"], CONFIG["SCREEN_WIDTH"], 3), dtype=np.uint8)
                
                if not is_recording:
                    cv2.circle(calib_screen, (target_x, target_y), 15, (0, 0, 255), -1)
                    cv2.putText(calib_screen, "Kijk naar de stip en druk op SPATIE", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.circle(calib_screen, (target_x, target_y), 15, (0, 255, 0), -1)
                    cv2.circle(calib_screen, (target_x, target_y), 15 + (collected_frames * 2), (0, 255, 0), 2)
                    
                    face_data = tracker.process_frame(frame)
                    
                    if face_data["face_detected"]:
                        pose_ok = (abs(face_data["yaw"]) < 12 and 
                                   abs(face_data["pitch"]) < 10 and 
                                   abs(face_data["roll"]) < 8)
                        ear_ok = face_data["blink"] == 0
                        
                        z_min = CONFIG["CALIB_Z_DIST_MIN"]
                        z_max = CONFIG["CALIB_Z_DIST_MAX"]
                        distance_ok = z_min < face_data["z_dist"] < z_max
                        
                        max_err = CONFIG["CALIB_MAX_POSE_ERROR"]
                        error_ok = face_data["pose_error"] < max_err
                        
                        # --- DEBUG INFORMATION ON SCREEN ---
                        cv2.putText(calib_screen, f"Frames verzameld: {collected_frames}/{frames_to_collect}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(calib_screen, f"Pose OK (<12,10,8): {pose_ok} (P:{face_data['pitch']:.0f} Y:{face_data['yaw']:.0f} R:{face_data['roll']:.0f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if pose_ok else (0, 0, 255), 1)
                        cv2.putText(calib_screen, f"Afstand OK ({z_min}-{z_max}): {distance_ok} (Z: {face_data['z_dist']:.0f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if distance_ok else (0, 0, 255), 1)
                        cv2.putText(calib_screen, f"Ogen Open (EAR>0.22): {ear_ok}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ear_ok else (0, 0, 255), 1)
                        cv2.putText(calib_screen, f"Error OK (<{max_err}): {error_ok} (Err: {face_data['pose_error']:.1f})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if error_ok else (0, 0, 255), 1)
                        # ----------------------------------
                        
                        if pose_ok and ear_ok and distance_ok and error_ok:
                            features_buffer.append(face_data["ml_features"])
                            collected_frames += 1
                        
                    if collected_frames >= frames_to_collect:
                        median_features = np.median(np.array(features_buffer), axis=0).tolist()
                        features_list.append(median_features)
                        targets_x.append(target_x)
                        targets_y.append(target_y)
                        
                        point_collected = True
                        cv2.imshow(window_name, calib_screen)
                        cv2.waitKey(200)
                
                cv2.imshow(window_name, calib_screen)
                if cv2.waitKey(1) & 0xFF == 32 and not is_recording:
                    is_recording = True

        cv2.destroyWindow(window_name)
        logging.info("Kalibratie voltooid. ML modellen trainen...")
        self.model_x.fit(features_list, targets_x)
        self.model_y.fit(features_list, targets_y)
        self.is_calibrated = True

    def predict_gaze(self, features: list) -> Tuple[int, int]:
        """
        Predicts X and Y screen coordinates from facial features using SVR,
        smoothed via adaptive EMA (Exponential Moving Average).

        Args:
            features (list): 11-dimensional feature list from FaceTracker.

        Returns:
            Tuple[int, int]: Estimated (X, Y) coordinates on the screen.
        """
        if not self.is_calibrated: return 0, 0
        
        pred_x = float(self.model_x.predict([features])[0])
        pred_y = float(self.model_y.predict([features])[0])
        
        if self.ema_x is None:
            self.ema_x, self.ema_y = pred_x, pred_y
        else:
            dx = abs(pred_x - self.ema_x)
            dy = abs(pred_y - self.ema_y)
            movement = math.hypot(dx, dy)

            # Adaptive EMA smoothing dynamically adjusts filtering based on movement speed
            if movement > 120:
                alpha = 0.65      
            elif movement > 40:
                alpha = 0.40
            else:
                alpha = 0.18      
                
            self.ema_x = alpha * pred_x + (1 - alpha) * self.ema_x
            self.ema_y = alpha * pred_y + (1 - alpha) * self.ema_y
            
        return max(0, min(CONFIG["SCREEN_WIDTH"], int(self.ema_x))), max(0, min(CONFIG["SCREEN_HEIGHT"], int(self.ema_y)))

class DataLogger:
    """
    Manages continuous logging of multimodal tracking data to a CSV file.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        self.writer = None
        
    def start(self) -> None:
        """Opens the CSV file and writes the header row."""
        self.file = open(self.filename, mode="w", newline="")
        self.writer = csv.writer(self.file)
        
        self.writer.writerow([
            "Timestamp", "Blink_Detected", "EAR_Avg", "Screen_X", "Screen_Y", 
            "Audio_Volume_RMS", "Pitch", "Yaw", "Roll", "Z_Distance", "Pose_Error",
            "Face_Width", "Brow_Contraction_Index", "Convergence", 
            "Eye_State", "Velocity_px_s", "Dispersion_px"
        ])
        
    def log_row(self, row_data: list) -> None:
        """Writes a single row of tracked data to the CSV."""
        if self.writer: self.writer.writerow(row_data)
            
    def stop(self) -> None:
        """Safely closes the active CSV file."""
        if self.file: self.file.close()

def main():
    """Main execution loop initializing trackers, UI, and data collection process."""
    audio = AudioTracker(samplerate=CONFIG["AUDIO_SAMPLERATE"])
    tracker = FaceTracker()
    calibrator = GazeCalibrator()
    cog_analyzer = CognitiveAnalyzer()
    logger = DataLogger(CONFIG["CSV_FILENAME"])
    
    cap = cv2.VideoCapture(CONFIG["WEBCAM_ID"])
    if not cap.isOpened():
        logging.error("Kan geen verbinding maken met de webcam.")
        return

    try:
        calibrator.run_calibration(cap, tracker)
        audio.start()
        logger.start()
        
        cv2.namedWindow("Cognitive Focus Tracker", cv2.WINDOW_NORMAL)
        
        while True:
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            face_data = tracker.process_frame(frame)
            vol = audio.get_volume()
            current_t = time.time()
            
            if face_data["face_detected"]:
                screen_x, screen_y = calibrator.predict_gaze(face_data["ml_features"])
                eye_state, velocity, dispersion = cog_analyzer.update(screen_x, screen_y, current_t)
                
                logger.log_row([
                    current_t, face_data["blink"], round(face_data["ear_avg"], 4), 
                    screen_x, screen_y, round(vol, 6), 
                    round(face_data["pitch"], 2), round(face_data["yaw"], 2), 
                    round(face_data["roll"], 2), round(face_data["z_dist"], 2),
                    round(face_data["pose_error"], 2),
                    round(face_data["face_width"], 2), round(face_data["brow_contraction_index"], 4), 
                    round(face_data["convergence"], 4), 
                    eye_state, round(velocity, 2), round(dispersion, 2)
                ])

                cv2.circle(frame, face_data["l_iris"], 4, (0, 255, 0), -1)
                cv2.circle(frame, face_data["r_iris"], 4, (0, 255, 0), -1)
                
                hud_color = (0, 255, 255) if eye_state == "Fixatie" else (255, 100, 100)
                cv2.putText(frame, f"Staat: {eye_state} | V:{velocity:.0f} D:{dispersion:.0f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
                cv2.putText(frame, f"Gaze X:{screen_x} Y:{screen_y} (Adaptief)", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"Pose P:{face_data['pitch']:.0f} Y:{face_data['yaw']:.0f} R:{face_data['roll']:.0f} Err:{face_data['pose_error']:.1f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Cognitive Focus Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    except Exception as e:
        logging.error(f"Fout: {e}")
    finally:
        cap.release()
        audio.stop()
        logger.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()