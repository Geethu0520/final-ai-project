# head_pose.py
import cv2
import mediapipe as mp # type: ignore
import numpy as np
import math
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 3D Model Points (Mapped to Facial Landmarks) - Standard PnP model points
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -50.0, -10.0),    # Chin
    (-30.0, 40.0, -10.0),   # Left eye
    (30.0, 40.0, -10.0),    # Right eye
    (-25.0, -30.0, -10.0),  # Left mouth corner
    (25.0, -30.0, -10.0)    # Right mouth corner
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Smoothing filter for stable head pose estimation
ANGLE_HISTORY_SIZE = 10
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)

# Global variables for state management
previous_state = "Looking at Screen"

def get_head_pose_angles(image_points, w, h):
    # Dynamic Camera Matrix Calculation
    focal_length = w  # A common approximation for focal length
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        # Standard Euler angles calculation (Pitch, Yaw, Roll)
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def smooth_angle(angle_history, new_angle):
    angle_history.append(new_angle)
    return np.mean(angle_history)

def process_head_pose(frame, calibrated_angles=None):
    global previous_state
    
    # Reset history for consistent smoothing if calibration changes or upon initial call
    if calibrated_angles is None and len(pitch_history) > 0:
        pitch_history.clear()
        yaw_history.clear()
        roll_history.clear()

    head_direction = "Looking at Screen"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # MediaPipe landmark indices for key points
        landmark_indices = {
            "nose": 1, "chin": 152, "left_eye": 33,
            "right_eye": 263, "left_mouth": 61, "right_mouth": 291
        }

        image_points = np.array([
            (int(landmarks.landmark[landmark_indices[key]].x * w),
             int(landmarks.landmark[landmark_indices[key]].y * h))
            for key in ["nose", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth"]
        ], dtype=np.float64)

        angles = get_head_pose_angles(image_points, w, h)
        if angles is None:
            return frame, head_direction

        pitch = smooth_angle(pitch_history, angles[0])
        yaw = smooth_angle(yaw_history, angles[1])
        roll = smooth_angle(roll_history, angles[2])

        if calibrated_angles is None:
            # Return current raw angles for calibration
            return frame, (pitch, yaw, roll)

        pitch_offset, yaw_offset, roll_offset = calibrated_angles
        PITCH_THRESHOLD = 8
        YAW_THRESHOLD = 12
        ROLL_THRESHOLD = 5

        # Check for look-away based on calibrated neutral position
        if abs(yaw - yaw_offset) <= YAW_THRESHOLD and abs(pitch - pitch_offset) <= PITCH_THRESHOLD and abs(roll - roll_offset) <= ROLL_THRESHOLD:
            current_state = "Looking at Screen"
        elif yaw < yaw_offset - 15:
            current_state = "Looking Left"
        elif yaw > yaw_offset + 15:
            current_state = "Looking Right"
        elif pitch > pitch_offset + 10:
            current_state = "Looking Up"
        elif pitch < pitch_offset - 10:
            current_state = "Looking Down"
        elif abs(roll - roll_offset) > 7:
            current_state = "Tilted"
        else:
            current_state = previous_state

        previous_state = current_state
        head_direction = current_state

    return frame, head_direction