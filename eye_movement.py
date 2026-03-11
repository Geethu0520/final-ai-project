# eye_movement.py
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices from MediaPipe Face Mesh
LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 153]  # approximate region
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 380]  # approximate region


def detect_pupil(eye_region):
    if eye_region is None or eye_region.size == 0:
        return None, None  # Prevent crash if eye region is invalid

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        pupil_contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(pupil_contour)
        return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
    
    return None, None


def process_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaze_direction = "Looking Center"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape
        left_eye_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_LANDMARKS])
        right_eye_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_LANDMARKS])

        # Get bounding rectangles for the eyes
        left_eye_rect = cv2.boundingRect(left_eye_points)
        right_eye_rect = cv2.boundingRect(right_eye_points)

        # Extract eye regions
        left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
        right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

        # Detect pupils
        left_pupil, left_bbox = detect_pupil(left_eye)
        right_pupil, right_bbox = detect_pupil(right_eye)

        # Draw bounding boxes and pupils
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)

        if left_pupil and left_bbox:
            cv2.circle(frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 5, (0, 0, 255), -1)
        if right_pupil and right_bbox:
            cv2.circle(frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 5, (0, 0, 255), -1)

        # Gaze Detection
        if left_pupil and right_pupil:
            lx, ly = left_pupil
            rx, ry = right_pupil

            eye_width = left_eye_rect[2]
            eye_height = left_eye_rect[3]
            norm_ly, norm_ry = ly / eye_height, ry / eye_height

            if lx < eye_width // 3 and rx < eye_width // 3:
                gaze_direction = "Looking Left"
            elif lx > 2 * eye_width // 3 and rx > 2 * eye_width // 3:
                gaze_direction = "Looking Right"
            elif norm_ly < 0.3 and norm_ry < 0.3:
                gaze_direction = "Looking Up"
            elif norm_ly > 0.5 and norm_ry > 0.5:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"

    return frame, gaze_direction