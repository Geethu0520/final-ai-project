import cv2
import base64
import time
import json
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

status_data = {"gaze": "", "head": "", "mobile": False, "mobile_image": None}

def get_latest_status():
    return status_data

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    calibrated_angles = None
    start_time = time.time()
    mobile_detected_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Calibration phase (first 5 seconds)
        if time.time() - start_time <= 5:
            cv2.putText(frame, "Calibrating... Look at the screen", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if calibrated_angles is None:
                _, calibrated_angles = process_head_pose(frame, None)
        else:
            # Process eye movement, head pose, and mobile detection
            frame, gaze = process_eye_movement(frame)
            frame, head = process_head_pose(frame, calibrated_angles)
            frame, mobile = process_mobile_detection(frame)

            # If mobile detected, save the frame as base64
            if mobile:
                # Resize image to reduce size
                small_frame = cv2.resize(frame, (320, 240))
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                # Convert to base64
                mobile_image_base64 = base64.b64encode(buffer).decode('utf-8')
                mobile_detected_frame = mobile_image_base64
            else:
                mobile_detected_frame = None

            status_data.update({
                "gaze": gaze,
                "head": head,
                "mobile": mobile,
                "mobile_image": mobile_detected_frame
            })

            # Overlay gaze, head, and mobile detection information
            cv2.putText(frame, f"Gaze: {gaze}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Head: {head}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Mobile Detected: {'Yes' if mobile else 'No'}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if mobile else (150, 150, 150), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()