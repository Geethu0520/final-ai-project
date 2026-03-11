# mobile_detection.py - FIXED & ROBUST VERSION

import cv2
import torch
from ultralytics import YOLO
import os

# Global variables
model = None
face_cascade = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# === CONFIGURATION ===
MOBILE_CLASS_NAME = "phone"        # Change this if your class name is different
CONFIDENCE_THRESHOLD = 0.72          # Increased from 0.6
FACE_OVERLAP_THRESHOLD = 0.35        # If >40% of detected "mobile" overlaps with face → ignore
MIN_FACE_SIZE = (70,70)            # Smaller faces ignored (reduces noise)

def load_model():
    global model
    if model is not None:
        return True

    model_path = "model/best_yolov12.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return False

    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"[INFO] YOLO model loaded successfully on {device}")
        
        # Print class names to verify
        if hasattr(model, 'names'):
            print("[INFO] Model classes:", model.names)
        else:
            print("[WARNING] Could not read class names from model.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = None
        return False

def load_face_cascade():
    global face_cascade
    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("[WARNING] Haar Cascade not loaded. Face filtering disabled.")
            face_cascade = None
        else:
            print("[INFO] Haar Cascade loaded for face filtering.")

def boxes_overlap_ratio(box_a, box_b):
    """Return intersection over box_a area"""
    x1, y1, x2, y2 = box_a
    fx, fy, fw, fh = box_b
    fx2, fy2 = fx + fw, fy + fh

    ix1 = max(x1, fx)
    iy1 = max(y1, fy)
    ix2 = min(x2, fx2)
    iy2 = min(y2, fy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    box_a_area = (x2 - x1) * (y2 - y1)
    return inter_area / box_a_area if box_a_area > 0 else 0.0

def is_near_face(mobile_box, faces):
    for (fx, fy, fw, fh) in faces:
        if boxes_overlap_ratio(mobile_box, (fx, fy, fw, fh)) > FACE_OVERLAP_THRESHOLD:
            return True
    return False

def process_mobile_detection(frame):
    global model, face_cascade

    mobile_detected = False

    # Load model if needed
    if model is None:
        if not load_model():
            return frame, mobile_detected

    # Load face detector
    if face_cascade is None:
        load_face_cascade()

    try:
        results = model(frame, device=device, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return frame, mobile_detected

    # Convert frame to gray for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []
    if face_cascade:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )

    h, w = frame.shape[:2]

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls_id] if hasattr(model, 'names') else f"Class {cls_id}"

            # === STRICT: Only allow "mobile" class ===
            if label.lower() != MOBILE_CLASS_NAME.lower():
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            mobile_box = (x1, y1, x2, y2)

            # === FACE FILTER: Ignore if overlaps significantly with face ===
            if face_cascade and is_near_face(mobile_box, faces):
                # Optional: draw red box on false positive
                cv2.rectangle(frame, (x1, y1), (y1 - 30), y1), (x1 + 120, y1), (0, 0, 255), -1
                cv2.putText(frame, "False Mobile", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue  # Skip this detection

            # === LEGIT MOBILE DETECTED ===
            mobile_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(f"[ALERT] Mobile detected! Confidence: {conf:.2f}")

    # Optional: Draw faces for debugging
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame, mobile_detected


# === Standalone Test ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Mobile Detection Test - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected = process_mobile_detection(frame.copy())

        status = "MOBILE DETECTED!" if detected else "No mobile"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255) if detected else (0, 255, 0), 3)

        cv2.imshow("Anti-Cheating Mobile Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()