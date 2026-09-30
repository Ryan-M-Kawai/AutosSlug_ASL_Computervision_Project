import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Step 1: Download the hand landmark model (only needed once) ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# --- Step 2: Set up the HandLandmarker ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Hand connections (replaces the old mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17)
]

def draw_landmarks(frame, hand_landmarks_list, handedness_list):
   h, w, _ = frame.shape
   for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        label = handedness[0].category_name  # "Left" or "Right"
        color = (0, 255, 0) if label == "Right" else (255, 0, 0)

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Label the hand near the wrist point
        wrist_x, wrist_y = points[0]
        cv2.putText(frame, label, (wrist_x - 20, wrist_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
# --- Step 3: Run webcam loop ---
cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 1

    if result.hand_landmarks:
        draw_landmarks(frame, result.hand_landmarks, result.handedness)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()