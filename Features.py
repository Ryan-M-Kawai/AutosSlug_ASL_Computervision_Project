"""
Shared utilities for extracting hand landmarks with MediaPipe's Tasks API
(mediapipe >= 1.0, where the old mp.solutions API no longer exists).
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

NUM_LANDMARKS = 21
NUM_FINGER_ANGLES = 10  
FEATURE_SIZE = NUM_LANDMARKS * 3 + 1 + NUM_FINGER_ANGLES  # x, y, z per landmark, plus handedness and finger angles


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def create_landmarker(running_mode=vision.RunningMode.VIDEO, num_hands=1):
    ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=running_mode,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_feature_vector(hand_landmarks, handedness_label=None):
    """
    Convert a list of 21 MediaPipe landmarks into a translation- and
    scale-invariant feature vector, so the same sign is recognized
    regardless of where the hand is in frame or how close it is.

    Steps:
      1. Subtract the wrist position (landmark 0) from every point.
      2. Scale by the distance from wrist to middle-finger MCP (landmark 9),
         which stays roughly constant for a given hand pose regardless of
         distance from the camera.
      3. Append a handedness value: 1.0 for "Right", 0.0 for "Left" (or
         anything else). Many signs look mirror-flipped between hands, so
         giving the model this bit helps it tell them apart. Pass
         handedness_label=None (default) to fill this slot with 0.5,
         i.e. "unknown" -- useful only for quick testing, not for real
         training data.

    Returns a feature vector of shape (64,): 63 landmark values + 1
    handedness value.
    """
    coordinates = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )
    wrist = coordinates[0].copy()
    coordinates -= wrist

    scale_ref = coordinates[9]
    scale = np.linalg.norm(scale_ref) + 1e-6
    coordinates /= scale

    if handedness_label is None:
        hand = 0.5
    else:
        hand = 1.0 if handedness_label == "Right" else 0.0

    return np.append(coordinates.flatten(), np.float32(hand))  # shape (64,)
def angle(landmark_1, landmark_2, landmark_3):
    v1 = np.array([landmark_1.x, landmark_1.y])
    v2 = np.array([landmark_2.x, landmark_2.y])
    v3 = np.array([landmark_3.x, landmark_3.y])
    dot_product = np.dot(v1 - v2, v3 - v2)
    norms = np.linalg.norm(v1 - v2) * np.linalg.norm(v3 - v2)
    if norms == 0:
        return 0
    angle_rad  = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
    angle_deg  = np.degrees(angle_rad)
    return angle_deg

def finger_angles(hand_landmarks):
    thumb_features = [angle(hand_landmarks[0], hand_landmarks[1], hand_landmarks[2])] + [angle(hand_landmarks[2], hand_landmarks[3], hand_landmarks[4])]
    index_features = [angle(hand_landmarks[5], hand_landmarks[6], hand_landmarks[7])] + [angle(hand_landmarks[6], hand_landmarks[7], hand_landmarks[8])]
    middle_features = [angle(hand_landmarks[9], hand_landmarks[10], hand_landmarks[11])] + [angle(hand_landmarks[10], hand_landmarks[11], hand_landmarks[12])]
    ring_features = [angle(hand_landmarks[13], hand_landmarks[14], hand_landmarks[15])] + [angle(hand_landmarks[14], hand_landmarks[15], hand_landmarks[16])]
    pinky_features = [angle(hand_landmarks[17], hand_landmarks[18], hand_landmarks[19])] + [angle(hand_landmarks[18], hand_landmarks[19], hand_landmarks[20])]
    hand_features = thumb_features + index_features + middle_features + ring_features + pinky_features
    return hand_features

def get_features(hand_landmarks, handedness_label=None):
    return np.concatenate([landmarks_to_feature_vector(hand_landmarks, handedness_label), np.array(finger_angles(hand_landmarks), dtype = np.float32)])

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]