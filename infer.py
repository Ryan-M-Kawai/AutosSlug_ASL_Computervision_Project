"""
Real-time ASL sign recognition using the trained model.
Detects both hands at once and shows a separate prediction label for
the left hand and the right hand.

"""

import os

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks.python import vision

from Features import create_landmarker, get_features, HAND_CONNECTIONS
from Classifier import ASLClassifier
from helper_functions import draw_landmarks
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "asl_model.pth")


CONFIDENCE_THRESHOLD = 0.8

def predict(model, device, hand_landmarks, handedness_label):
    features = get_features(hand_landmarks, handedness_label)
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    print(f"pred_idx: {pred_idx.item()}")
    return pred_idx.item(), confidence.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    label_map = checkpoint["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}

    model = ASLClassifier(
        input_size=checkpoint["input_size"], num_classes=checkpoint["num_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # num_hands=2 so both hands are detected in the same frame
    landmarker = create_landmarker(running_mode=vision.RunningMode.VIDEO, num_hands=2)
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

        left_text, right_text = None, None

        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            handedness_label = handedness[0].category_name  # "Left" or "Right"
            draw_landmarks(frame, hand_landmarks)

            pred_idx, confidence = predict(model, device, hand_landmarks, handedness_label)
            pred_label = idx_to_label[pred_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                text = f"{handedness_label}: {pred_label} ({confidence:.2f})"
            else:
                text = f"{handedness_label}: ? ({confidence:.2f})"

            if handedness_label == "Right":
                left_text = text
            else:
                right_text = text

        # Fixed on-screen positions so each hand's label always shows in the
        # same spot regardless of detection order -- left label top-left,
        # right label top-right.
        if left_text:
            cv2.putText(frame, left_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        if right_text:
            cv2.putText(frame, right_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 255), 3)

        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()