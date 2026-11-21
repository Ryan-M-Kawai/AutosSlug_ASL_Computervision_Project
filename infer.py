"""
Real-time ASL sign recognition using the trained model.
Detects both hands at once and shows a separate prediction label for
the left hand and the right hand.

A chart of the model's per-class probabilities for each detected hand is
drawn in a panel to the right of the camera image.

Keys:
    q   quit
    c   switch the panel between a bar chart and a pie chart
"""

import os

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision

from Features import create_landmarker, get_features, HAND_CONNECTIONS
from Classifier import ASLClassifier
from helper_functions import draw_landmarks
from Visualization import render_panel
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "asl_model.pth")


CONFIDENCE_THRESHOLD = 0.8

# Chart title colours, matching the label colours drawn on the frame below.
HAND_COLORS = {"Right": (0, 255, 0), "Left": (200, 0, 255)}

def predict(model, device, hand_landmarks, handedness_label):
    features = get_features(hand_landmarks, handedness_label)
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    return pred_idx.item(), confidence.item(), probs[0].cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    label_map = checkpoint["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]  # in model-output order
    chart_mode = "bar"

    model = ASLClassifier(
        input_size=checkpoint["input_size"], num_classes=checkpoint["num_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # num_hands=2 so both hands are detected in the same frame
    landmarker = create_landmarker(running_mode=vision.RunningMode.VIDEO, num_hands=2)
    camera_index =0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera")
        
    else:
        print("Camera opened successfully using ")
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
        hands = []  # (title, probabilities, colour) per detected hand, for the chart panel

        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            handedness_label = handedness[0].category_name  # "Left" or "Right"
            draw_landmarks(frame, hand_landmarks)

            pred_idx, confidence, probs = predict(model, device, hand_landmarks, handedness_label)
            pred_label = idx_to_label[pred_idx]
            hands.append((f"{handedness_label} hand", probs,
                          HAND_COLORS.get(handedness_label, (255, 255, 255))))

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

        panel = render_panel(hands, class_names, frame.shape[0], chart_mode, CONFIDENCE_THRESHOLD)
        cv2.imshow("ASL Recognition", np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        if key == ord('c'):
            chart_mode = "pie" if chart_mode == "bar" else "bar"

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
