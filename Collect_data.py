"""
Collect labeled ASL hand-sign training data using your webcam.

Usage:
    python collect_data.py

Controls:
    - Type a label (e.g. "A", "B", "HELLO") and press Enter to start
      collecting samples for that sign.
    - Hold the sign steady in front of the camera; frames are captured
      automatically while a hand is detected.
    - Press 'q' to stop collecting the current label. You'll be dropped
      back to the label prompt, where you can type the same label to
      keep adding to it, or a new label to switch signs.
    - Type "delete <label>" at the prompt to permanently remove all
      collected samples for that label (asks for confirmation first).
    - Press Ctrl+C at the label prompt to quit and save.

Output:
    data/landmarks.csv  -- one row per sample: label, then feature values.
    Feature layout is defined once in Features.py (FEATURE_SIZE) so this
    file, infer.py, and model.py can never drift out of sync.
"""

import csv
import os
from collections import Counter
from helper_functions import load_existing_data, print_summary, delete_label_data, update_readme, download_csv
import cv2
import mediapipe as mp

from Features import create_landmarker, get_features, FEATURE_SIZE
from mediapipe.tasks.python import vision

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
CSV_HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_SIZE)]

# README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
# START = "<!-- DATA_COUNTS_START -->"
# END = "<!-- DATA_COUNTS_END -->"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_csv(CSV_PATH)

    counts = load_existing_data(CSV_PATH)
    print_summary(counts)
    update_readme(counts)
    landmarker = create_landmarker(running_mode=vision.RunningMode.VIDEO, num_hands=1)
    cap = cv2.VideoCapture(0)
    timestamp_ms = 0

    try:
        while True:
            entry = input(
                f"\nEnter label to collect "
                f"'delete <label>' to remove a label's data, or Ctrl+C to quit: "
            ).strip()
            if not entry:
                continue

            if entry.lower().startswith("delete "):
                label_to_delete = entry[len("delete "):].strip()
                if not label_to_delete:
                    print("Usage: delete <label>")
                    continue
                confirm = input(
                    f"Delete ALL samples for '{label_to_delete}'? This cannot be undone. [y/N]: "
                ).strip().lower()
                if confirm == "y":
                    deleted = delete_label_data(CSV_PATH, label_to_delete)
                    if deleted:
                        counts.pop(label_to_delete, None)
                else:
                    print("Cancelled.")
                continue

            label = entry
            count = counts[label]  # resume from however many already exist
            print(f"Collecting for '{label}' ({count} so far). Hold the sign steady. Press 'q' to stop.")

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    timestamp_ms += 1

                    if result.hand_landmarks:
                        hand_landmarks = result.hand_landmarks[0]
                        handedness_label = result.handedness[0][0].category_name
                        features = get_features(hand_landmarks, handedness_label)
                        writer.writerow([label] + features.tolist())
                        count += 1
                        counts[label] = count

                        h, w, _ = frame.shape
                        for lm in hand_landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    cv2.putText(
                        frame, f"Label: {label}  Samples: {count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    )
                    cv2.imshow("Collecting Data", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        f.flush()
                        print(f"Stopped. '{label}' now has {count} samples total.")
                        break

    except KeyboardInterrupt:
        print("\nStopped collecting.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    update_readme(counts)
    print(f"Saved data to {CSV_PATH}")


if __name__ == "__main__":
    main()