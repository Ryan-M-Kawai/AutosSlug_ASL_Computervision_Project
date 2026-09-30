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

import cv2
import mediapipe as mp

from Features import create_landmarker, get_features, FEATURE_SIZE
from mediapipe.tasks.python import vision

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
CSV_HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_SIZE)]
SAMPLES_PER_LABEL_TARGET = 200  # rough guideline, not enforced


def ensure_csv_exists(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def load_existing_counts(csv_path):
    """Return a Counter of {label: sample_count} from an existing CSV, if any."""
    counts = Counter()
    if not os.path.exists(csv_path):
        return counts

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                counts[row[0]] += 1
    return counts


def print_summary(counts):
    if not counts:
        print("No existing data found -- starting fresh.")
        return

    print("Existing data on file:")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n} samples")
    print(f"  Total: {sum(counts.values())} samples across {len(counts)} labels")


def delete_label_data(csv_path, label):
    """
    Permanently remove every row belonging to `label` from the CSV.
    Returns the number of samples deleted.
    """
    if not os.path.exists(csv_path):
        print("No data file found -- nothing to delete.")
        return 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Data file is empty -- nothing to delete.")
        return 0

    header, data_rows = rows[0], rows[1:]
    kept_rows = [row for row in data_rows if row and row[0] != label]
    deleted_count = len(data_rows) - len(kept_rows)

    if deleted_count == 0:
        print(f"No samples found for label '{label}'.")
        return 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept_rows)

    print(f"Deleted {deleted_count} samples for label '{label}'.")
    return deleted_count


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ensure_csv_exists(CSV_PATH)

    counts = load_existing_counts(CSV_PATH)
    print_summary(counts)

    landmarker = create_landmarker(running_mode=vision.RunningMode.VIDEO, num_hands=1)
    cap = cv2.VideoCapture(0)
    timestamp_ms = 0

    try:
        while True:
            entry = input(
                f"\nEnter label to collect (target ~{SAMPLES_PER_LABEL_TARGET} samples), "
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
    print(f"Saved data to {CSV_PATH}")


if __name__ == "__main__":
    main()