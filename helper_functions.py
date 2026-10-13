import cv2
from Features import HAND_CONNECTIONS
import os
import csv
from collections import Counter
from Features import FEATURE_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
CSV_HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_SIZE)]

README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
START = "<!-- DATA_COUNTS_START -->"
END = "<!-- DATA_COUNTS_END -->"

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)



def download_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def load_existing_data(csv_path):
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


def update_readme(counts):
    """Replace the dataset summary block at the bottom of the README."""
    if counts:
        lines = ["| Label | Samples |", "|-------|---------|"]
        for label, n in sorted(counts.items()):
            lines.append(f"| {label} | {n} |")
        lines.append(f"| **Total** | **{sum(counts.values())}** ({len(counts)} labels) |")
        summary = "\n".join(lines)
    else:
        summary = "No training data collected yet."

    block = f"{START}\n## Dataset Summary\n\n{summary}\n{END}"

    text = ""
    if os.path.exists(README_PATH):
        with open(README_PATH, "r") as f:
            text = f.read()

    if START in text and END in text:
        text = text.split(START)[0] + block + text.split(END, 1)[1]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"

    with open(README_PATH, "w") as f:
        f.write(text)
