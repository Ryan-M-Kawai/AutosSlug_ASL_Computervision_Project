"""
Summarise the collected training data (data/landmarks.csv) as charts of how
many samples each label has: a bar chart and a pie chart side by side.

Usage:
    python panda_analysis.py

Writes dataset_summary.png next to this file.
"""

import os

import cv2
import numpy as np
import pandas as pd

from Visualization import draw_bar_chart, draw_pie_chart

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "dataset_summary.png")

CHART_SIZE = (480, 640)  # (height, width) of each chart


def main():
    # Only the label column is needed. Reading it as text keeps labels such as
    # "1" or "NA" from being turned into numbers or NaN.
    df = pd.read_csv(CSV_PATH, usecols=["label"], dtype=str, keep_default_na=False)
    labels, counts = np.unique(df["label"].to_numpy(), return_counts=True)
    total = counts.sum()
    if total == 0:
        print(f"No samples in {CSV_PATH} yet. Collect some data first (python Collect_data.py).")
        return

    for label, n in zip(labels, counts):
        print(f"{label}: {n} ({n / total:.1%})")
    print(f"Total: {total} samples across {len(labels)} labels")
    print(f"Largest label has {counts.max() / counts.min():.1f}x the samples of the smallest")

    bars = draw_bar_chart(counts, labels, CHART_SIZE, "Samples per label", fmt="{:.0f}")
    pie = draw_pie_chart(counts, labels, CHART_SIZE, f"Share of {total} samples")
    cv2.imwrite(OUT_PATH, np.hstack([bars, pie]))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
