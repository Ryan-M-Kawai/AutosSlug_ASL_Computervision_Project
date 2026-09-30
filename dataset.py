import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ASLLandmarkDataset(Dataset):
    def __init__(self, csv_path, label_map=None):
        df = pd.read_csv(csv_path)

        self.labels_raw = df["label"].values
        self.features = df.drop(columns=["label"]).values.astype(np.float32)

        if label_map is None:
            unique_labels = sorted(set(self.labels_raw))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.targets = np.array(
            [self.label_map[label] for label in self.labels_raw], dtype=np.int64
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

    def save_label_map(self, path):
        with open(path, "w") as f:
            json.dump(self.label_map, f, indent=2)

    @staticmethod
    def load_label_map(path):
        with open(path, "r") as f:
            return json.load(f)