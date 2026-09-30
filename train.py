"""
Train the ASL hand-sign classifier on data collected with collect_data.py.

Usage:
    python train.py
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ASLLandmarkDataset
from Classifier import ASLClassifier
from Features import FEATURE_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "asl_model.pth")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "label_map.json")

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ASLLandmarkDataset(CSV_PATH)
    dataset.save_label_map(LABEL_MAP_PATH)
    num_classes = len(dataset.label_map)
    print(f"Loaded {len(dataset)} samples across {num_classes} classes: {list(dataset.label_map.keys())}")

    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ASLClassifier(input_size=FEATURE_SIZE, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss: {train_loss:.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": dataset.label_map,
                    "input_size": FEATURE_SIZE,
                    "num_classes": num_classes,
                },
                CHECKPOINT_PATH,
            )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()