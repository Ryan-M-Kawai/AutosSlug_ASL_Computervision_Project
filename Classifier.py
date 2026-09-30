import torch.nn as nn


class ASLClassifier(nn.Module):
    """
    Simple feedforward network over the 64-dim feature vector: 21 hand
    landmarks x, y, z (63 values) plus a handedness flag (1 value). This
    is enough for static single-frame ASL signs (letters, numbers, simple
    words) since the landmark positions and hand side carry the shape
    information.
    """

    def __init__(self, input_size=74, num_classes=26, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)