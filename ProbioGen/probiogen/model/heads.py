from __future__ import annotations

from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))
