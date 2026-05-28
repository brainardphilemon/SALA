# models/classifiers.py

import torch.nn as nn

class StrongMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.2, act="gelu"):
        super().__init__()
        Act = nn.Mish if act == "mish" else nn.GELU

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.fc_in(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out(h).squeeze(-1)