import torch
from torch import nn


class PriorDiscriminator(nn.Module):
    def __init__(self, code_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),              # for BCE loss
        )
    def forward(self, z):
        return self.net(z).view(-1)   # (batch,) probabilities
