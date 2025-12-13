import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_variables import *

class ActionDecoder(nn.Module):
    def __init__(self, input_dim=latent_dim + deterministic_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state_features):
        return self.net(state_features)

class ValueDecoder(nn.Module):
    def __init__(self, input_dim=latent_dim + deterministic_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state_features):
        return self.net(state_features).squeeze(-1)