import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)