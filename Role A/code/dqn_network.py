import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_dim=581, action_dim=169, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = DQNNetwork()
    dummy_state = torch.zeros((1, 581))
    q_values = model(dummy_state)

    print("Q-values shape:", q_values.shape)
    