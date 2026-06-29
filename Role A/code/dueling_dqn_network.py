import torch
import torch.nn as nn


class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim=581, action_dim=169, hidden_dim=256):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


if __name__ == "__main__":
    model = DuelingDQNNetwork()

    dummy_state = torch.zeros((1, 581))

    q_values = model(dummy_state)

    print("Q-values shape:", q_values.shape)