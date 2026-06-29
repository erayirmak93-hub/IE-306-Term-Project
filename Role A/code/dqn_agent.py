import random
import numpy as np
import torch

from dqn_network import DQNNetwork


class DQNAgent:

    def __init__(
        self,
        state_dim=581,
        action_dim=169,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_net = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(self.device)

    def select_action(self, state, action_mask):

        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            return self.action_dim - 1

        if random.random() < self.epsilon:
            return int(random.choice(valid_actions))

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        q_values = q_values.cpu().numpy()[0]

        q_values[action_mask == 0] = -1e9

        return int(np.argmax(q_values))

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


if __name__ == "__main__":

    dummy_state = np.zeros(581, dtype=np.float32)

    dummy_mask = np.ones(169, dtype=np.int8)

    agent = DQNAgent()

    action = agent.select_action(
        dummy_state,
        dummy_mask
    )

    print("Selected action:", action)

    print("Epsilon:", agent.epsilon)

    agent.decay_epsilon()

    print("Epsilon after decay:", agent.epsilon)