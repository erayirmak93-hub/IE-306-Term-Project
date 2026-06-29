import os
import sys
import torch
import gymnasium as gym
import drone_dispatch_env

PROJECT_DIR = r"C:\Users\Berk\Desktop\IE306_drone_dispatch_env"
DESKTOP_DIR = r"C:\Users\Berk\Desktop"

sys.path.append(PROJECT_DIR)
sys.path.append(DESKTOP_DIR)

from drone_dispatch_env import Config, evaluate
from dqn_agent import DQNAgent
from dueling_dqn_network import DuelingDQNNetwork
from state_processor import preprocess_state


MODEL_PATH = os.path.join(PROJECT_DIR, "weights", "dueling_dqn_100ep.pt")


class DuelingPolicy:
    def __init__(self):
        self.agent = DQNAgent(epsilon=0.0)
        self.agent.policy_net = DuelingDQNNetwork().to(self.agent.device)
        self.agent.policy_net.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.agent.device)
        )
        self.agent.policy_net.eval()

    def act(self, obs):
        state = preprocess_state(obs)
        action_mask = obs["action_mask"]
        return self.agent.select_action(state, action_mask)


policy = DuelingPolicy()

results = evaluate(
    policy,
    Config(),
    seeds=[0, 1, 2]
)

print(results["mean"])