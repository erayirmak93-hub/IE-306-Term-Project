import os
import sys
import numpy as np
import torch
import gymnasium as gym
import drone_dispatch_env

PROJECT_DIR = r"C:\Users\Berk\Desktop\IE306_drone_dispatch_env"
DESKTOP_DIR = r"C:\Users\Berk\Desktop"

sys.path.append(PROJECT_DIR)
sys.path.append(DESKTOP_DIR)

from dqn_agent import DQNAgent
from state_processor import preprocess_state
from dueling_dqn_network import DuelingDQNNetwork


MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "weights",
    "dueling_dqn_100ep.pt"
)

EVAL_SEEDS = [0, 1, 2, 3, 4]


def evaluate_dueling_dqn():
    env = gym.make("DroneDispatch-v0")

    agent = DQNAgent(epsilon=0.0)
    agent.policy_net = DuelingDQNNetwork().to(agent.device)

    agent.policy_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device)
    )

    agent.policy_net.eval()

    rewards = []
    steps_list = []

    for seed in EVAL_SEEDS:
        obs, info = env.reset(seed=seed)

        state = preprocess_state(obs)
        action_mask = obs["action_mask"]

        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, action_mask)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            state = preprocess_state(next_obs)
            action_mask = next_obs["action_mask"]

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

        print(
            f"Seed {seed} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps}"
        )

    print("\nEvaluation Summary")
    print("Mean Reward:", np.mean(rewards))
    print("Std Reward:", np.std(rewards))
    print("Mean Steps:", np.mean(steps_list))


if __name__ == "__main__":
    evaluate_dueling_dqn()