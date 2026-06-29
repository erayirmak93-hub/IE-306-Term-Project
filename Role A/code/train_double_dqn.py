import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import drone_dispatch_env

PROJECT_DIR = r"C:\Users\Berk\Desktop\IE306_drone_dispatch_env"
DESKTOP_DIR = r"C:\Users\Berk\Desktop"

sys.path.append(PROJECT_DIR)
sys.path.append(DESKTOP_DIR)

from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from state_processor import preprocess_state
from dqn_network import DQNNetwork

NUM_EPISODES = 100
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE_FREQ = 2


def train_double_dqn():

    env = gym.make("DroneDispatch-v0")

    agent = DQNAgent()

    target_net = DQNNetwork().to(agent.device)
    target_net.load_state_dict(agent.policy_net.state_dict())

    optimizer = optim.Adam(
        agent.policy_net.parameters(),
        lr=LR
    )

    loss_fn = nn.SmoothL1Loss()

    replay_buffer = ReplayBuffer(capacity=50000)

    for episode in range(NUM_EPISODES):

        obs, info = env.reset(seed=episode)

        state = preprocess_state(obs)
        action_mask = obs["action_mask"]

        total_reward = 0
        done = False
        step_count = 0
        losses = []

        while not done:

            action = agent.select_action(
                state,
                action_mask
            )

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            next_state = preprocess_state(next_obs)
            next_action_mask = next_obs["action_mask"]

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_action_mask
            )

            state = next_state
            action_mask = next_action_mask

            total_reward += reward
            step_count += 1

            if len(replay_buffer) >= BATCH_SIZE:

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    next_action_masks
                ) = replay_buffer.sample(BATCH_SIZE)

                states = torch.tensor(
                    states,
                    dtype=torch.float32
                ).to(agent.device)

                actions = torch.tensor(
                    actions,
                    dtype=torch.long
                ).unsqueeze(1).to(agent.device)

                rewards = torch.tensor(
                    rewards,
                    dtype=torch.float32
                ).unsqueeze(1).to(agent.device)

                next_states = torch.tensor(
                    next_states,
                    dtype=torch.float32
                ).to(agent.device)

                dones = torch.tensor(
                    dones,
                    dtype=torch.float32
                ).unsqueeze(1).to(agent.device)

                next_action_masks = torch.tensor(
                    next_action_masks,
                    dtype=torch.float32
                ).to(agent.device)

                q_values = (
                    agent.policy_net(states)
                    .gather(1, actions)
                )

                with torch.no_grad():

                    next_q_policy = agent.policy_net(next_states)

                    next_q_policy[next_action_masks == 0] = -1e9

                    best_next_actions = next_q_policy.argmax(
                        1,
                        keepdim=True
                    )

                    next_q_target = target_net(next_states)

                    next_q_values = next_q_target.gather(
                        1,
                        best_next_actions
                    )

                    target_q_values = (
                        rewards
                        + GAMMA * next_q_values * (1 - dones)
                    )

                loss = loss_fn(
                    q_values,
                    target_q_values
                )

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    agent.policy_net.parameters(),
                    max_norm=10.0
                )

                optimizer.step()

                losses.append(loss.item())

        agent.decay_epsilon()

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(
                agent.policy_net.state_dict()
            )

        avg_loss = np.mean(losses) if losses else 0

        print(
            f"Episode {episode+1}/{NUM_EPISODES} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {step_count} | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    weights_dir = os.path.join(
        PROJECT_DIR,
        "weights"
    )

    os.makedirs(
        weights_dir,
        exist_ok=True
    )

    torch.save(
        agent.policy_net.state_dict(),
        os.path.join(
            weights_dir,
            "double_dqn_100ep.pt"
        )
    )

    print("Training finished.")
    print(
        "Model saved to weights/double_dqn_100ep.pt"
    )


if __name__ == "__main__":
    train_double_dqn()