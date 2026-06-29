import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.chdir(root_dir)

import gymnasium as gym
import drone_dispatch_env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import csv

LEARNING_RATE = 1e-3
GAMMA = 0.99
NUM_EPISODES = 1000
SEEDS = [0, 1, 2]


def flatten_obs(obs_dict):
    # DÜZELTİLDİ: Grid geri döndü, doğru normalize edildi (/3.0), Time hatası giderildi.
    drones = obs_dict['drones'].flatten() / 20.0
    orders = obs_dict['orders'].flatten() / 20.0
    grid = obs_dict['grid'].flatten() / 3.0
    time_val = obs_dict['time'] if isinstance(obs_dict['time'], np.ndarray) else np.array([obs_dict['time']])
    flat_state = np.concatenate([drones, orders, grid, time_val])
    return torch.FloatTensor(flat_state).unsqueeze(0)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # KAPASİTE ARTIRILDI: 581 Input'u sindirmesi için 512 Nöron eklendi.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, state, mask):
        logits = self.net(state)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
        logits = logits.masked_fill(~mask_tensor, -1e9)
        return torch.softmax(logits, dim=-1)


def calculate_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # TENSOR MATEMATİĞİ: Daha stabil öğrenme için normalize edildi.
    return (returns - returns.mean()) / (returns.std() + 1e-8)


def main():
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for seed in SEEDS:
        print(f"\n🚀 REINFORCE Başlıyor -> Seed: {seed} | Nöron: 512 | Grid: ON")
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = gym.make("DroneDispatch-v0")
        obs, _ = env.reset(seed=seed)

        # DİNAMİK BOYUT
        input_dim = flatten_obs(obs).shape[1]
        output_dim = env.action_space.n

        policy = PolicyNetwork(input_dim, output_dim)
        optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

        log_file = f"logs/reinforce_seed_{seed}.csv"

        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_return"])

            for episode in range(1, NUM_EPISODES + 1):
                obs, _ = env.reset()
                log_probs, rewards = [], []
                done = False

                while not done:
                    state_tensor = flatten_obs(obs)
                    action_probs = policy(state_tensor, obs['action_mask'])
                    m = Categorical(action_probs)
                    action = m.sample()

                    next_obs, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated

                    log_probs.append(m.log_prob(action))
                    rewards.append(reward)
                    obs = next_obs


                returns = calculate_returns(rewards, GAMMA)
                baseline = returns.mean()  # Teorik kitaba uygun net baseline
                policy_loss = torch.stack([-lp * (G - baseline) for lp, G in zip(log_probs, returns)]).mean()

                optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                ep_return = sum(rewards)
                writer.writerow([episode, ep_return])

                if episode % 250 == 0:
                    print(f"  Episode {episode:04d} | Return: {ep_return:.1f}")

        # Her Seed ayrı kaydediliyor
        model_name = f"weights/reinforce_seed{seed}.pt"
        torch.save(policy.state_dict(), model_name)
        print(f"  ✅ Model Kaydedildi ({model_name})")


if __name__ == "__main__":
    main()