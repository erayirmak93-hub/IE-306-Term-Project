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
GAE_LAMBDA = 0.95
NUM_EPISODES = 2000
SEEDS = [0, 1, 2]


def flatten_obs(obs_dict):
    # KÖRLÜK BİTTİ: Grid geri döndü ve normalize edildi!
    drones = obs_dict['drones'].flatten() / 20.0
    orders = obs_dict['orders'].flatten() / 20.0
    grid = obs_dict['grid'].flatten() / 3.0
    time_val = obs_dict['time'] if isinstance(obs_dict['time'], np.ndarray) else np.array([obs_dict['time']])
    flat_state = np.concatenate([drones, orders, grid, time_val])
    return torch.FloatTensor(flat_state).unsqueeze(0)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        # AĞ KAPASİTESİ ARTIRILDI: 581 inputu sindirebilmesi için 512 -> 256
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, state, mask):
        shared_features = self.shared_net(state)
        logits = self.actor(shared_features)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
        logits = logits.masked_fill(~mask_tensor, -1e9)
        return torch.softmax(logits, dim=-1), self.critic(shared_features)


def compute_gae(rewards, values, next_value, gamma, gae_lambda):
    returns, advantages = [], []
    gae = 0
    val_tensors = [v.detach() for v in values] + [next_value.detach()]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * val_tensors[step + 1] - val_tensors[step]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + val_tensors[step])
    return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)


def train_agent(seed, use_adv_norm):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("DroneDispatch-v0")
    obs, _ = env.reset(seed=seed)

    input_dim = flatten_obs(obs).shape[1]
    output_dim = env.action_space.n
    model = ActorCritic(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    norm_status = "ON" if use_adv_norm else "OFF"
    log_file = f"logs/a2c_adv_norm_{norm_status}_seed_{seed}.csv"

    print(f"\n🚀 A2C Başlıyor -> Seed: {seed} | Adv Norm: {norm_status} | Input: {input_dim} | Nöron: 512")

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_return"])

        for episode in range(1, NUM_EPISODES + 1):
            obs, _ = env.reset()
            log_probs, values, rewards, entropies = [], [], [], []
            done = False
            # Exploration'ın tamamen sıfırlanmasını engelledik
            entropy_coef = max(0.01, 0.05 * (1 - episode / NUM_EPISODES))

            while not done:
                state_tensor = flatten_obs(obs)
                action_probs, value = model(state_tensor, obs['action_mask'])
                m = Categorical(action_probs)
                action = m.sample()

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                log_probs.append(m.log_prob(action))
                values.append(value.squeeze())
                rewards.append(reward)
                entropies.append(m.entropy())
                obs = next_obs

            next_state_tensor = flatten_obs(obs)
            _, next_value = model(next_state_tensor, obs['action_mask'])
            next_value = torch.tensor(0.0) if done else next_value.squeeze()

            returns, advantages = compute_gae(rewards, values, next_value, GAMMA, GAE_LAMBDA)
            if use_adv_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = torch.stack([-lp * adv.detach() for lp, adv in zip(log_probs, advantages)]).mean()
            critic_loss = torch.stack([(R - v).pow(2) for R, v in zip(returns, values)]).mean()
            entropy_loss = torch.stack(entropies).mean()
            total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            ep_return = sum(rewards)
            writer.writerow([episode, ep_return])
            if episode % 500 == 0:
                print(f"  Episode {episode:04d} | Return: {ep_return:.1f}")

    model_name = f"weights/a2c_{norm_status}_seed{seed}.pt"
    torch.save(model.state_dict(), model_name)
    print(f"  ✅ Model Kaydedildi ({model_name})")


def main():
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    for s in SEEDS: train_agent(s, True)
    for s in SEEDS: train_agent(s, False)


if __name__ == "__main__":
    main()