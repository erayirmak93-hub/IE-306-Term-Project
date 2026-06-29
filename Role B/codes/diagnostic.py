import sys
import os

# Bu blok weights klasörünü bulmasını sağlayacak
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.chdir(root_dir)

import gymnasium as gym
import os
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config
from drone_dispatch_env.baselines import make_baseline
import torch
import torch.nn as nn
import numpy as np


def flatten_obs(obs_dict):
    drones = obs_dict['drones'].flatten() / 20.0
    orders = obs_dict['orders'].flatten() / 20.0
    grid = obs_dict['grid'].flatten() / 3.0
    time_val = obs_dict['time'] if isinstance(obs_dict['time'], np.ndarray) else np.array([obs_dict['time']])
    flat_state = np.concatenate([drones, orders, grid, time_val])
    return torch.FloatTensor(flat_state).unsqueeze(0)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, state, mask):
        shared = self.shared_net(state)
        logits = self.actor(shared).masked_fill(~torch.BoolTensor(mask).unsqueeze(0), -1e9)
        return torch.softmax(logits, dim=-1), self.critic(shared)


class A2CPolicy:
    def __init__(self, path, input_dim, output_dim):
        self.model = ActorCritic(input_dim, output_dim)
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, obs):
        with torch.no_grad():
            probs, _ = self.model(flatten_obs(obs), obs['action_mask'])
            return torch.argmax(probs).item()


def run_diagnostic(policy, env, name):
    obs, _ = env.reset(seed=42)  # Her iki ajan da aynı siparişleri görsün diye seed sabitlendi
    done = False
    steps = 0
    total_reward = 0

    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        total_reward += reward

    print(f"🔍 --- {name} DAVRANIŞ ANALİZİ ---")
    print(f"Toplam Adım Sayısı: {steps}")
    print(f"Bölüm Sonu Toplam Ödül: {total_reward:.2f}")
    print(f"Simülatör Final Çıktısı (Info): {info}")
    print("-" * 60)


def main():
    config = Config()
    env = gym.make("DroneDispatch-v0")

    obs, _ = env.reset()
    input_dim = flatten_obs(obs).shape[1]
    output_dim = env.action_space.n

    print("\n🚀 DAVRANIŞSAL TEŞHİS BAŞLIYOR (Sadece 1 Bölüm)\n")

    # 1. Random Ajan Ne Yapıyor?
    random_policy = make_baseline("random", config)
    run_diagnostic(random_policy, env, "RANDOM BASELINE")

    # 2. Greedy Baseline Ne Yapıyor? (BUNU EKLE)
    greedy_policy = make_baseline("greedy", config)
    run_diagnostic(greedy_policy, env, "GREEDY NEAREST BASELINE")

    # 3. Bizim Ajan (Seed 1) Ne Yapıyor?
    best_model_path = "weights/a2c_OFF_seed0.pt"
    if os.path.exists(best_model_path):
        a2c_policy = A2CPolicy(best_model_path, input_dim, output_dim)
        run_diagnostic(a2c_policy, env, "A2C AJANI (Ours)")
    else:
        print("Model dosyası bulunamadı!")

    env.close()


if __name__ == "__main__":
    main()