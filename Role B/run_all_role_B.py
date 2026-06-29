import os
import sys
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate
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


def get_eval_metrics(model_path, config, seeds, input_dim, output_dim):
    try:
        policy = A2CPolicy(model_path, input_dim, output_dim)
        score = evaluate(policy, config, seeds=seeds)
        return score['mean']['cost_per_order'], score['mean'].get('success_rate', 0) * 100
    except Exception:
        return float('inf'), 0.0


def main():
    config = Config()
    test_seeds = [100, 101, 102]

    print("\n🚀 AKADEMİK STANDARTLARDA EVALUATION BAŞLIYOR... (Sıfır Sızıntı)\n")

    env = gym.make("DroneDispatch-v0")
    obs, _ = env.reset()
    input_dim = flatten_obs(obs).shape[1]
    output_dim = env.action_space.n
    env.close()

    rand_score = evaluate(make_baseline("random", config), config, seeds=test_seeds)
    greedy_score = evaluate(make_baseline("greedy", config), config, seeds=test_seeds)

    try:
        milp_score = evaluate(make_baseline("milp", config), config, seeds=test_seeds)
        milp_cost = f"{milp_score['mean']['cost_per_order']:.2f}"
        milp_succ = f"{milp_score['mean'].get('success_rate', 0) * 100:.1f}%"
    except Exception:
        milp_cost, milp_succ = "N/A", "N/A"

    # DATA LEAKAGE ENGELLENDİ: Sadece Seed 0 (ON) ana model olarak seçildi.
    primary_model_path = "weights/a2c_ON_seed0.pt"
    p_cost, p_succ = get_eval_metrics(primary_model_path, config, test_seeds, input_dim, output_dim)

    # FORMATLAMA HATASI BURADA DÜZELTİLDİ (:<12.2f)
    print("=" * 75)
    print(f"{'Method / Baseline':<35} | {'Cost/Order':<12} | {'Success Rate'}")
    print("-" * 75)
    print(
        f"{'Random Baseline':<35} | {rand_score['mean']['cost_per_order']:<12.2f} | {rand_score['mean'].get('success_rate', 0) * 100:.1f}%")
    print(
        f"{'Greedy Nearest Baseline (The Bar)':<35} | {greedy_score['mean']['cost_per_order']:<12.2f} | {greedy_score['mean'].get('success_rate', 0) * 100:.1f}%")
    print(f"{'MILP Rolling Horizon':<35} | {milp_cost:<12} | {milp_succ}")

    # Eğer model okunamadıysa N/A bas, okunduysa formatla
    p_cost_str = f"{p_cost:<12.2f}" if p_cost != float('inf') else f"{'N/A':<12}"
    print(f"{'A2C + GAE (Primary: Seed0 ON)':<35} | {p_cost_str} | {p_succ:.1f}%")
    print("=" * 75)

    print("\n📊 ABLATION & SEED SENSITIVITY BREAKDOWN (Test Seeds [100,101,102])")
    print("-" * 75)

    a2c_variants = [
        ("A2C_ON_Seed0 (Primary)", "weights/a2c_ON_seed0.pt"),
        ("A2C_ON_Seed1", "weights/a2c_ON_seed1.pt"),
        ("A2C_ON_Seed2", "weights/a2c_ON_seed2.pt"),
        ("A2C_OFF_Seed0", "weights/a2c_OFF_seed0.pt"),
        ("A2C_OFF_Seed1", "weights/a2c_OFF_seed1.pt"),
        ("A2C_OFF_Seed2", "weights/a2c_OFF_seed2.pt"),
    ]

    all_costs = []
    for label, path in a2c_variants:
        if os.path.exists(path):
            cost, succ = get_eval_metrics(path, config, test_seeds, input_dim, output_dim)
            cost_str = f"{cost:.2f}" if cost != float('inf') else "N/A"
            print(f" -> {label:<25} | Cost/Order: {cost_str:<6} | Success Rate: {succ:.1f}%")
            if "ON" in label and cost != float('inf'):
                all_costs.append(cost)
        else:
            print(f" -> {label:<25} | Eksik Model Dosyası!")

    if all_costs:
        print("-" * 75)
        print(f"A2C + GAE (Adv Norm ON) Summary: Mean Cost = {np.mean(all_costs):.2f} ± {np.std(all_costs):.2f}")
        print("=" * 75)


if __name__ == "__main__":
    main()