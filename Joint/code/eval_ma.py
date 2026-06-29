"""
eval_ma.py
==========
Evaluate a trained IDQN (parameter-shared) policy on DroneDispatchMA-v0,
and compare it head-to-head with a random baseline on the SAME seeds.

MA env exposes no cost_per_order in info, so we score by EPISODE RETURN
(sum of all agents' rewards over the episode), averaged over several seeds.
Higher return = better policy. Beating random is the minimum sanity bar;
the report's centralized-vs-decentralized story uses the same return metric.

Run:
  python code/eval_ma.py --config configs/idqn.yaml \
         --weights weights/idqn_share1_id1_seed0.pt --episodes 10
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml
import gymnasium as gym
import drone_dispatch_env  # noqa: F401


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def maybe_add_id(vec, aid, agent_ids, use_id):
    if not use_id:
        return np.asarray(vec, dtype=np.float32)
    one = np.zeros(len(agent_ids), dtype=np.float32)
    one[agent_ids.index(aid)] = 1.0
    return np.concatenate([np.asarray(vec, dtype=np.float32), one])


def run_episode_return(env, seed, agent_ids, use_id, policy_fn):
    """policy_fn(agent_id, encoded_obs) -> int action. Returns total return."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    done_all = False
    steps = 0
    while not done_all and steps < 2000:
        actions = {}
        for aid in agent_ids:
            enc = maybe_add_id(np.asarray(obs[aid]).ravel(), aid, agent_ids, use_id)
            actions[aid] = policy_fn(aid, enc)
        obs, rew, term, trunc, info = env.step(actions)
        total += sum(float(rew[a]) for a in agent_ids)
        done_all = all(bool(term[a]) or bool(trunc[a]) for a in agent_ids)
        steps += 1
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    use_id = cfg["use_agent_id"]

    env = gym.make(cfg["env_id"], disable_env_checker=True)
    obs, info = env.reset(seed=0)
    agent_ids = list(obs.keys())
    n_agents = len(agent_ids)
    n_actions = env.action_space[agent_ids[0]].n
    raw_dim = len(np.asarray(obs[agent_ids[0]]).ravel())
    obs_dim = raw_dim + (n_agents if use_id else 0)

    # --- load trained shared network ---
    q = QNetwork(obs_dim, n_actions, cfg["hidden"])
    q.load_state_dict(torch.load(args.weights, map_location="cpu"))
    q.eval()

    def idqn_policy(aid, enc):
        with torch.no_grad():
            qv = q(torch.tensor(enc).unsqueeze(0))
            return int(qv.argmax(dim=1).item())

    def random_policy(aid, enc):
        return np.random.randint(n_actions)

    seeds = list(range(args.episodes))

    idqn_returns, rand_returns = [], []
    for s in seeds:
        idqn_returns.append(run_episode_return(env, s, agent_ids, use_id, idqn_policy))
        rand_returns.append(run_episode_return(env, s, agent_ids, use_id, random_policy))

    idqn_returns = np.array(idqn_returns)
    rand_returns = np.array(rand_returns)

    print("\n================ MA EVAL (episode return) ================")
    print(f"seeds evaluated: {seeds}")
    print(f"IDQN   : mean {idqn_returns.mean():8.2f}  std {idqn_returns.std():7.2f}")
    print(f"random : mean {rand_returns.mean():8.2f}  std {rand_returns.std():7.2f}")
    print("----------------------------------------------------------")
    diff = idqn_returns.mean() - rand_returns.mean()
    verdict = "IDQN BEATS random" if diff > 0 else "IDQN does NOT beat random"
    print(f"difference: {diff:+.2f}  -> {verdict}")
    print("==========================================================")


if __name__ == "__main__":
    main()
