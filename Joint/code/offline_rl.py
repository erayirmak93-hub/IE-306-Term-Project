"""
offline_rl.py
=============
Offline RL on a static logged dataset (DroneDispatch-v0, centralized).

The story this file tells (this IS the assignment's required narrative):
  1. NAIVE offline DQN: train Q-learning on the fixed dataset with no
     correction. Because the data does not cover every (state, action),
     the max in the Bellman target keeps picking over-optimistic
     out-of-distribution (OOD) actions, so Q-values blow up. We LOG the
     mean Q over training to SHOW this divergence.  -> the failure.
  2. CQL (Conservative Q-Learning): add a term that pushes DOWN Q-values
     for actions NOT in the data and pushes UP Q-values for actions that
     ARE in the data. This stops the over-estimation.  -> the fix.
  3. BC (Behavioral Cloning): just imitate the dataset's actions
     (supervised classification). A baseline CQL must beat.

Run one method at a time:
  python offline_rl.py --config configs/offline.yaml --method naive
  python offline_rl.py --config configs/offline.yaml --method cql
  python offline_rl.py --config configs/offline.yaml --method bc

Each run saves weights/<method>.pt and logs/offline_<method>.csv
(the csv has mean_q per eval point, so you can plot the blow-up).
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import drone_dispatch_env as dde


# ----------------------------------------------------------------------
# Q-network: obs_dim -> hidden -> hidden -> n_actions
# ----------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(path):
    d = dde.load_offline_dataset(path)
    obs = torch.tensor(np.asarray(d["observations"]), dtype=torch.float32)
    act = torch.tensor(np.asarray(d["actions"]), dtype=torch.int64)
    rew = torch.tensor(np.asarray(d["rewards"]), dtype=torch.float32)
    nobs = torch.tensor(np.asarray(d["next_observations"]), dtype=torch.float32)
    # episode boundary = terminal OR timeout (spec Section 9)
    term = np.asarray(d["terminals"]).astype(np.float32)
    tout = np.asarray(d["timeouts"]).astype(np.float32)
    done = torch.tensor(np.maximum(term, tout), dtype=torch.float32)
    return obs, act, rew, nobs, done


def minibatches(n, batch_size, rng):
    idx = rng.permutation(n)
    for i in range(0, n - batch_size + 1, batch_size):
        yield idx[i:i + batch_size]


def train(config, method):
    seed = config["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs, act, rew, nobs, done = load_dataset(config["dataset_path"])
    n = obs.shape[0]
    obs_dim = obs.shape[1]              # 181
    n_actions = int(act.max().item()) + 1   # covers the actions seen in data
    print(f"dataset: {n} transitions, obs_dim={obs_dim}, n_actions={n_actions}")

    q = QNet(obs_dim, n_actions, config["hidden"])
    q_target = QNet(obs_dim, n_actions, config["hidden"])
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=config["lr"])

    gamma = config["gamma"]
    alpha = config["cql_alpha"]          # conservatism weight (only used by cql)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    log_path = os.path.join("logs", f"offline_{method}.csv")
    lf = open(log_path, "w", newline="")
    logger = csv.writer(lf)
    logger.writerow(["update", "loss", "mean_q"])

    n_updates = config["n_updates"]
    batch_size = config["batch_size"]
    update = 0

    while update < n_updates:
        for batch_idx in minibatches(n, batch_size, rng):
            b = torch.as_tensor(batch_idx, dtype=torch.int64)
            o, a, r, no, d = obs[b], act[b], rew[b], nobs[b], done[b]

            q_all = q(o)                              # [B, n_actions]
            q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next = q_target(no).max(dim=1)[0]
                target = r + gamma * max_next * (1.0 - d)

            if method == "bc":
                # Behavioral cloning: supervised classification of dataset actions.
                # (No Bellman target at all; we just imitate.)
                logits = q_all
                loss = F.cross_entropy(logits, a)

            elif method == "naive":
                # Plain offline DQN. No correction -> over-estimation.
                loss = F.mse_loss(q_taken, target)

            elif method == "cql":
                # CQL = DQN Bellman loss + conservatism penalty.
                bellman = F.mse_loss(q_taken, target)
                # push DOWN logsumexp over all actions, push UP data action.
                # (this is the discrete-CQL conservative term)
                logsumexp_q = torch.logsumexp(q_all, dim=1)
                conservative = (logsumexp_q - q_taken).mean()
                loss = bellman + alpha * conservative

            else:
                raise ValueError(f"unknown method {method}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            # sync target net
            if update % config["target_sync"] == 0:
                q_target.load_state_dict(q.state_dict())

            # log mean Q over a fixed probe batch -> this is where naive blows up
            if update % config["log_every"] == 0:
                with torch.no_grad():
                    mean_q = q(obs[:2048]).max(dim=1)[0].mean().item()
                logger.writerow([update, float(loss.item()), mean_q])
                print(f"[{method}] update {update}/{n_updates} "
                      f"loss={loss.item():.3f} mean_q={mean_q:.2f}")

            update += 1
            if update >= n_updates:
                break

    weight_path = os.path.join("weights", f"offline_{method}.pt")
    torch.save(q.state_dict(), weight_path)
    lf.close()
    print(f"DONE [{method}]. weights -> {weight_path}, log -> {log_path}")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True, choices=["naive", "cql", "bc"])
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.method)
