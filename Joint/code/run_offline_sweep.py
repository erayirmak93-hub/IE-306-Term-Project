"""
run_offline_sweep.py
====================
Runs the WHOLE offline story unattended, in one command:
  - naive offline DQN            (the over-estimation FAILURE)
  - CQL at alpha = 1, 10, 50, 100 (the FIX + the alpha ablation)
  - behavioral cloning (BC)       (the baseline CQL must beat)

For each run it writes logs/offline_<tag>.csv (with mean_q per step, so you
can plot the blow-up) and weights/offline_<tag>.pt, then prints a final
summary table of the LAST mean_q for every run -> that table IS your
offline ablation result.

Just run:
  python code/run_offline_sweep.py --config configs/offline.yaml

Go out. Come back. Everything below is done.
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
    term = np.asarray(d["terminals"]).astype(np.float32)
    tout = np.asarray(d["timeouts"]).astype(np.float32)
    done = torch.tensor(np.maximum(term, tout), dtype=torch.float32)
    return obs, act, rew, nobs, done


def minibatches(n, batch_size, rng):
    idx = rng.permutation(n)
    for i in range(0, n - batch_size + 1, batch_size):
        yield idx[i:i + batch_size]


def run_one(cfg, method, alpha, tag, data):
    """Train one method. Returns the final mean_q."""
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs, act, rew, nobs, done = data
    n = obs.shape[0]
    obs_dim = obs.shape[1]
    n_actions = int(act.max().item()) + 1

    q = QNet(obs_dim, n_actions, cfg["hidden"])
    q_target = QNet(obs_dim, n_actions, cfg["hidden"])
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg["lr"])
    gamma = cfg["gamma"]

    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    log_path = os.path.join("logs", f"offline_{tag}.csv")
    lf = open(log_path, "w", newline="")
    logger = csv.writer(lf)
    logger.writerow(["update", "loss", "mean_q"])

    n_updates = cfg["n_updates"]
    batch_size = cfg["batch_size"]
    update = 0
    last_mean_q = float("nan")

    print(f"\n===== RUN: {tag}  (method={method}, alpha={alpha}) =====")
    while update < n_updates:
        for batch_idx in minibatches(n, batch_size, rng):
            b = torch.as_tensor(batch_idx, dtype=torch.int64)
            o, a, r, no, d = obs[b], act[b], rew[b], nobs[b], done[b]

            q_all = q(o)
            q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next = q_target(no).max(dim=1)[0]
                target = r + gamma * max_next * (1.0 - d)

            if method == "bc":
                loss = F.cross_entropy(q_all, a)
            elif method == "naive":
                loss = F.mse_loss(q_taken, target)
            elif method == "cql":
                bellman = F.mse_loss(q_taken, target)
                logsumexp_q = torch.logsumexp(q_all, dim=1)
                conservative = (logsumexp_q - q_taken).mean()
                loss = bellman + alpha * conservative

            opt.zero_grad()
            loss.backward()
            opt.step()

            if update % cfg["target_sync"] == 0:
                q_target.load_state_dict(q.state_dict())

            if update % cfg["log_every"] == 0:
                with torch.no_grad():
                    mean_q = q(obs[:2048]).max(dim=1)[0].mean().item()
                last_mean_q = mean_q
                logger.writerow([update, float(loss.item()), mean_q])
                print(f"[{tag}] update {update}/{n_updates} "
                      f"loss={loss.item():.2f} mean_q={mean_q:.2f}")

            update += 1
            if update >= n_updates:
                break

    torch.save(q.state_dict(), os.path.join("weights", f"offline_{tag}.pt"))
    lf.close()
    return last_mean_q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # load dataset ONCE, reuse for every run
    data = load_dataset(cfg["dataset_path"])

    # the full sweep: (method, alpha, tag)
    runs = [
        ("naive", 0.0,  "naive"),
        ("cql",   1.0,  "cql_a1"),
        ("cql",   10.0, "cql_a10"),
        ("cql",   50.0, "cql_a50"),
        ("cql",   100.0,"cql_a100"),
        ("bc",    0.0,  "bc"),
    ]

    summary = []
    for method, alpha, tag in runs:
        final_q = run_one(cfg, method, alpha, tag, data)
        summary.append((tag, method, alpha, final_q))

    # final ablation table
    print("\n\n================ OFFLINE SUMMARY ================")
    print(f"{'tag':<10}{'method':<8}{'alpha':<8}{'final_mean_q':>14}")
    print("-" * 40)
    for tag, method, alpha, final_q in summary:
        print(f"{tag:<10}{method:<8}{alpha:<8}{final_q:>14.2f}")
    print("================================================")
    print("naive should be HUGE (over-estimation). CQL should shrink as alpha grows.")


if __name__ == "__main__":
    main()
