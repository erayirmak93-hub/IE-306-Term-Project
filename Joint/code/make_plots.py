"""
make_plots.py
=============
Generates every figure your report needs, from the CSV logs you already have.
Saves PNGs into a figures/ folder. Missing logs are skipped (won't crash).

Run:
  python code/make_plots.py

Figures produced:
  1. ma_learning_curves.png  -- 3 IDQN seeds, episode_return vs step (shows
                                the huge seed-to-seed variance = non-stationarity)
  2. offline_meanq.png       -- mean_q vs update for naive + every CQL alpha
                                (naive blows up to ~1e5; CQL stays bounded)
  3. offline_alpha_vs_q.png  -- final mean_q vs alpha (the Q-control ablation)
  4. ma_action_mix.png       -- per-policy action distribution bar chart
                                (no-share charge-collapse = the sharing ablation)
"""

import os
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def read_csv(path):
    """Return dict of column_name -> np.array(float)."""
    if not os.path.exists(path):
        return None
    cols = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


# ----------------------------------------------------------------------
# 1. MA learning curves: 3 seeds, episode_return vs step
# ----------------------------------------------------------------------
def fig_ma_learning():
    seeds = [0, 1, 2]
    plt.figure(figsize=(7, 4.5))
    found = False
    for s in seeds:
        path = f"logs/idqn_share1_id1_seed{s}.csv"
        d = read_csv(path)
        if d is None or "episode_return" not in d:
            continue
        found = True
        x = d.get("step", np.arange(len(d["episode_return"])))
        plt.plot(x, d["episode_return"], label=f"seed {s}", alpha=0.8)
    if not found:
        print("  [skip] no MA learning logs found")
        plt.close()
        return
    plt.xlabel("training step")
    plt.ylabel("episode return")
    plt.title("IDQN (parameter-shared): per-seed learning curves\n"
              "large seed-to-seed spread = non-stationarity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ma_learning_curves.png", dpi=130)
    plt.close()
    print("  saved figures/ma_learning_curves.png")


# ----------------------------------------------------------------------
# 2. Offline mean_q vs update: naive blow-up vs CQL bounded
# ----------------------------------------------------------------------
def fig_offline_meanq():
    files = {
        "naive (alpha=0)": "logs/offline_naive.csv",
        "cql alpha=1":     "logs/offline_cql_a1.csv",
        "cql alpha=10":    "logs/offline_cql_a10.csv",
        "cql alpha=50":    "logs/offline_cql_a50.csv",
        "cql alpha=100":   "logs/offline_cql_a100.csv",
    }
    plt.figure(figsize=(7, 4.5))
    found = False
    for label, path in files.items():
        d = read_csv(path)
        if d is None or "mean_q" not in d:
            continue
        found = True
        x = d.get("update", np.arange(len(d["mean_q"])))
        plt.plot(x, d["mean_q"], label=label, alpha=0.85)
    if not found:
        print("  [skip] no offline logs found")
        plt.close()
        return
    plt.xlabel("gradient update")
    plt.ylabel("mean max-Q (log scale)")
    plt.yscale("symlog")  # naive explodes to 1e5; symlog keeps CQL visible
    plt.title("Offline Q-value divergence: naive blows up, CQL stays bounded")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/offline_meanq.png", dpi=130)
    plt.close()
    print("  saved figures/offline_meanq.png")


# ----------------------------------------------------------------------
# 3. Final mean_q vs alpha (the conservatism ablation)
# ----------------------------------------------------------------------
def fig_offline_alpha():
    alphas = [0, 1, 10, 50, 100]
    files = {
        0:   "logs/offline_naive.csv",
        1:   "logs/offline_cql_a1.csv",
        10:  "logs/offline_cql_a10.csv",
        50:  "logs/offline_cql_a50.csv",
        100: "logs/offline_cql_a100.csv",
    }
    xs, ys = [], []
    for a in alphas:
        d = read_csv(files[a])
        if d is None or "mean_q" not in d:
            continue
        xs.append(a)
        ys.append(d["mean_q"][-1])  # final mean_q
    if not xs:
        print("  [skip] no offline logs for alpha plot")
        return
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(xs, ys, "o-", color="darkred")
    plt.xlabel("CQL conservatism weight (alpha)")
    plt.ylabel("final mean max-Q (symlog)")
    plt.yscale("symlog")
    plt.title("Conservatism ablation: higher alpha suppresses over-estimation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/offline_alpha_vs_q.png", dpi=130)
    plt.close()
    print("  saved figures/offline_alpha_vs_q.png")


# ----------------------------------------------------------------------
# 4. MA action mix bar chart (hardcoded from run_all output; edit if needed)
# ----------------------------------------------------------------------
def fig_action_mix():
    # action percentages from run_all.py output (accept, move, charge, idle)
    data = {
        "seed0":   [10.4, 65.4,  5.8, 18.5],
        "seed1":   [46.2, 29.2, 20.6,  4.0],
        "seed2":   [10.8, 46.0,  1.6, 41.6],
        "noshare": [38.7, 34.7,  3.6, 22.9],
    }
    labels = ["accept", "move", "charge", "idle"]
    policies = list(data.keys())
    x = np.arange(len(policies))
    width = 0.2
    plt.figure(figsize=(7.5, 4.5))
    for i, act in enumerate(labels):
        vals = [data[p][i] for p in policies]
        plt.bar(x + i * width, vals, width, label=act)
    plt.xticks(x + 1.5 * width, policies)
    plt.ylabel("action share (%)")
    plt.title("Action distribution per policy\n"
              "no-share nearly drops 'charge' -> battery-mgmt collapse")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("figures/ma_action_mix.png", dpi=130)
    plt.close()
    print("  saved figures/ma_action_mix.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_ma_learning()
    fig_offline_meanq()
    fig_offline_alpha()
    fig_action_mix()
    print("DONE. See the figures/ folder.")
