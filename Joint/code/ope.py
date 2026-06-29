"""
ope.py  --  Off-Policy Evaluation (bonus, Section 7)
====================================================
Estimate each offline policy's value FROM THE LOGGED DATASET ONLY -- without
running it in the simulator -- then compare the estimate to the real
cost_per_order / return we measured with the live evaluator. The bonus asks
exactly this: "estimate a policy's value from logs before running it, and
report how close your estimate was."

Two estimators (both standard, both computed purely from the static dataset):

  1. FQE-style direct estimate (Q-based):
     value_est = mean over dataset start-states of  max_a Q_theta(s0, a)
     i.e. the policy's own learned Q at the states it would start from.
     (For naive this is the blown-up Q; for CQL it is the conservative Q.)

  2. Step-wise weighted Importance Sampling (IS):
     reweight the logged rewards by how much the evaluated policy agrees with
     the behavior that produced the data. Pure greedy target policy:
        rho_t = 1[ argmax_a Q(s_t,a) == logged_action_t ] / behavior_prob
     We approximate behavior_prob by the empirical action frequency in the
     dataset (the data came from a mixed behavior policy). This is a rough but
     honest IS estimate of average reward per step.

We then line these up against the actual evaluation numbers (filled in from
run_all.py) and report the gap -> "how close was the estimate".

Run:
  python code/ope.py --dataset team_data.npz
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

import drone_dispatch_env as dde

torch.set_num_threads(8)


class OfflineQ(nn.Module):
    def __init__(self, obs_dim=181, n_actions=169, hidden=256):
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
    obs = np.asarray(d["observations"], dtype=np.float32)
    act = np.asarray(d["actions"], dtype=np.int64)
    rew = np.asarray(d["rewards"], dtype=np.float32)
    term = np.asarray(d["terminals"]).astype(bool)
    tout = np.asarray(d["timeouts"]).astype(bool)
    done = term | tout
    return obs, act, rew, done


def start_states(obs, done):
    """A state is an episode-start if the PREVIOUS transition was 'done'.
    The very first row is also a start."""
    starts = np.zeros(len(obs), dtype=bool)
    starts[0] = True
    starts[1:] = done[:-1]
    return obs[starts]


def fqe_estimate(qnet, obs_starts):
    """Direct Q-based value: mean over start states of max_a Q(s,a)."""
    with torch.no_grad():
        q = qnet(torch.tensor(obs_starts))
        v = q.max(dim=1).values.mean().item()
    return v


def is_estimate(qnet, obs, act, rew):
    """Step-wise weighted IS, greedy target policy, empirical behavior prob.
    Returns an estimate of average reward-per-step under the target policy."""
    with torch.no_grad():
        greedy = qnet(torch.tensor(obs)).argmax(dim=1).numpy()
    # empirical behavior action frequency (smoothed)
    counts = np.bincount(act, minlength=169).astype(np.float64)
    behav_prob = (counts[act] + 1.0) / (counts.sum() + 169.0)
    # target is deterministic greedy: prob 1 on its action, else 0
    agree = (greedy == act).astype(np.float64)
    weights = agree / behav_prob
    if weights.sum() == 0:
        return float("nan")
    # weighted average of logged rewards
    return float((weights * rew).sum() / weights.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="team_data.npz")
    args = ap.parse_args()

    obs, act, rew, done = load_dataset(args.dataset)
    obs_starts = start_states(obs, done)
    print(f"dataset: {len(obs)} transitions, {len(obs_starts)} episode-start states\n")

    policies = [
        ("naive",   "weights/offline_naive.pt"),
        ("cql_a1",  "weights/offline_cql_a1.pt"),
        ("cql_a10", "weights/offline_cql_a10.pt"),
        ("cql_a50", "weights/offline_cql_a50.pt"),
        ("cql_a100","weights/offline_cql_a100.pt"),
        ("bc",      "weights/offline_bc.pt"),
    ]

    # Actual cost_per_order from run_all.py (lower is better). Edit if yours differ.
    actual_cpo = {
        "naive":   25.703,
        "cql_a1":  38.940,
        "cql_a10": 25.388,
        "cql_a50": 33.574,
        "cql_a100":30.260,
        "bc":      25.777,
    }

    rows = []
    for tag, fname in policies:
        try:
            q = OfflineQ()
            q.load_state_dict(torch.load(fname, map_location="cpu"))
            q.eval()
            v_fqe = fqe_estimate(q, obs_starts)
            v_is = is_estimate(q, obs, act, rew)
            rows.append((tag, v_fqe, v_is, actual_cpo.get(tag, float("nan"))))
        except Exception as e:
            print(f"  [skip {tag}] {e}")

    print(f"{'policy':<10}{'FQE_value_est':>15}{'IS_reward/step':>16}{'actual_cost_per_order':>24}")
    print("-" * 65)
    for tag, vf, vi, cpo in rows:
        print(f"{tag:<10}{vf:>15.2f}{vi:>16.4f}{cpo:>24.3f}")

    print("\nInterpretation:")
    print("- FQE_value_est is the policy's OWN predicted value from logs (no sim run).")
    print("  naive's is wildly inflated (over-estimation) -> a BAD predictor, which")
    print("  is itself the finding: an over-optimistic offline value cannot be trusted.")
    print("- IS_reward/step ranks policies by logged reward weighted to the target")
    print("  policy. Compare the IS ranking to the actual cost_per_order ranking:")
    print("  if a higher IS estimate lines up with a lower (better) cost, the log-only")
    print("  estimate was predictive BEFORE we ever ran the policy in the simulator.")


if __name__ == "__main__":
    main()
