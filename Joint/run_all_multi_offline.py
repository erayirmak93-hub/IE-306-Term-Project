"""
run_all.py
==========
ONE command that loads every trained model and prints the full results table.

It produces two blocks:

  (A) OFFLINE (centralized env, real cost_per_order via the instructor's
      evaluate()): naive / CQL / BC wrapped as Policy objects, compared
      head-to-head with random, greedy_nearest, milp_rolling on the SAME
      metric the assignment grades on (cost_per_order, lower is better).

  (B) MULTI-AGENT (DroneDispatchMA-v0, episode return + action mix):
      the 3 parameter-shared seeds and the no-share ablation, vs random.

Run:
  python code/run_all.py
  python code/run_all.py --config configs/eval_standard.yaml --seeds 0,1,2,3,4
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import drone_dispatch_env as dde
from drone_dispatch_env.config import Config

torch.set_num_threads(8)


# ======================================================================
# networks (must match how each model was trained)
# ======================================================================
class OfflineQ(nn.Module):
    """offline net: 181 -> 256 -> 256 -> 169"""
    def __init__(self, obs_dim=181, n_actions=169, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MAQ(nn.Module):
    """MA net: 67 -> 128 -> 128 -> 4 (59 obs + 8 agent-id)"""
    def __init__(self, obs_dim=67, n_actions=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# (A) OFFLINE  -- wrap a trained Q-net as a Policy, score with evaluate()
# ======================================================================
def flatten_central_obs(obs):
    """match the dataset encoding: drones + orders + time, NO grid -> 181."""
    return np.concatenate([
        np.asarray(obs["drones"]).ravel(),
        np.asarray(obs["orders"]).ravel(),
        np.asarray(obs["time"]).ravel(),
    ]).astype(np.float32)


class OfflinePolicy:
    """Adapts a trained offline Q-net to the frozen Policy interface: act(obs)."""
    def __init__(self, qnet, cfg):
        self.q = qnet
        self.cfg = cfg

    def act(self, obs):
        flat = flatten_central_obs(obs)
        with torch.no_grad():
            qv = self.q(torch.tensor(flat).unsqueeze(0)).squeeze(0).numpy()
        mask = np.asarray(obs.get("action_mask", np.ones_like(qv)))
        # forbid invalid actions: set their Q to -inf so argmax never picks them
        qv = np.where(mask > 0, qv, -1e9)
        return int(np.argmax(qv))


def eval_offline(cfg, seeds):
    print("\n========== (A) OFFLINE  -- cost_per_order (lower is better) ==========")
    rows = []

    # learned offline policies (all CQL alphas, to find the best cost_per_order)
    for tag, fname in [("offline_naive",   "weights/offline_naive.pt"),
                       ("offline_cql_a1",  "weights/offline_cql_a1.pt"),
                       ("offline_cql_a10", "weights/offline_cql_a10.pt"),
                       ("offline_cql_a50", "weights/offline_cql_a50.pt"),
                       ("offline_cql_a100","weights/offline_cql_a100.pt"),
                       ("offline_bc",      "weights/offline_bc.pt")]:
        try:
            q = OfflineQ()
            q.load_state_dict(torch.load(fname, map_location="cpu"))
            q.eval()
            pol = OfflinePolicy(q, cfg)
            res = dde.evaluate(pol, cfg, seeds)
            cpo = res["mean"].get("cost_per_order", float("nan"))
            rows.append((tag, cpo, res["mean"]))
        except Exception as e:
            print(f"  [skip {tag}] {e}")

    # classical baselines for comparison
    for name in ["random", "greedy_nearest", "milp_rolling"]:
        try:
            pol = dde.make_baseline(name, cfg)
            res = dde.evaluate(pol, cfg, seeds)
            cpo = res["mean"].get("cost_per_order", float("nan"))
            rows.append((name, cpo, res["mean"]))
        except Exception as e:
            print(f"  [skip {name}] {e}")

    print(f"\n{'policy':<18}{'cost_per_order':>16}{'success':>10}{'ontime':>10}")
    print("-" * 54)
    for tag, cpo, m in sorted(rows, key=lambda r: r[1]):
        print(f"{tag:<18}{cpo:>16.3f}{m.get('success_rate',float('nan')):>10.3f}"
              f"{m.get('ontime_rate',float('nan')):>10.3f}")
    print("(beating greedy_nearest on cost_per_order is the bar.)")


# ======================================================================
# (B) MULTI-AGENT -- episode return + action mix on DroneDispatchMA-v0
# ======================================================================
def ma_encode(obs, agent_ids, use_id=True):
    out = {}
    for aid in agent_ids:
        v = np.asarray(obs[aid]).ravel().astype(np.float32)
        if use_id:
            oh = np.zeros(len(agent_ids), dtype=np.float32)
            oh[agent_ids.index(aid)] = 1.0
            v = np.concatenate([v, oh])
        out[aid] = v
    return out


def ma_episode(env, seed, agent_ids, policy_fn):
    obs, _ = env.reset(seed=seed)
    enc = ma_encode(obs, agent_ids)
    total, steps, action_hist = 0.0, 0, np.zeros(4, dtype=int)
    done = False
    while not done and steps < 2000:
        actions = {}
        for aid in agent_ids:
            a = policy_fn(aid, enc[aid])
            actions[aid] = a
            action_hist[a] += 1
        obs, rew, term, trunc, _ = env.step(actions)
        enc = ma_encode(obs, agent_ids)
        total += sum(float(rew[a]) for a in agent_ids)
        done = all(bool(term[a]) or bool(trunc[a]) for a in agent_ids)
        steps += 1
    return total, action_hist


def eval_ma(episodes=10):
    print("\n========== (B) MULTI-AGENT  -- episode return (higher is better) ==========")
    env = gym.make("DroneDispatchMA-v0", disable_env_checker=True)
    obs, _ = env.reset(seed=0)
    agent_ids = list(obs.keys())
    n_actions = env.action_space[agent_ids[0]].n
    seeds = list(range(episodes))

    def make_idqn_fn(weight_file):
        q = MAQ()
        q.load_state_dict(torch.load(weight_file, map_location="cpu"))
        q.eval()
        def fn(aid, enc):
            with torch.no_grad():
                return int(q(torch.tensor(enc).unsqueeze(0)).argmax(1).item())
        return fn

    def random_fn(aid, enc):
        return np.random.randint(n_actions)

    runs = {
        "idqn_seed0": "weights/idqn_share1_id1_seed0.pt",
        "idqn_seed1": "weights/idqn_share1_id1_seed1.pt",
        "idqn_seed2": "weights/idqn_share1_id1_seed2.pt",
        "idqn_noshare": "weights/idqn_share0_id1_seed0.pt",
    }

    results = {}
    action_mix = {}
    for tag, wf in runs.items():
        try:
            fn = make_idqn_fn(wf)
            rets, hist = [], np.zeros(4, dtype=int)
            for s in seeds:
                r, h = ma_episode(env, s, agent_ids, fn)
                rets.append(r); hist += h
            results[tag] = (np.mean(rets), np.std(rets))
            action_mix[tag] = hist / hist.sum() * 100
        except Exception as e:
            print(f"  [skip {tag}] {e}")

    # random baseline
    rrets = [ma_episode(env, s, agent_ids, random_fn)[0] for s in seeds]
    results["random"] = (np.mean(rrets), np.std(rrets))

    # shared-seeds aggregate (the 3-seed mean +/- std the assignment asks for)
    shared = [results[t][0] for t in ["idqn_seed0", "idqn_seed1", "idqn_seed2"] if t in results]
    if shared:
        print(f"\n3-seed shared IDQN: mean {np.mean(shared):8.2f}  std {np.std(shared):7.2f}")

    print(f"\n{'policy':<16}{'return_mean':>12}{'return_std':>12}")
    print("-" * 40)
    for tag, (m, s) in results.items():
        print(f"{tag:<16}{m:>12.2f}{s:>12.2f}")

    print(f"\n{'policy':<16}{'accept%':>9}{'move%':>9}{'charge%':>9}{'idle%':>9}")
    print("-" * 52)
    for tag, mix in action_mix.items():
        print(f"{tag:<16}{mix[0]:>9.1f}{mix[1]:>9.1f}{mix[2]:>9.1f}{mix[3]:>9.1f}")
    print("(no-share collapses onto fewer actions -> the param-sharing ablation result.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_standard.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ma_episodes", type=int, default=10)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    eval_offline(cfg, seeds)
    eval_ma(args.ma_episodes)
    print("\nDONE. (A) offline cost_per_order table + (B) multi-agent return/action table.")


if __name__ == "__main__":
    main()
