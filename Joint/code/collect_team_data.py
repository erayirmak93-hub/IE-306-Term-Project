"""
collect_team_data.py
====================
Rolls out the THREE teammates' trained policies (roles A/B/C) on the centralized
env (DroneDispatch-v0) and logs their transitions in OUR offline dataset format,
then merges with the existing offline_data.npz to build the mixed-quality,
team-sourced dataset the assignment asks for (Ch.20).

Each model sees the observation in ITS OWN format (the way it was trained):
  - DQN (A):       581-dim, grid included, no normalize
  - A2C (B):       581-dim, grid included, normalized /20,/20,/3
  - REINFORCE (B): same 581 as A2C
  - DynaQ (C):     181-dim, grid excluded, normalized /50 + clip
...but every transition is SAVED in our canonical offline format:
  - 181-dim: drones + orders + time, NO grid, NO normalize
so it lines up with offline_data.npz and the CQL/naive training code.

No training happens here -- models are frozen, we just run episodes and record.

Run:
  python code/collect_team_data.py --episodes 8
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import drone_dispatch_env  # noqa: F401

torch.set_num_threads(8)


# ---------------------------------------------------------------------------
# observation encoders (one per training recipe)
# ---------------------------------------------------------------------------
def enc_581_raw(obs):
    """DQN (A): drones+orders+grid+time, no normalize -> 581"""
    return np.concatenate([
        obs["drones"].flatten(),
        obs["orders"].flatten(),
        obs["grid"].flatten(),
        np.atleast_1d(obs["time"]).flatten(),
    ]).astype(np.float32)


def enc_581_norm(obs):
    """A2C / REINFORCE (B): grid included, normalized /20,/20,/3 -> 581"""
    return np.concatenate([
        obs["drones"].flatten() / 20.0,
        obs["orders"].flatten() / 20.0,
        obs["grid"].flatten() / 3.0,
        np.atleast_1d(obs["time"]).flatten(),
    ]).astype(np.float32)


def enc_181_dyna(obs):
    """DynaQ (C): drones+orders+time, no grid, /50 + clip -> 181"""
    out = np.concatenate([
        obs["drones"].flatten(),
        obs["orders"].flatten(),
        np.atleast_1d(obs["time"]).flatten(),
    ]).astype(np.float32)
    return np.clip(out / 50.0, -5.0, 5.0)


def enc_181_canonical(obs):
    """OUR offline save format: drones+orders+time, no grid, no normalize -> 181"""
    return np.concatenate([
        obs["drones"].flatten(),
        obs["orders"].flatten(),
        np.atleast_1d(obs["time"]).flatten(),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# networks (must match teammates' code exactly)
# ---------------------------------------------------------------------------
class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(581, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 169),
        )
    def forward(self, x): return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(581, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, 169)
        self.critic = nn.Linear(256, 1)
    def forward(self, x):
        h = self.shared_net(x)
        return self.actor(h), self.critic(h)


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(581, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 169),
        )
    def forward(self, x): return self.net(x)


class DynaQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(181, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 169),
        )
    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------------------
# build an act(obs)->action function for each model
# ---------------------------------------------------------------------------
def make_act(kind, path):
    if kind == "dqn":
        net = DQNNet(); net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()
        enc = enc_581_raw
        def act(obs):
            mask = np.asarray(obs["action_mask"])
            with torch.no_grad():
                q = net(torch.tensor(enc(obs)).unsqueeze(0)).squeeze(0).numpy()
            q[mask == 0] = -1e9
            return int(np.argmax(q))
        return act

    if kind == "a2c":
        net = ActorCritic(); net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()
        enc = enc_581_norm
        def act(obs):
            mask = np.asarray(obs["action_mask"], dtype=bool)
            with torch.no_grad():
                logits, _ = net(torch.tensor(enc(obs)).unsqueeze(0))
                logits = logits.squeeze(0).numpy()
            logits[~mask] = -1e9
            return int(np.argmax(logits))
        return act

    if kind == "reinforce":
        net = PolicyNet(); net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()
        enc = enc_581_norm
        def act(obs):
            mask = np.asarray(obs["action_mask"], dtype=bool)
            with torch.no_grad():
                logits = net(torch.tensor(enc(obs)).unsqueeze(0)).squeeze(0).numpy()
            logits[~mask] = -1e9
            return int(np.argmax(logits))
        return act

    if kind == "dyna":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = DynaQNet(); net.load_state_dict(ck["q_net"]); net.eval()
        enc = enc_181_dyna
        def act(obs):
            mask = np.asarray(obs["action_mask"])
            with torch.no_grad():
                q = net(torch.tensor(enc(obs)).unsqueeze(0)).squeeze(0).numpy()
            q[mask == 0] = -1e9
            return int(np.argmax(q))
        return act

    raise ValueError(kind)


# ---------------------------------------------------------------------------
# roll out one policy, record transitions in canonical 181 format
# ---------------------------------------------------------------------------
def rollout(act_fn, episodes, base_seed=5000):
    env = gym.make("DroneDispatch-v0", disable_env_checker=True)
    O, A, R, NO, TERM, TOUT = [], [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            a = act_fn(obs)
            s = enc_181_canonical(obs)
            next_obs, r, term, trunc, info = env.step(a)
            O.append(s)
            A.append(a)
            R.append(float(r))
            NO.append(enc_181_canonical(next_obs))
            TERM.append(bool(term))
            TOUT.append(bool(trunc))
            obs = next_obs
            done = term or trunc
    return (np.array(O, dtype=np.float32), np.array(A, dtype=np.int64),
            np.array(R, dtype=np.float32), np.array(NO, dtype=np.float32),
            np.array(TERM, dtype=bool), np.array(TOUT, dtype=bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8,
                    help="episodes PER model (more = bigger dataset)")
    ap.add_argument("--out", default="team_data.npz")
    ap.add_argument("--merge_with", default="offline_data.npz")
    ap.add_argument("--merged_out", default="offline_data_team.npz")
    args = ap.parse_args()

    # the 4 teammate models (edit paths if yours differ)
    models = [
        ("dqn",       "weights/dqn_100ep_improved.pt"),
        ("a2c",       "weights/a2c_OFF_seed0.pt"),
        ("reinforce", "weights/reinforce_seed0.pt"),
        ("dyna",      "weights/dyna_q_seed0.pt"),
    ]

    allO, allA, allR, allNO, allTERM, allTOUT = [], [], [], [], [], []
    for kind, path in models:
        try:
            act = make_act(kind, path)
            o, a, r, no, term, tout = rollout(act, args.episodes)
            print(f"  {kind:10s} {path:35s} -> {len(a):6d} transitions, "
                  f"mean_reward={r.mean():+.4f}")
            allO.append(o); allA.append(a); allR.append(r)
            allNO.append(no); allTERM.append(term); allTOUT.append(tout)
        except Exception as e:
            print(f"  [skip {kind}] {e}")

    if not allO:
        print("No team data collected. Check weight paths.")
        return

    team = dict(
        observations=np.concatenate(allO),
        actions=np.concatenate(allA),
        rewards=np.concatenate(allR),
        next_observations=np.concatenate(allNO),
        terminals=np.concatenate(allTERM),
        timeouts=np.concatenate(allTOUT),
    )
    np.savez(args.out, **team)
    print(f"\nteam-only dataset saved: {args.out} ({len(team['actions'])} transitions)")

    # merge with the existing baseline dataset
    try:
        import drone_dispatch_env as dde
        base = dde.load_offline_dataset(args.merge_with)
        keys = ["observations", "actions", "rewards", "next_observations",
                "terminals", "timeouts"]
        merged = {}
        for k in keys:
            b = np.asarray(base[k])
            t = team[k]
            # match dtypes
            if b.dtype != t.dtype:
                t = t.astype(b.dtype)
            merged[k] = np.concatenate([b, t])
        np.savez(args.merged_out, **merged)
        print(f"MERGED dataset saved: {args.merged_out} "
              f"({len(merged['actions'])} transitions = "
              f"{len(base['actions'])} baseline + {len(team['actions'])} team)")
    except Exception as e:
        print(f"[merge skipped] {e}")
        print(f"You can still train on the team-only file: {args.out}")


if __name__ == "__main__":
    main()
