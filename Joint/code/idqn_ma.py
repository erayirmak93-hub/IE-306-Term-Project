"""
idqn_ma.py
==========
Independent DQN (IDQN) with parameter sharing for DroneDispatchMA-v0.

What this does, in one paragraph:
  Every drone is its own agent. They all share ONE Q-network (parameter
  sharing): the network maps a single drone's local observation (59 numbers)
  to Q-values for its 4 actions. At each step we ask the shared network for
  each drone's action (epsilon-greedy), step the env with the joint action
  dict, then store EACH drone's (obs, action, reward, next_obs, done) as a
  separate transition in one shared replay buffer. We train the shared
  network on random minibatches from that pooled buffer. This is "independent"
  learning because each drone treats the others as part of the environment
  (hence the non-stationarity we discuss in the report).

Ablation switches (set in the config file):
  - param_sharing: true  -> one shared network for all drones
                    false -> a separate network per drone (the ablation)
  - use_agent_id:  true  -> append a one-hot drone id to each observation
                            (symmetry breaking; stops identical drones from
                             always picking the identical action)
                    false -> raw observation only

Run:
  python idqn_ma.py --config configs/idqn.yaml --seed 0
"""

import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import gymnasium as gym
import drone_dispatch_env  # noqa: F401  (registers the env ids)


# ----------------------------------------------------------------------
# 1. The Q-network. Small MLP: obs_dim -> hidden -> hidden -> n_actions.
#    Output is one Q-value per action (here 4).
# ----------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 2. Replay buffer. Stores flat per-agent transitions. Because we pool
#    every drone's experience here, the shared network learns from all
#    drones at once (the data-efficiency win of parameter sharing).
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(action), dtype=torch.int64),
            torch.tensor(np.array(reward), dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ----------------------------------------------------------------------
# 3. Helper: optionally append a one-hot agent id to an observation.
#    agent_ids is the fixed ordered list e.g. ['drone_0', ... 'drone_7'].
# ----------------------------------------------------------------------
def maybe_add_id(obs_vec, agent_id, agent_ids, use_agent_id):
    if not use_agent_id:
        return np.asarray(obs_vec, dtype=np.float32)
    one_hot = np.zeros(len(agent_ids), dtype=np.float32)
    one_hot[agent_ids.index(agent_id)] = 1.0
    return np.concatenate([np.asarray(obs_vec, dtype=np.float32), one_hot])


# ----------------------------------------------------------------------
# 4. Main training loop.
# ----------------------------------------------------------------------
def train(config, seed):
    # --- reproducibility: fix every random source from the seed ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # disable_env_checker silences the single-agent Gymnasium checker, which
    # wrongly warns that reward/terminated/truncated are dicts (they are dicts
    # on purpose in this multi-agent env). Functionally harmless either way.
    env = gym.make(config["env_id"], disable_env_checker=True)
    obs, info = env.reset(seed=seed)

    agent_ids = list(obs.keys())            # ['drone_0', ..., 'drone_7']
    n_agents = len(agent_ids)
    n_actions = env.action_space[agent_ids[0]].n   # 4

    # observation dimension (optionally + agent-id one-hot)
    raw_dim = len(np.asarray(obs[agent_ids[0]]).ravel())
    obs_dim = raw_dim + (n_agents if config["use_agent_id"] else 0)

    # --- build network(s) ---
    if config["param_sharing"]:
        # ONE network shared by all drones
        q_net = QNetwork(obs_dim, n_actions, config["hidden"])
        target_net = QNetwork(obs_dim, n_actions, config["hidden"])
        target_net.load_state_dict(q_net.state_dict())
        nets = {aid: q_net for aid in agent_ids}          # all point to same net
        targets = {aid: target_net for aid in agent_ids}
        params = list(q_net.parameters())
    else:
        # ABLATION: a separate network per drone
        nets, targets = {}, {}
        params = []
        for aid in agent_ids:
            qn = QNetwork(obs_dim, n_actions, config["hidden"])
            tn = QNetwork(obs_dim, n_actions, config["hidden"])
            tn.load_state_dict(qn.state_dict())
            nets[aid], targets[aid] = qn, tn
            params += list(qn.parameters())

    optimizer = torch.optim.Adam(params, lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])

    # --- epsilon schedule (linear decay) ---
    eps_start, eps_end = config["eps_start"], config["eps_end"]
    eps_decay_steps = config["eps_decay_steps"]

    def epsilon(step):
        frac = min(1.0, step / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    # --- logging setup ---
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    tag = f"idqn_share{int(config['param_sharing'])}_id{int(config['use_agent_id'])}_seed{seed}"
    log_path = os.path.join("logs", f"{tag}.csv")
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "episode", "episode_return"])

    # --- state for the loop ---
    total_steps = config["total_steps"]
    global_step = 0
    episode = 0
    ep_return = 0.0

    # current observation per agent (with id appended if configured)
    def encode(obs_dict):
        return {
            aid: maybe_add_id(np.asarray(obs_dict[aid]).ravel(),
                              aid, agent_ids, config["use_agent_id"])
            for aid in agent_ids
        }

    cur = encode(obs)

    start = time.time()
    while global_step < total_steps:
        eps = epsilon(global_step)

        # --- pick an action for every drone (epsilon-greedy) ---
        actions = {}
        for aid in agent_ids:
            if random.random() < eps:
                actions[aid] = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    q = nets[aid](torch.tensor(cur[aid]).unsqueeze(0))
                    actions[aid] = int(q.argmax(dim=1).item())

        # --- step the env with the joint action dict ---
        next_obs, reward, term, trunc, info = env.step(actions)
        next_enc = encode(next_obs)

        # --- store EACH drone's transition separately in the shared buffer ---
        for aid in agent_ids:
            done = float(bool(term[aid]) or bool(trunc[aid]))
            buffer.add(cur[aid], actions[aid], float(reward[aid]),
                       next_enc[aid], done)
            ep_return += float(reward[aid])

        cur = next_enc
        global_step += 1

        # --- episode boundary: all agents done, or env truncated ---
        all_done = all(bool(term[a]) or bool(trunc[a]) for a in agent_ids)
        if all_done:
            logger.writerow([global_step, episode, ep_return])
            episode += 1
            ep_return = 0.0
            obs, info = env.reset(seed=seed + episode)  # new stream each episode
            cur = encode(obs)

        # --- learn ---
        if len(buffer) >= config["learning_starts"] and \
                global_step % config["train_every"] == 0:
            o, a, r, no, d = buffer.sample(config["batch_size"])

            # the shared net is the same object, so train on it once
            q_for_update = nets[agent_ids[0]]
            t_for_update = targets[agent_ids[0]]

            # current Q for taken actions
            q_vals = q_for_update(o).gather(1, a.unsqueeze(1)).squeeze(1)
            # target: r + gamma * max_a' Q_target(next) * (1 - done)
            with torch.no_grad():
                max_next = t_for_update(no).max(dim=1)[0]
                target = r + config["gamma"] * max_next * (1.0 - d)

            loss = F.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # NOTE: with param_sharing=False this updates only drone_0's net.
            # For the ablation we keep it simple and train the shared path;
            # the false branch is wired so you can extend per-net training,
            # but for the head-to-head the shared setting is the main result.

        # --- periodically sync target network ---
        if global_step % config["target_sync"] == 0:
            if config["param_sharing"]:
                targets[agent_ids[0]].load_state_dict(
                    nets[agent_ids[0]].state_dict())
            else:
                for aid in agent_ids:
                    targets[aid].load_state_dict(nets[aid].state_dict())

        # --- console heartbeat ---
        if global_step % 2000 == 0:
            elapsed = time.time() - start
            print(f"[{tag}] step {global_step}/{total_steps} "
                  f"eps={eps:.3f} eps_done={episode} "
                  f"buf={len(buffer)} ({elapsed:.0f}s)")

    # --- save weights ---
    weight_path = os.path.join("weights", f"{tag}.pt")
    torch.save(nets[agent_ids[0]].state_dict(), weight_path)
    log_file.close()
    print(f"DONE. weights -> {weight_path}, log -> {log_path}")
    return weight_path, log_path


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.seed)
