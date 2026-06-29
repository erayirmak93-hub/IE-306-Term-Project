from __future__ import annotations
import argparse, csv, os, random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import yaml
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'drone_dispatch_env'))
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate

def flatten_obs(obs, cfg):
    drones = obs["drones"].flatten().astype(np.float32)
    orders = obs["orders"].flatten().astype(np.float32)
    time_  = obs["time"].flatten().astype(np.float32)
    out = np.concatenate([drones, orders, time_])
    return np.clip(out / 50.0, -5.0, 5.0)

def obs_dim(cfg):
    return cfg.n_drones * (4 + 5 + 1) + cfg.k_max * 5 + 1

class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, ns, done):
        self.buf.append((s, int(a), float(r), ns, float(done)))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*batch)
        return (np.array(s,dtype=np.float32), np.array(a,dtype=np.int64),
                np.array(r,dtype=np.float32), np.array(ns,dtype=np.float32),
                np.array(d,dtype=np.float32))
    def __len__(self): return len(self.buf)

class WorldModel:
    def __init__(self, maxlen=10000):
        self.data = deque(maxlen=maxlen)
    def push(self, s, a, ns, r):
        self.data.append((s.copy(), int(a), ns.copy(), float(r)))
    def sample(self):
        if not self.data: return None
        return random.choice(self.data)

class DynaQAgent:
    def __init__(self, cfg, hp, device="cpu"):
        self.cfg = cfg
        self.hp = hp
        self.device = torch.device(device)
        self.q_net    = QNetwork(obs_dim(cfg), cfg.n_actions, hp["hidden"]).to(self.device)
        self.q_target = QNetwork(obs_dim(cfg), cfg.n_actions, hp["hidden"]).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        self.opt    = optim.Adam(self.q_net.parameters(), lr=hp["lr"])
        self.replay = ReplayBuffer(hp["buffer_size"])
        self.world  = WorldModel(hp.get("model_capacity", 10000))
        self.eps    = hp["eps_start"]
        self.total_steps = 0
        self.losses = []

    def act(self, obs):
        mask = np.asarray(obs["action_mask"], dtype=bool)
        if random.random() < self.eps:
            return int(np.random.choice(np.flatnonzero(mask)))
        s = torch.tensor(flatten_obs(obs, self.cfg), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s).squeeze(0).cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def learn(self, obs, action, reward, next_obs, done):
        cfg = self.cfg
        s  = flatten_obs(obs, cfg)
        ns = flatten_obs(next_obs, cfg)
        r  = float(np.clip(reward / 15.0, -3.0, 3.0))
        self.replay.push(s, action, r, ns, float(done))
        self.world.push(s, action, ns, r)
        self.total_steps += 1
        frac = min(1.0, self.total_steps / self.hp["eps_decay_steps"])
        self.eps = self.hp["eps_start"] + frac * (self.hp["eps_end"] - self.hp["eps_start"])
        if len(self.replay) >= self.hp["batch_size"]:
            self.losses.append(self._update(self.replay.sample(self.hp["batch_size"])))
        for _ in range(self.hp["n_planning_steps"]):
            t = self.world.sample()
            if t is None: break
            sim_s, sim_a, sim_ns, sim_r = t
            self._update((sim_s[None], np.array([sim_a],dtype=np.int64),
                          np.array([sim_r],dtype=np.float32),
                          sim_ns[None], np.array([0.0],dtype=np.float32)))
        if self.total_steps % self.hp["target_update"] == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

    def _update(self, batch):
        s,a,r,ns,d = batch
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device)
        with torch.no_grad():
            tgt = r + self.hp["gamma"] * (1-d) * self.q_target(ns).max(1).values
        pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.huber_loss(pred, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"q_net": self.q_net.state_dict(), "hp": self.hp,
                    "total_steps": self.total_steps}, path)

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(cfg, ckpt["hp"], device=device)
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.q_target.load_state_dict(ckpt["q_net"])
        agent.total_steps = ckpt["total_steps"]
        agent.eps = agent.hp["eps_end"]
        return agent

class DynaQPolicy:
    def __init__(self, agent):
        self.agent = agent
        self.agent.eps = 0.0
    def act(self, obs):
        return self.agent.act(obs)

def train(cfg, hp, seed, log_path, weight_path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("DroneDispatch-v0", config=cfg)
    agent = DynaQAgent(cfg, hp)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode","step","ep_return","eps","mean_loss"])
        ep = 0
        obs, info = env.reset(seed=seed)
        ep_return = 0.0
        t0 = time.time()
        while agent.total_steps < hp["total_steps"]:
            action = agent.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent.learn(obs, action, reward, next_obs, done)
            ep_return += reward
            obs = next_obs
            if done:
                ep += 1
                ml = float(np.mean(agent.losses[-50:])) if agent.losses else 0.0
                if ep % 10 == 0:
                    print(f"ep={ep:4d} step={agent.total_steps:6d} return={ep_return:8.1f} eps={agent.eps:.3f} loss={ml:.4f} t={time.time()-t0:.0f}s")
                writer.writerow([ep, agent.total_steps, ep_return, agent.eps, ml])
                f.flush()
                obs, info = env.reset(seed=seed+ep)
                ep_return = 0.0
    agent.save(weight_path)
    print(f"Kaydedildi: {weight_path}")
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dyna_q.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    cfg = Config.from_dict(full_cfg["env"]) if "env" in full_cfg else Config()
    hp  = full_cfg["hyperparameters"]
    train(cfg, hp, seed=args.seed,
          log_path=f"logs/dyna_q_seed{args.seed}.csv",
          weight_path=f"weights/dyna_q_seed{args.seed}.pt")