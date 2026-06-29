import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate, RandomPolicy, GreedyNearest, MILPRolling
from dyna_q import DynaQAgent, DynaQPolicy
import yaml, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/dyna_q.yaml")
parser.add_argument("--seeds", default="0,1,2")
args = parser.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]

with open(args.config) as f:
    full_cfg = yaml.safe_load(f)
cfg = Config()

policies = {
    "random":         RandomPolicy(cfg, seed=0),
    "greedy_nearest": GreedyNearest(cfg),
    "milp_rolling":   MILPRolling(cfg),
}

if os.path.exists("weights/dyna_q_seed0.pt"):
    agent = DynaQAgent.load("weights/dyna_q_seed0.pt", cfg)
    policies["dyna_q"] = DynaQPolicy(agent)

metrics = ["cost_per_order","success_rate","ontime_rate","episode_return"]
results = {}
for name, policy in policies.items():
    print(f"Evaluating {name}...")
    r = evaluate(policy, cfg, seeds=seeds)
    results[name] = r["mean"]

print("\n=== RESULTS TABLE ===")
print(f"{'Policy':<20}", end="")
for m in metrics:
    print(f"{m:>18}", end="")
print()
print("-"*90)
for name, vals in results.items():
    print(f"{name:<20}", end="")
    for m in metrics:
        print(f"{vals[m]:>18.4f}", end="")
    print()

if "greedy_nearest" in results and "dyna_q" in results:
    g = results["greedy_nearest"]["cost_per_order"]
    d = results["dyna_q"]["cost_per_order"]
    print(f"\nGreedy cpo: {g:.4f} | Dyna-Q cpo: {d:.4f} | Fark: {g-d:+.4f}")