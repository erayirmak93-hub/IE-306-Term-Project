import sys, os, json, copy
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'drone_dispatch_env'))
sys.path.insert(0, os.path.dirname(__file__))
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate
from dyna_q import DynaQAgent, DynaQPolicy, train

SWEEP = [0, 5, 10, 20, 50]
SEEDS = [0, 1, 2]

with open("configs/dyna_q.yaml") as f:
    full_cfg = yaml.safe_load(f)
cfg = Config()
base_hp = full_cfg["hyperparameters"]

results = {}
for n in SWEEP:
    hp = copy.deepcopy(base_hp)
    hp["n_planning_steps"] = n
    cpoes = []
    for seed in SEEDS:
        tag = f"ablation_n{n}_seed{seed}"
        agent = train(cfg, hp, seed=seed,
                      log_path=f"logs/{tag}.csv",
                      weight_path=f"weights/{tag}.pt")
        policy = DynaQPolicy(agent)
        r = evaluate(policy, cfg, seeds=[10,11,12])
        cpo = r["mean"]["cost_per_order"]
        cpoes.append(cpo)
        print(f"n={n} seed={seed} cpo={cpo:.4f}")
    results[n] = {"mean": float(np.mean(cpoes)), "std": float(np.std(cpoes))}

print("\n=== ABLATION: n_planning_steps ===")
print(f"{'n':>6} | {'mean cpo':>10} | {'std':>8}")
print("-"*30)
for n in SWEEP:
    print(f"{n:>6} | {results[n]['mean']:>10.4f} | {results[n]['std']:>8.4f}")

os.makedirs("logs", exist_ok=True)
with open("logs/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nKaydedildi: logs/ablation_results.json")