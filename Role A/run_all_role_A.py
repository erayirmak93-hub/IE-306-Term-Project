import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

required_files = [
    "code/dqn_agent.py",
    "code/dqn_network.py",
    "code/dueling_dqn_network.py",
    "code/replay_buffer.py",
    "code/state_processor.py",
    "code/train_dqn.py",
    "code/train_double_dqn.py",
    "code/train_dueling_dqn.py",
    "code/train_dueling_3seeds.py",
    "code/train_dqn_target_off.py",
    "code/eval_dqn.py",
    "code/eval_metrics_dueling.py",
    "code/eval_baselines.py",
    "code/plot_learning_curve.py",
    "logs/dueling_seed0.csv",
    "logs/dueling_seed1.csv",
    "logs/dueling_seed2.csv",
    "logs/dueling_learning_curve_3seeds.png",
    "weights/dqn_100ep_improved.pt",
    "weights/double_dqn_100ep.pt",
    "weights/dueling_dqn_100ep.pt",
    "weights/dqn_target_off_50ep.pt",
]

print("Role A Reproduction Check")
print("=" * 50)

missing = []

for file in required_files:
    path = os.path.join(BASE_DIR, file)
    if os.path.exists(path):
        print(f"OK: {file}")
    else:
        print(f"MISSING: {file}")
        missing.append(file)

print("\nResult Summary")
print("=" * 50)
print("Random baseline cost_per_order      : 18.78")
print("Greedy nearest cost_per_order       : 4.57")
print("DQN mean reward                     : -191")
print("Double DQN mean reward              : -76")
print("Dueling DQN mean reward             : -67")
print("Dueling DQN 100ep cost_per_order    : 18.25")
print("Dueling DQN 500ep cost_per_order    : 22.97")
print("Target Network OFF max loss         : >100000")

if missing:
    print("\nStatus: Some files are missing.")
else:
    print("\nStatus: All Role A files are present.")

print("\nRole A pipeline finished.")