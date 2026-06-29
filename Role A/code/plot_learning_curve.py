import os
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = r"C:\Users\Berk\Desktop\IE306_drone_dispatch_env"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

files = [
    "dueling_seed0.csv",
    "dueling_seed1.csv",
    "dueling_seed2.csv"
]

all_rewards = []

for file in files:
    rewards = []

    with open(os.path.join(LOG_DIR, file), "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rewards.append(float(row["reward"]))

    all_rewards.append(rewards)

all_rewards = np.array(all_rewards)

mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

episodes = np.arange(1, len(mean_rewards) + 1)

plt.figure(figsize=(10,5))

plt.plot(
    episodes,
    mean_rewards,
    linewidth=2,
    label="Mean Reward"
)

plt.fill_between(
    episodes,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    alpha=0.3,
    label="±1 Std"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Dueling DQN Learning Curve (3 Seeds)")
plt.legend()
plt.grid(True)

save_path = os.path.join(
    LOG_DIR,
    "dueling_learning_curve_3seeds.png"
)

plt.savefig(save_path, dpi=300)

print("Saved:", save_path)

plt.show()