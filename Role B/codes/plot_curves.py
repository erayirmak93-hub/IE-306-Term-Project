import sys
import os


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.chdir(root_dir)

import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os


def load_csv(filepath):
    episodes, returns = [], []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Header'ı atla
        for row in reader:
            episodes.append(float(row[0]))
            returns.append(float(row[1]))
    return np.array(episodes), np.array(returns)


def plot_mean_std(method_name, file_pattern, color, label):
    files = glob.glob(file_pattern)
    if not files:
        print(f"Uyarı: {file_pattern} için dosya bulunamadı!")
        return

    all_returns = []
    episodes = None

    for f in files:
        eps, rets = load_csv(f)
        if episodes is None:
            episodes = eps
        all_returns.append(rets)

    all_returns = np.array(all_returns)
    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)

    plt.plot(episodes, mean_returns, label=label, color=color)
    plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, color=color, alpha=0.2)


def main():
    plt.figure(figsize=(10, 6))

    # REINFORCE, A2C (Norm ON) ve A2C (Norm OFF) grafiklerini çiz
    plot_mean_std("REINFORCE", "logs/reinforce_seed_*.csv", "blue", "REINFORCE (3 Seeds)")
    plot_mean_std("A2C_ON", "logs/a2c_adv_norm_ON_seed_*.csv", "green", "A2C + AdvNorm ON (3 Seeds)")
    plot_mean_std("A2C_OFF", "logs/a2c_adv_norm_OFF_seed_*.csv", "red", "A2C + AdvNorm OFF (Ablation)")

    plt.title("Learning Curves (Mean ± Std over 3 Seeds)")
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.legend()
    plt.grid(True)

    plt.savefig("logs/learning_curves.png")
    print("✅ Grafikler başarıyla 'logs/learning_curves.png' olarak kaydedildi!")


if __name__ == "__main__":
    main()