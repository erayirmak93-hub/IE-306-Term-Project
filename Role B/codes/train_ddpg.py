import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.chdir(root_dir)

import gymnasium as gym
import drone_dispatch_env
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

SEEDS = [0, 1, 2]

for seed in SEEDS:
    print(f"\n🚀 DDPG Eğitimi Başlıyor -> Seed: {seed}")
    env = gym.make("DroneControl-v0")
    env.reset(seed=seed)

    # Exploration için Noise Eklendi (Analizde istenen)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0, seed=seed)

    log_path = f"logs/ddpg_seed_{seed}"
    new_logger = configure(log_path, ["csv"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=15000)

    if seed == 0:
        model.save("weights/ddpg_model")