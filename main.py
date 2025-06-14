# from pipeline import main
from ex import dummy
from env.aquarium import Aquarium
import numpy as np
import time

from stable_baselines3 import DDPG, PPO
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import BaseCallback
from network import CustomTD3Policy
from utils.wrappers import MultiAgentEnvWrapper
from gymnasium.wrappers import FrameStackObservation
from eval import run
from datetime import datetime

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if a new episode started
        infos = self.locals.get("rewards", [])
        print(self.locals)
        print(infos)
        for info in infos:
            if "episode" in info:
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)
                print(f"Step: {self.num_timesteps}, Episode Reward: {reward}")
        return True



env = Aquarium(
    observable_sharks=0,
    observable_fishes=3,
    observable_walls=0,
    size=35,
    max_steps=200,
    max_fish=8,
    max_sharks=4,
    torus=True,
    fish_collision=True,
    lock_screen=False,
    seed=42,
    show_gui=False,
    use_global_reward=True,
)

env.select_fish_types(2,0,0)
env.select_shark_types(4)

env = MultiAgentEnvWrapper(env=env)
env = FrameStackObservation(env,stack_size=10)

log_dir = "./tensorboard_logs/"


model = PPO('MlpPolicy',env=env,verbose=1,tensorboard_log=log_dir,device="cpu",learning_rate=2e-4)
# model = DDPG( policy=CustomTD3Policy,env=env,verbose=1,tensorboard_log=log_dir,device="cpu")

# print("Actor:")
# print(model.policy.actor)

# model = PPO.load("ppo_torus_sharks_with_penalty",env=env)

model.learn(500_000)
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
save_path = 'models/PPO_model_' + now + '.zip'
model.save(save_path)

# model = PPO.load('ppo_torus_sharks_with_penalty.zip',env=env)

run(save_path)