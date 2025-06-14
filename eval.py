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
from pathlib import Path


def run(model_path: Path, n_runs: int = 10):
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
        show_gui=True,
        use_global_reward=True,
    )

    env.select_fish_types(2,0,0)
    env.select_shark_types(4)

    env = MultiAgentEnvWrapper(env=env)
    env = FrameStackObservation(env,stack_size=10)


    model = PPO.load(model_path,env=env,device="cpu")
    for k in range(n_runs):
        obs, _ = env.reset()
        i = 0
        rewards = []
        tot_rew = 0

        while not env.unwrapped.is_finished:
            i += 1
            action, _ = model.predict(obs,deterministic=True)
            # action = model.action_space.sample()
            obs, reward, done, trunc , _ = env.step(action)
            time.sleep(0.01)
            if(np.random.random()>0.8):
                env.render()
            rewards.append(reward)
            tot_rew += reward
            if(done):
                break
        print('Somethings is Done with reward',tot_rew)
