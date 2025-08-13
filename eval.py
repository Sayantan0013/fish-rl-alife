from env.aquarium import Aquarium
import numpy as np
import time

from stable_baselines3 import DDPG, PPO, TD3
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import BaseCallback
from network import CustomTD3Policy
from utils.wrappers import MultiAgentEnvWrapper
from gymnasium.wrappers import FrameStackObservation
from utils.utils import make_env_with_args
from pathlib import Path
import os


def run(args: dict, model_path: Path, n_runs: int = 10):
    env = make_env_with_args(Aquarium, args)

    env = MultiAgentEnvWrapper(env=env)
    # env = FrameStackObservation(MultiAgentEnvWrapper(env),stack_size=args.stack_size)
    # env = FrameStackObservation(env,stack_size=10)


    if os.path.exists(model_path):
        # model = PPO.load(model_path,env=env,device="cpu")
        model = TD3.load(model_path,env=env,device="cpu", custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "policy_class": CustomTD3Policy,
    })
        print('Found Model')

    else:   
        print('Did not found the model')
        model = PPO('MlpPolicy',env=env,device='cpu')

    for k in range(n_runs):
        obs, _ = env.reset()
        i = 0
        rewards = []
        tot_rew = 0

        while not env.unwrapped.is_finished:
            i += 1
            if os.path.exists(model_path):
                action, _ = model.predict(obs,deterministic=True)
            else:
                action = model.action_space.sample()

            obs, reward, done, trunc , _ = env.step(action)
            time.sleep(0.01)
            # if(np.random.random()>0.8):
            env.render()
            rewards.append(reward)
            tot_rew += reward
            if(done):
                break
        print('Somethings is Done with reward',tot_rew)
