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
from pathlib import Path
import os


def run(args: dict, model_path: Path, n_runs: int = 10):
    env = Aquarium(
        observable_sharks=args.observable_sharks,
        observable_fishes=args.observable_fishes,
        observable_walls=args.observable_walls,
        size=args.size,
        max_steps=args.max_steps,
        max_fish=args.max_fish,
        max_sharks=args.max_sharks,
        torus=args.torus,
        fish_collision=args.fish_collision,
        lock_screen=args.lock_screen,
        seed=args.seed,
        show_gui=True,
        use_global_reward=args.use_global_reward,
    )

    env.select_fish_types(args.num_fish,0,0)
    env.select_shark_types(args.max_sharks)

    env = MultiAgentEnvWrapper(env=env)
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
