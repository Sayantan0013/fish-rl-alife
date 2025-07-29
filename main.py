# from pipeline import main
from env.aquarium import Aquarium
import numpy as np
import time
import os

from stable_baselines3 import DDPG, PPO, TD3
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.callbacks import BaseCallback
from network import CustomTD3Policy
from utils.wrappers import MultiAgentEnvWrapper
from utils.utils import make_env_with_args, parse_args
# from utils.callbacks import WandbCallback
from gymnasium.wrappers import FrameStackObservation
from eval import run
from datetime import datetime
    

if __name__ == "__main__":

    args = parse_args()
    
    if args.train:
        
        wandb.init(
            project="fish-marl",
            config={"algo": "TD3"},
            sync_tensorboard=True,   # You don't want tensorboard
            monitor_gym=False,        # You don't want video
            save_code=False,
        )
        
        wandb.config.update(args)

        # env = MultiAgentEnvWrapper(make_env_with_args(Aquarium, args))
        
        def make_envs_from_base(base_env_class, num_envs):
            def make_env():
                env_copy = make_env_with_args(base_env_class, args)
                return MultiAgentEnvWrapper(env_copy)
            return [make_env for _ in range(num_envs)]

        envs = make_envs_from_base(Aquarium,8)
        env = SubprocVecEnv(envs)
        env = VecMonitor(env)

        # env = FrameStackObservation(env,stack_size=10)

        log_dir = "./tensorboard_logs/"
        # new_logger = configure(folder=None, format_strings=["wandb"]) 
        
        
        if args.load_model_path and os.path.exists(args.load_model_path):
            model = TD3.load(args.load_model_path,env=env,device="cpu", custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "policy_class": CustomTD3Policy,
            })
            print(f'successfully loaded the model {args.load_model_path}')
        else:
            # model = PPO('MlpPolicy',env=env,verbose=1,tensorboard_log=log_dir,device="cpu")
            model = TD3(
                    policy=CustomTD3Policy, 
                    env=env, verbose=0, 
                    tensorboard_log=log_dir, 
                    device="cuda",
                    )

        # model.set_logger(new_logger)
        model.learn(args.total_timesteps, progress_bar=True, callback=WandbCallback())

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = 'models/PPO_model_' + now + '.zip'
        model.save(save_path)
    else:
        save_path = 'models/PPO_model_2025-07-29_04-11-25.zip'

    run(args, save_path, n_runs = 5)