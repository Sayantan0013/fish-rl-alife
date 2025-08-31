from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import DDPG, PPO, TD3, A2C
from torch.utils.tensorboard import SummaryWriter
from utils.wrappers import MultiAgentEnvWrapper
from utils.utils import make_env_with_args
from network import CustomTD3Policy
from env.aquarium import Aquarium
from datetime import datetime
from pathlib import Path
import numpy as np
import time
import os


def run(args: dict, model_path: Path, n_runs: int = 10):
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("insight_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env_with_args(Aquarium, args)
    env = MultiAgentEnvWrapper(env, args)
    writer = SummaryWriter(log_dir=str(log_dir))

    if os.path.exists(model_path):
        model = TD3.load(model_path, env=env, device="cpu", custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "policy_class": CustomTD3Policy,
        })
        print('Found Model')
    else:
        print('Did not find the model')
        model = TD3('MlpPolicy', env=env, device='cpu')

    for run_idx in range(n_runs):
        obs, _ = env.reset()
        rewards = []
        tot_rew = 0
        step = 0

        while not env.unwrapped.is_finished:
            if os.path.exists(model_path):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = model.action_space.sample()

            obs, reward, done, trunc, _ = env.step(action)

            # Add images with run index in tag name
            writer.add_image(f"attention/run_{run_idx}", model.policy.actor.mu.last_attention, global_step=step, dataformats='CHW')
            writer.add_image(f"key/run_{run_idx}", model.policy.actor.mu.last_k, global_step=step, dataformats='CHW')
            writer.add_image(f"query/run_{run_idx}", model.policy.actor.mu.last_q, global_step=step, dataformats='CHW')
            writer.add_image(f"message/run_{run_idx}", model.policy.actor.mu.last_v, global_step=step, dataformats='CHW')

            norms = np.linalg.norm(model.policy.actor.mu.last_attention, axis=1).flatten()

            # Log each norm as a scalar
            for i in range(len(norms)):
                writer.add_scalar(f'attention/agent_{i}', norms[i], global_step=step)


            time.sleep(0.01)

            img = env.render(render_mode=args.eval_render_mode)
            writer.add_image(f"game_play/run_{run_idx}", img, global_step=step, dataformats='HWC')

            rewards.append(reward)
            tot_rew += reward
            step += 1
            if done:
                break

        print(f'Run {run_idx} completed with total reward {tot_rew}')
