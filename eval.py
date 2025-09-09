from utils.utils import get_dump_dir, make_env_with_args, parse_args, load_model_actor
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import DDPG, PPO, TD3, A2C
from torch.utils.tensorboard import SummaryWriter
from utils.wrappers import MultiAgentEnvWrapper
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from utils.loggers import log_step_to_hdf5
from utils.iteractive import get_interactive_action
from env.aquarium import Aquarium
from argparse import Namespace
from datetime import datetime
from pathlib import Path
import numpy as np
import pygame
import pickle
import torch
import time
import os


def run(args: Namespace, model_path: Path, n_runs: int = 10):
    # Create timestamped log directory
    # Overrriding Show GUI
    args.show_gui = 'Laptop' in torch.cuda.get_device_name(0)

    env = make_env_with_args(Aquarium, args)
    env = MultiAgentEnvWrapper(env, args)

    model = load_model_actor(env, model_path, args)

    average_total_reward = []
    average_episode_length = []
    dump_path = get_dump_dir(args.wandb_run_id)


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

            new_obs, reward, done, trunc, _ = env.step(action)

            time.sleep(0.002)
            env.render(render_mode=args.eval_render_mode)

            data_dict = {
                        "attention": model.policy.actor.mu.last_attention,
                        "key": model.policy.actor.mu.last_k,
                        "query": model.policy.actor.mu.last_q,
                        "value": model.policy.actor.mu.last_v,
                        "obs": obs,
                        "reward": reward,
            }

            obs = new_obs
            log_step_to_hdf5(dump_path, run_idx, step, data_dict)

            rewards.append(reward)
            tot_rew += reward
            step += 1
            if done:
                break

        print(f'Run {run_idx} completed with total reward {tot_rew}')
        average_total_reward.append(tot_rew)
        average_episode_length.append(step)

    return np.mean(average_total_reward), np.mean(average_episode_length)



def play(args: Namespace, model_path: str, n_runs: int = 1):

    plt.switch_backend('Agg')

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("insight_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    args.show_gui = True
    # args.max_sharks = 1

    env = make_env_with_args(Aquarium, args)
    env = MultiAgentEnvWrapper(env, args)
    writer = SummaryWriter(log_dir=str(log_dir))


    WIDTH, HEIGHT = 1200, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visulization")

    model = load_model_actor(env, model_path, args)

    average_total_reward = []
    average_episode_length = []

    for run_idx in range(n_runs):
        obs, _ = env.reset()
        rewards = []
        tot_rew = 0
        step = 0

        base_points = []
        pcas = []
        for idx, attr in enumerate(['key', 'query', 'value']):
            data_root = 'models/embeddings'
            base_points.append(np.load(f'{data_root}/data/{attr}.npy'))

            pca_filename =f'{data_root}/{attr}.pkl'
            pcas.append(pickle.load(open(pca_filename,'rb')))

            # base_points.append(pcas[idx].transform(np.load(f'{data_root}/raw_data/{attr}.npy')))

        while not env.unwrapped.is_finished:
            if os.path.exists(model_path):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = model.action_space.sample()

            if args.interactive:
                action = get_interactive_action(action)


            obs, reward, done, trunc, _ = env.step(action)


            # Update the NumPy array (example: random noise each frame)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            fig = plt.figure(figsize=(6, 6), dpi=100)
            gs = gridspec.GridSpec(2, 2, figure=fig)

            # Create four random NumPy arrays (example images)
            for idx, attr in enumerate(['key', 'query', 'value']):
                if idx < 2:
                    ax = fig.add_subplot(gs[idx//2, idx%2])
                    ax.set_xlim(-2, 2)
                    ax.set_ylim(-2, 2)
                elif idx == 2:
                    ax = fig.add_subplot(gs[idx//2, 0])
                    ax.set_xlim(-8, 8)
                    ax.set_ylim(-8, 8)
                elif idx == 3:
                    ax = fig.add_subplot(gs[2:, :])
                cur_data = np.nan

                format_type = np.uint8
                if attr == 'key':
                    # print(model.policy.actor.mu.last_k.min(),model.policy.actor.mu.last_k.max())
                    cur_data = model.policy.actor.mu.last_k.astype(format_type)
                    # print(cur_data.min(),cur_data.max())
                elif attr == 'query':
                    cur_data = model.policy.actor.mu.last_q.astype(format_type)
                elif attr == 'value':
                    cur_data = model.policy.actor.mu.last_v.astype(format_type)
                elif attr == 'attention':
                    cur_data = model.policy.actor.mu.last_attention

                cur_data = cur_data[0]

                # pca_filename =f'{data_root}/{attr}.pkl'
                # pca = pickle.load(open(pca_filename,'rb'))

                transformed_data = pcas[idx].transform(cur_data)

                ax.scatter(base_points[idx][:, 0], base_points[idx][:, 1], s=1) # 's' controls the size of the points

                ax.scatter(transformed_data[:, 0], transformed_data[:, 1], s=10) # 's' controls the size of the points

                # ax.imshow(cur_data)

                ax.set_title(f"{attr.capitalize()}")
                ax.axis('off')


            plt.tight_layout()

            # Draw the canvas
            fig.canvas.draw()

            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            combined_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            array = combined_array.reshape(600, 600, 4)[:, :, :3]  # (H, W, 3)


            frame[:,WIDTH//2:] = array
            # Convert NumPy array to Pygame surface
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            # swapaxes because Pygame expects (width, height) order

            # Draw the surface
            #
            screen.blit(surface, (WIDTH // 2, 0))

            # Update only the right half
            pygame.display.update(pygame.Rect(WIDTH // 2, 0, WIDTH // 2, HEIGHT))

            pygame.display.flip()


            time.sleep(0.005)
            if args.show_gui:
                img = env.render(render_mode=args.eval_render_mode)
                # writer.add_image(f"game_play/run_{run_idx}", img, global_step=step, dataformats='HWC')

            rewards.append(reward)
            tot_rew += reward
            step += 1
            if done:
                break

        print(f'Run {run_idx} completed with total reward {tot_rew}')
        average_total_reward.append(tot_rew)
        average_episode_length.append(step)

    # cv2.destroyAllWindows()
    return np.mean(average_total_reward), np.mean(average_episode_length)

if __name__ == "__main__":
    args = parse_args()

    save_path = f'models/{args.load_model_path}.zip'

    play(args, save_path, n_runs = 10)
