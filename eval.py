from utils.utils import make_env_with_args, parse_args
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import DDPG, PPO, TD3, A2C
from torch.utils.tensorboard import SummaryWriter
from utils.wrappers import MultiAgentEnvWrapper
from matplotlib import pyplot as plt
from network import CustomTD3Policy
from env.aquarium import Aquarium
from argparse import Namespace
from datetime import datetime
from pathlib import Path
import numpy as np
import pygame
import torch
import time
import os


def run(args: Namespace, model_path: Path, n_runs: int = 10):
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("insight_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Overrriding Show GUI
    args.show_gui = 'Laptop' in torch.cuda.get_device_name(0)

    env = make_env_with_args(Aquarium, args)
    env = MultiAgentEnvWrapper(env, args)
    writer = SummaryWriter(log_dir=str(log_dir))

    if os.path.exists(model_path):
        model = TD3.load(model_path, env=env, device="cuda", custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "policy_class": CustomTD3Policy,
        })
        print('Found Model')
    else:
        print('Did not find the model')
        model = TD3('MlpPolicy', env=env, device='cpu')

    average_total_reward = []
    average_episode_length = []

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
            if args.show_gui:
                img = env.render(render_mode=args.eval_render_mode)
                writer.add_image(f"game_play/run_{run_idx}", img, global_step=step, dataformats='HWC')

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
    args.max_sharks = 1

    env = make_env_with_args(Aquarium, args)
    env = MultiAgentEnvWrapper(env, args)
    writer = SummaryWriter(log_dir=str(log_dir))


    WIDTH, HEIGHT = 1200, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visulization")

    model = TD3(
            policy=CustomTD3Policy,
            env=env, verbose=0,
            device="cuda",
            learning_rate=args.learning_rate,
            tau=0.01,
            policy_kwargs={
                "net_arch":
                    {
                        "pi": {
                            "net_arch": [args.pi_hidden_size] * args.pi_network_depth,
                            "key_net_arch": [args.key_hidden_size] * args.key_network_depth,
                            "msg_net_arch": [args.msg_hidden_size] * args.msg_network_depth,
                            "key_dim": args.key_dim,
                            "msg_dim": args.msg_dim,
                            "activation": args.activation,
                        },
                        "qf": [args.qf_hidden_size] * args.qf_network_depth,
                    }
                }
            )

    average_total_reward = []
    average_episode_length = []

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

            action = np.zeros((1,2))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action[0,0] = 1.
                    elif event.key == pygame.K_LEFT:
                        action[0, 1] = -1.0
                    elif event.key == pygame.K_RIGHT:
                        action[0, 1] = 1.0

            keys = pygame.key.get_pressed()


            if keys[pygame.K_SPACE]:
                action[0, 0] = 1.
            if keys[pygame.K_LEFT]:
                action[0, 1] = -1.0
            if keys[pygame.K_RIGHT]:
                action[0, 1] = 1.0

            print("Updated action:", action)

            obs, reward, done, trunc, _ = env.step(action)


            # Update the NumPy array (example: random noise each frame)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # Create four random NumPy arrays (example images)
            img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img3 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img4 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=100)

            axes[0, 0].imshow(img1); axes[0, 0].set_title("Image 1"); axes[0, 0].axis('off')
            axes[0, 1].imshow(img2); axes[0, 1].set_title("Image 2"); axes[0, 1].axis('off')
            axes[1, 0].imshow(img3); axes[1, 0].set_title("Image 3"); axes[1, 0].axis('off')
            axes[1, 1].imshow(img4); axes[1, 1].set_title("Image 4"); axes[1, 1].axis('off')

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

            screen.blit(surface, (WIDTH // 2, 0))

            # Update only the right half
            pygame.display.update(pygame.Rect(WIDTH // 2, 0, WIDTH // 2, HEIGHT))

            pygame.display.flip()

            # Add images with run index in tag name
            # writer.add_image(f"attention/run_{run_idx}", model.policy.actor.mu.last_attention, global_step=step, dataformats='CHW')
            # writer.add_image(f"key/run_{run_idx}", model.policy.actor.mu.last_k, global_step=step, dataformats='CHW')
            # writer.add_image(f"query/run_{run_idx}", model.policy.actor.mu.last_q, global_step=step, dataformats='CHW')
            # writer.add_image(f"message/run_{run_idx}", model.policy.actor.mu.last_v, global_step=step, dataformats='CHW')

            # norms = np.linalg.norm(model.policy.actor.mu.last_attention, axis=1).flatten()

            # Log each norm as a scalar
            # for i in range(len(norms)):
            #     writer.add_scalar(f'attention/agent_{i}', norms[i], global_step=step)


            time.sleep(0.01)
            if args.show_gui:
                img = env.render(render_mode=args.eval_render_mode)
                writer.add_image(f"game_play/run_{run_idx}", img, global_step=step, dataformats='HWC')

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
    model_name = 'lr_1e3_pk'

    save_path = f'models/{model_name}.zip'

    play(args, save_path)
