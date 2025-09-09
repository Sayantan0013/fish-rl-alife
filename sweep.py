import argparse
import random
from typing import Dict, Any
from argparse import Namespace

from main import train
import numpy as np
import torch
import wandb


# -----------------------------
# Repro (optional)
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_run(config: Dict[str, Any] = None):
    with wandb.init(config=config):
        cfg = wandb.config
        cfg['wandb_run_id']= wandb.run.id

        set_seed(cfg['seed'])
        average_total_reward, average_episode_length = train(Namespace(**cfg))
        wandb.summary["average_total_reward"] = average_total_reward
        wandb.summary["average_episode_length"] = average_episode_length


# -----------------------------
# Sweep definition
# -----------------------------
def sweep_config(project: str):
    """
    Sweep configuration for your network with key and message modules.
    """
    return {
        "name": "multi-module-net-sweep",
        "method": "bayes",  # or "random", "grid"
        "metric": {"name": "average_total_reward", "goal": "maximize"},
        "parameters": {
            # Main Actor network
            "pi_hidden_size": {"values": [32, 64, 128, 256]},
            "pi_network_depth": {"values": [1, 2, 3, 4]},

            # Main Critic network
            "qf_hidden_size": {"values": [32, 64, 128, 256]},
            "qf_network_depth": {"values": [1, 2, 3, 4]},

            # Key network
            "key_hidden_size": {"values": [8, 16, 32]},
            "key_network_depth": {"values": [1, 2, 3]},

            # Message network
            "msg_hidden_size": {"values": [16, 32, 64]},
            "msg_network_depth": {"values": [1, 2, 3]},

            # Dimensions
            "key_dim": {"values": [4, 8, 12]},
            "msg_dim": {"values": [8, 16, 32]},

            # Optimization
            "learning_rate": {
                "min": 1e-5,
                "max": 5e-3,
                "distribution": "log_uniform_values"
            },

            # Activation
            "activation": {
                "values": ["relu", "tanh", "sigmoid", "leaky_relu", "elu"]
            },
            # === Fixed RL Defaults ===
            "seed": {"value": 42},
            "total_timesteps": {"value": 500_000},
            "stack_size": {"value": 1},
            "num_envs": {"value": 16},
            "rnn_hidden_state_dim": {"value": 32},

            # === Environment Defaults ===
            "observable_sharks": {"value": 0},
            "observable_fishes": {"value": 2},
            "observable_walls": {"value": 0},
            "size": {"value": 30},
            "max_steps": {"value": 300},
            "max_fish": {"value": 8},
            "max_sharks": {"value": 4},
            "n_random_fish": {"value": 1},
            "n_turnaway_fish": {"value": 3},

            # === Reward & Coop Params ===
            "bump_penalty": {"value": -0.01},
            "companion_coeff": {"value": 18.0},

            # === Flags (as fixed values) ===
            "torus": {"value": True},
            "train": {"value": True},
            "angle": {"value": False},
            "fish_collision": {"value": True},
            "lock_screen": {"value": False},
            "show_gui": {"value": False},
            "use_global_reward": {"value": True},
            "coop": {"value": False},
            "stun": {"value": False},
            "wandb": {"value": True},  # Enable logging

            # === Paths (keep None) ===
            "load_model_path": {"value": None},
            "save_model_path": {"value": None},
            "models_dir": {"value": None},

            # === Rendering ===
            "sync_tensorboard": {"value": True},
            "monitor_gym": {"value": False},
            "save_code": {"value": False},
            "n_eval_runs": {"value": 5},
            "eval_render_mode": {"value": "rgb_array_only"}
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10
        }
    }

# -----------------------------
# Entrypoint: create + run sweep
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="fish-rl-sweep", help="W&B project name")
    parser.add_argument("--count", type=int, default=30, help="Number of runs this agent will execute")
    parser.add_argument("--resume_sweep", type=str, default=None, help="Existing sweep_id to resume (skip creation)")
    args = parser.parse_args()

    sweep_cfg = sweep_config(args.project)

    if args.resume_sweep:
        sweep_id = args.resume_sweep
        print(f"[wandb] Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.project)
        print(f"[wandb] Created sweep: {sweep_id}")

    # Launch the local agent that will pull configs and run `train_one_run`
    wandb.agent(sweep_id, function=train_one_run, count=args.count)


if __name__ == "__main__":
    main()
