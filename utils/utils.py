from stable_baselines3 import  TD3
from argparse import ArgumentParser
from network import CustomTD3Policy
from datetime import datetime
from gymnasium import Env
from pathlib import Path
import zipfile
import torch
import shutil
import time
import os


def get_dump_dir(wandb_run_id):
    if not os.path.exists("dumps"):
        os.makedirs("dumps")
    if wandb_run_id:
        dump_path = f"dumps/{wandb_run_id}.h5"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dump_path = f"dumps/{timestamp}.h5"
    return dump_path

def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def check_model_path(input_string):
    # Convert the input to a Path object
    if not input_string:
        return False
    path = Path(input_string)


    # Check and modify the path components
    if path.parts[0] != 'models':
        path = Path('models') / path
        if path.suffix != '.zip':
            path = path.parent / (path.name + '.zip')

    print(path)

    if not path.exists():
        print('No Model found to load')
        return False
    return str(path)

def companion_coeff_mapper(value):
    # Apply your desired function to the value here
    # For example, let's say you want to square the input value
    transformed_value = 2 * float(value) * float(value)
    return transformed_value

def make_env_with_args(Env: Env, args):
    env = Env(
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
        show_gui=args.show_gui,
        use_global_reward=args.use_global_reward,
        companion_coeff = args.companion_coeff,
        bump_penalty = args.bump_penalty,
        direction_with_angle = args.angle,
        coop=args.coop,
        allow_stun_move = args.stun,
        observe_time = args.observe_time,
        observe_position = args.observe_position,
    )

    env.select_fish_types(args.n_random_fish, args.n_turnaway_fish,0)
    env.select_shark_types(args.max_sharks)

    return env

def load_model_actor(env, model_path, args):
    model = TD3(
            policy=CustomTD3Policy,
            env=env, verbose=0,
            device="cuda",
            learning_rate=args.learning_rate,
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

    print(model_path)

    if os.path.exists(model_path):
        extract_dir = "tmp_actor_extract"
        os.makedirs(extract_dir, exist_ok=True)

        # Extract policy.pth from the zip
        with zipfile.ZipFile(model_path, 'r') as archive:
            archive.extractall(extract_dir)

        policy_path = os.path.join(extract_dir, "policy.pth")
        policy_dict = torch.load(policy_path, map_location="cuda")

        actor_state_dict = {k.replace("actor.", ""): v for k, v in policy_dict.items() if k.startswith("actor.")}
        model.policy.actor.load_state_dict(actor_state_dict)
        print('Found Model and loaded policy')

        shutil.rmtree(extract_dir)
    else:
        print('Did not find the model Initializating random policy')

    return model

def parse_args():
    parser = ArgumentParser(description="Environment configuration")

    parser.add_argument('--observable_sharks', type=int, default=0, help='Number of observable sharks')
    parser.add_argument('--observable_fishes', type=int, default=2, help='Number of observable fishes')
    parser.add_argument('--observable_walls', type=int, default=0, help='Number of observable walls')
    parser.add_argument('--size', type=int, default=30, help='Size of the environment')
    parser.add_argument('--max_steps', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--max_fish', type=int, default=8, help='Maximum number of fish')
    parser.add_argument('--max_sharks', type=int, default=4, help='Maximum number of sharks')
    parser.add_argument('--load_model_path','-l', type=str, default=None, help='Load the Given model')
    parser.add_argument('--save_model_path','-s', type=str, default=None, help='Save model path')
    parser.add_argument('--models_dir','-d', type=str, default=None, help='Model saving base dicrectory')
    parser.add_argument('--total_timesteps','-t', type=int, default=200_000, help='Total number of training timesteps')
    parser.add_argument('--stack_size', type=int, default=1, help='Stack size for frame stacking')
    parser.add_argument('--num_envs', '-ne', type=int, default=16, help='Number of Parallel environments in gymnasium')
    parser.add_argument('--n_random_fish', '-rf', type=int, default=1, help='Number of Random Fish in the sea')
    parser.add_argument('--n_turnaway_fish','-tf',type=int, default=3, help='Number of Turnaway fish in the sea')
    parser.add_argument('--n_static_fish', '-sf', type=int, default=0, help='Number of Static fish in the sea')
    parser.add_argument('--rnn_hidden_state_dim', type=int, default=32, help='Dimension of RNN hidden state')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help='Learning rate for the model')
    parser.add_argument('--activation', '-a', type=str, default='tanh', help='Activation function for the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the model')
    parser.add_argument('--eval_render_mode','-erm', type=str, choices=['human', 'rgb_array', 'rgb_array_only'], default='rgb_array', help='Render mode for evaluation')
    parser.add_argument('--n_eval_runs', type=int, default=5, help='Number of evaluation runs')

    parser.add_argument('--torus', action='store_true', help='Enable toroidal world (wrap around edges)')
    parser.add_argument('--no_torus', dest='torus', action='store_false', help='Disable toroidal world')
    parser.set_defaults(torus=True)


    # Co-op Params
    parser.add_argument('--bump_penalty', '-bp', type=float, default=-0.01, help='Penalty on bumping into fish on its own')
    parser.add_argument('--companion_coeff', '-cc', type=companion_coeff_mapper, default=15., help='Radius Multiplier for companionship')

    parser.add_argument('--train', action='store_true', help='Enable training')
    parser.add_argument('--eval', dest='train', action='store_false', help='Only Evaluation')
    parser.set_defaults(train=True)

    parser.add_argument('--angle', action='store_true', help='Use Angle in the agent obvservation')
    parser.add_argument('--vector', dest='angle', action='store_false', help='Use sin cos vector as agent observation')
    parser.set_defaults(angle=True)

    parser.add_argument('--fish_collision', action='store_true', help='Enable fish collision')
    parser.add_argument('--no_fish_collision', dest='fish_collision', action='store_false', help='Disable fish collision')
    parser.set_defaults(fish_collision=True)

    parser.add_argument('--lock_screen', action='store_true', help='Enable lock screen')
    parser.add_argument('--no_lock_screen', dest='lock_screen', action='store_false', help='Disable lock screen')
    parser.set_defaults(lock_screen=False)

    parser.add_argument('--show_gui', action='store_true', help='Show GUI')
    parser.add_argument('--no_show_gui', dest='show_gui', action='store_false', help='Hide GUI')
    parser.set_defaults(show_gui=False)

    parser.add_argument('--use_global_reward', action='store_true', help='Use global reward')
    parser.add_argument('--no_use_global_reward', dest='use_global_reward', action='store_false', help='Do not use global reward')
    parser.set_defaults(use_global_reward=True)

    parser.add_argument('--wandb',action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--no_wandb', dest='wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.set_defaults(wandb=False)

    parser.add_argument('--coop', action='store_true', help='Enable cooperative mode')
    parser.add_argument('--no_coop', dest='coop', action='store_false', help='Disable cooperative mode')
    parser.set_defaults(coop=True)

    parser.add_argument('--stun', action='store_true', help='Enable stun move')
    parser.add_argument('--no_stun', dest='stun', action='store_false', help='Disable stun move')
    parser.set_defaults(stun=False)

    parser.add_argument('--observe_time', action='store_true', help='Enable time observation')
    parser.add_argument('--no_observe_time', dest='observe_time', action='store_false', help='Disable time observation')
    parser.set_defaults(observe_time=False)

    parser.add_argument('--observe_position', action='store_true', help='Enable position observation')
    parser.add_argument('--no_observe_position', dest='observe_position', action='store_false', help='Disable position observation')
    parser.set_defaults(observe_position=False)

    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode (user controlled agent)')
    parser.add_argument('--no_interactive', dest='interactive', action='store_false', help='Disable interactive mode')
    parser.set_defaults(interactive=False)


    ## Netowrk Params
    parser.add_argument('--pi_hidden_size', type=int, default=64, help='Hidden size for the pi network')
    parser.add_argument('--pi_network_depth', type=int, default=1, help='Depth of the pi network')

    parser.add_argument('--qf_hidden_size', type=int, default=64, help='Hidden size for the qf network')
    parser.add_argument('--qf_network_depth', type=int, default=1, help='Depth of the qf network')

    parser.add_argument('--key_hidden_size', type=int, default=16, help='Hidden size for the key network')
    parser.add_argument('--key_network_depth', type=int, default=3, help='Depth of the key network')

    parser.add_argument('--msg_hidden_size', type=int, default=16, help='Hidden size for the msg network')
    parser.add_argument('--msg_network_depth', type=int, default=2, help='Depth of the msg network')

    parser.add_argument('--key_dim', type=int, default=4, help='Key dimension')
    parser.add_argument('--msg_dim', type=int, default=16, help='Message dimension')


    return parser.parse_args()
