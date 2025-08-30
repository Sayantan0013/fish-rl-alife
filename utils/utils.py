from sys import flags
import time
from gymnasium import Env
from argparse import ArgumentParser
from pathlib import Path

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
    path = Path(input_string)

    # Check and modify the path components
    if path.parts[0] != 'models':
        path = Path('models') / path
    if path.suffix != '.zip':
        path = path.with_suffix('.zip')

    if not path.exists():
        print('No Model found to load')
        return False
    return str(path)

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
    )

    env.select_fish_types(args.n_random_fish, args.n_turnaway_fish,0)
    env.select_shark_types(args.max_sharks)

    return env

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
    parser.add_argument('--num_envs', '-ne', type=int, default=8, help='Number of Parallel environments in gymnasium')
    parser.add_argument('--n_random_fish', '-rf', type=int, default=1, help='Number of Random Fish in the sea')
    parser.add_argument('--n_turnaway_fish','-tf',type=int, default=3, help='Number of Turnaway fish in the sea')
    parser.add_argument('--rnn_hidden_state_dim', type=int, default=32, help='Dimension of RNN hidden state')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the model')

    parser.add_argument('--torus', action='store_true', help='Enable toroidal world (wrap around edges)')
    parser.add_argument('--no_torus', dest='torus', action='store_false', help='Disable toroidal world')
    parser.set_defaults(torus=True)


    # Co-op Params
    parser.add_argument('--bump_penalty', '-bp', type=float, default=-0.01, help='Penalty on bumping into fish on its own')
    parser.add_argument('--companion_coeff', '-cc', type=float, default=5., help='Radius Multiplier for companionship')

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

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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

    ## Netowrk Params
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for the network')
    parser.add_argument('--network_depth', type=int, default=2, help='Depth of the network')

    return parser.parse_args()
