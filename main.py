from argparse import Namespace
from utils.utils import make_env_with_args, parse_args, check_model_path
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import DDPG, PPO, TD3, A2C
from utils.wrappers import MultiAgentEnvWrapper
from wandb.integration.sb3 import WandbCallback
# from utils.callbacks import WandbCallback
from network import CustomTD3Policy

from env.aquarium import Aquarium
from datetime import datetime
from pathlib import Path
from eval import run
import wandb



def train(args):
    if args.train:

        # env = MultiAgentEnvWrapper(make_env_with_args(Aquarium, args))

        def make_envs_from_base(base_env_class, num_envs):
            def make_env():
                env_copy = make_env_with_args(base_env_class, args)
                return MultiAgentEnvWrapper(env_copy, args)
                # return FrameStackObservation(MultiAgentEnvWrapper(env_copy),stack_size=args.stack_size)

            return [make_env for _ in range(num_envs)]

        envs = make_envs_from_base(Aquarium, args.num_envs)
        env = SubprocVecEnv(envs)
        env = VecMonitor(env)

        # env = FrameStackObservation(env,stack_size=10)

        log_dir = "./tensorboard_logs/"

        if model_path := check_model_path(args.load_model_path):
            model = TD3.load(model_path,env=env,device="cuda", custom_objects={
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

        print(f'Model observation shape: {model.observation_space.shape}')
        print(f"Model action space: {model.action_space.shape}")
        model.learn(args.total_timesteps, progress_bar=True, callback=WandbCallback())

        # if not save_path :=
        #     now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     save_path = 'models/TD3_model_' + now + '.zip'

        save_path =  Path(f'models/{args.save_model_path}').with_suffix('.zip')
        model.save(save_path)
    else:
        save_path = check_model_path(args.load_model_path)


    args.show_gui = True
    average_total_reward, average_episode_length = run(args, save_path, n_runs = args.n_eval_runs)

    print(f"Average total reward: {average_total_reward:.2f}")
    print(f"Average episode length: {average_episode_length:.2f}")

    return average_total_reward, average_episode_length



if __name__ == "__main__":
    args = parse_args()

    wandb.init(
        project="fish-marl-cur-experiment",
        config={"algo": "TD3"},
        sync_tensorboard=True,   # You don't want tensorboard
        monitor_gym=False,        # You don't want video
        save_code=False,
        mode="disabled" if not args.wandb else "online",
    )

    wandb.config.update(args)

    train(args)
