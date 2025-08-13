
from typing import Dict, Optional
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if a new episode started
        infos = self.locals.get("rewards", [])
        print(self.locals)
        print(infos)
        for info in infos:
            if "episode" in info:
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)
                print(f"Step: {self.num_timesteps}, Episode Reward: {reward}")
        return True

class WandbCallback(BaseCallback):
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}

    def _on_training_start(self) -> None:
        """Initialize wandb at the start of training."""

        import wandb

        wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            sync_tensorboard=True,
        )

    def _on_step(self) -> bool:
        """Log metrics at each step."""
        import wandb

        # Log training metrics

        if self.locals.get("dones", None) is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    episode_info = self.locals["infos"][i]
                    wandb.log(
                        {
                            "train/episode_length": episode_info.get("episode", {}).get(
                                "l", 0
                            ),
                            "train/episode_reward": episode_info.get("episode", {}).get(
                                "r", 0
                            ),
                            "train/global_step": self.num_timesteps,
                        }
                    )

        return True

    def _on_rollout_end(self) -> None:
        """Log rollout metrics."""
        import wandb
        print(self.locals.keys())
        print("/n/n/n")
        wandb.log(
            {
                "train/rollout_ep_len_mean": self.locals.get("ep_info_buffer", {}).get(
                    "l", 0
                ),
                "train/rollout_ep_rew_mean": self.locals.get("ep_info_buffer", {}).get(
                    "r", 0
                ),
                "train/rollout_ep_len_std": self.locals.get("ep_info_buffer", {}).get(
                    "l_std", 0
                ),
                "train/rollout_ep_rew_std": self.locals.get("ep_info_buffer", {}).get(
                    "r_std", 0
                ),
            }
        )

    def _on_training_end(self) -> None:
        """Close wandb at the end of training."""

        import wandb

        wandb.finish()
