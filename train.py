"""Train a PPO agent for differential-drive navigation using Stable-Baselines3."""

import argparse
import os
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import DiffDriveNavEnv


class MetricsCallback(BaseCallback):
    """Logs per-episode metrics (return, success, collision) for later plotting."""

    def __init__(self, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []
        self.episode_collisions: list[bool] = []
        self.episode_timesteps: list[int] = []  # global timestep when episode ended

    def _on_step(self) -> bool:
        # check for completed episodes in the monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_successes.append(info.get("success", False))
                self.episode_collisions.append(info.get("collision", False))
                self.episode_timesteps.append(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        np.savez(
            self.log_dir / "training_metrics.npz",
            returns=np.array(self.episode_returns),
            lengths=np.array(self.episode_lengths),
            successes=np.array(self.episode_successes),
            collisions=np.array(self.episode_collisions),
            timesteps=np.array(self.episode_timesteps),
        )
        if self.verbose:
            print(f"Saved training metrics to {self.log_dir / 'training_metrics.npz'}")


def make_env(render_mode=None):
    env = DiffDriveNavEnv(render_mode=render_mode)
    env = Monitor(env, info_keywords=("success", "collision"))
    return env


def main():
    parser = argparse.ArgumentParser(description="Train PPO on DiffDriveNav")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # training env
    if args.n_envs > 1:
        train_env = SubprocVecEnv([lambda: make_env() for _ in range(args.n_envs)])
    else:
        train_env = make_env()

    # eval env (single instance)
    eval_env = make_env()

    # callbacks
    metrics_cb = MetricsCallback(log_dir=args.log_dir, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=args.net_arch),
        tensorboard_log=args.log_dir,
        seed=args.seed,
        verbose=1,
    )

    print(f"Training PPO for {args.total_timesteps} timesteps...")
    print(f"  Network: {args.net_arch}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.n_epochs}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[metrics_cb, eval_cb],
    )

    # save final model
    final_path = os.path.join(args.checkpoint_dir, "ppo_final")
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")

    # save hyperparameters
    hparams = vars(args)
    with open(os.path.join(args.log_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
