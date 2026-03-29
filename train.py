"""Train PPO or REINFORCE on the DiffDriveNav environment.

Usage:
    python train.py --algo ppo --total-timesteps 1000000
    python train.py --algo reinforce --total-timesteps 1000000
"""

import argparse
import json
import os
import time

import numpy as np

from env import DiffDriveNavEnv
from agents import PPOAgent, ReinforceAgent, RolloutBuffer


def collect_rollout(env, agent, buffer, n_steps):
    """Collect n_steps of experience into buffer. Returns episode stats."""
    obs, _ = getattr(collect_rollout, "_state", (None, None))
    if obs is None:
        obs, _ = env.reset()

    ep_returns, ep_lengths, ep_successes, ep_collisions = [], [], [], []
    ep_reward = getattr(collect_rollout, "_ep_reward", 0.0)
    ep_len = getattr(collect_rollout, "_ep_len", 0)

    for _ in range(n_steps):
        action, log_prob, value = agent.select_action(obs)
        clipped = np.clip(action, -1.0, 1.0)
        next_obs, reward, terminated, truncated, info = env.step(clipped)
        done = terminated or truncated

        # store original action (not clipped) to match log_prob
        buffer.store(obs, action, log_prob, reward, value, float(done))
        ep_reward += reward
        ep_len += 1

        if done:
            ep_returns.append(ep_reward)
            ep_lengths.append(ep_len)
            ep_successes.append(info.get("success", False))
            ep_collisions.append(info.get("collision", False))
            ep_reward = 0.0
            ep_len = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

    # save state for next call
    collect_rollout._state = (obs, None)
    collect_rollout._ep_reward = ep_reward
    collect_rollout._ep_len = ep_len

    return {
        "returns": ep_returns,
        "lengths": ep_lengths,
        "successes": ep_successes,
        "collisions": ep_collisions,
        "last_obs": obs,
    }


def collect_episodes(env, agent, buffer, n_episodes):
    """Collect full episodes for REINFORCE. Returns episode stats."""
    ep_returns, ep_lengths, ep_successes, ep_collisions = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            clipped = np.clip(action, -1.0, 1.0)
            next_obs, reward, terminated, truncated, info = env.step(clipped)
            done = terminated or truncated
            buffer.store(obs, action, log_prob, reward, value, float(done))
            ep_reward += reward
            obs = next_obs

        ep_returns.append(ep_reward)
        ep_lengths.append(info["step"])
        ep_successes.append(info.get("success", False))
        ep_collisions.append(info.get("collision", False))

    return {
        "returns": ep_returns,
        "lengths": ep_lengths,
        "successes": ep_successes,
        "collisions": ep_collisions,
    }


def train_ppo(env, args):
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_range,
        entropy_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
    )

    all_returns, all_lengths, all_successes, all_collisions, all_timesteps = [], [], [], [], []
    total_steps = 0

    # reset rollout collector state
    for attr in ("_state", "_ep_reward", "_ep_len"):
        if hasattr(collect_rollout, attr):
            delattr(collect_rollout, attr)
    n_updates = args.total_timesteps // args.n_steps
    start = time.time()

    for update in range(1, n_updates + 1):
        buffer = RolloutBuffer()
        stats = collect_rollout(env, agent, buffer, args.n_steps)
        total_steps += args.n_steps

        # get value of last observation for GAE bootstrap
        with __import__("torch").no_grad():
            _, _, next_val = agent.select_action(stats["last_obs"])

        train_stats = agent.update(buffer, next_val)

        # log
        for r in stats["returns"]:
            all_returns.append(r)
            all_timesteps.append(total_steps)
        all_lengths.extend(stats["lengths"])
        all_successes.extend(stats["successes"])
        all_collisions.extend(stats["collisions"])

        if update % 10 == 0 and stats["returns"]:
            recent_r = all_returns[-50:] if len(all_returns) >= 50 else all_returns
            recent_s = all_successes[-50:] if len(all_successes) >= 50 else all_successes
            fps = total_steps / (time.time() - start)
            print(
                f"[PPO] step={total_steps:>8d} | "
                f"ep_ret={np.mean(recent_r):>7.1f} | "
                f"success={np.mean(recent_s)*100:>5.1f}% | "
                f"pg_loss={train_stats['pg_loss']:>.4f} | "
                f"entropy={train_stats['entropy']:>.3f} | "
                f"fps={fps:>.0f}"
            )

    return agent, all_returns, all_lengths, all_successes, all_collisions, all_timesteps


def train_reinforce(env, args):
    agent = ReinforceAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.ent_coef,
        hidden=args.hidden,
    )

    all_returns, all_lengths, all_successes, all_collisions, all_timesteps = [], [], [], [], []
    total_steps = 0
    start = time.time()
    episode = 0

    while total_steps < args.total_timesteps:
        # collect single episode
        buffer = RolloutBuffer()
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            clipped = np.clip(action, -1.0, 1.0)
            next_obs, reward, terminated, truncated, info = env.step(clipped)
            done = terminated or truncated
            buffer.store(obs, action, log_prob, reward, value, float(done))
            ep_reward += reward
            obs = next_obs

        ep_len = info["step"]
        total_steps += ep_len
        episode += 1

        # update after each episode
        train_stats = agent.update(buffer)

        all_returns.append(ep_reward)
        all_timesteps.append(total_steps)
        all_lengths.append(ep_len)
        all_successes.append(info.get("success", False))
        all_collisions.append(info.get("collision", False))

        if episode % 50 == 0:
            recent_r = all_returns[-100:] if len(all_returns) >= 100 else all_returns
            recent_s = all_successes[-100:] if len(all_successes) >= 100 else all_successes
            fps = total_steps / (time.time() - start)
            print(
                f"[REINFORCE] step={total_steps:>8d} ep={episode:>5d} | "
                f"ep_ret={np.mean(recent_r):>7.1f} | "
                f"success={np.mean(recent_s)*100:>5.1f}% | "
                f"pg_loss={train_stats['pg_loss']:>.4f} | "
                f"entropy={train_stats['entropy']:>.3f} | "
                f"fps={fps:>.0f}"
            )

    return agent, all_returns, all_lengths, all_successes, all_collisions, all_timesteps


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on DiffDriveNav")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "reinforce"])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    # PPO-specific
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    # REINFORCE-specific
    parser.add_argument("--episodes-per-update", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = DiffDriveNavEnv()
    np.random.seed(args.seed)
    __import__("torch").manual_seed(args.seed)

    print(f"Training {args.algo.upper()} for {args.total_timesteps} timesteps...")

    if args.algo == "ppo":
        agent, returns, lengths, successes, collisions, timesteps = train_ppo(env, args)
    else:
        agent, returns, lengths, successes, collisions, timesteps = train_reinforce(env, args)

    env.close()

    # save model
    model_path = os.path.join(args.checkpoint_dir, f"{args.algo}_final.pt")
    agent.save(model_path)
    print(f"Saved model to {model_path}")

    # save metrics
    metrics_path = os.path.join(args.log_dir, f"{args.algo}_metrics.npz")
    np.savez(
        metrics_path,
        returns=np.array(returns),
        lengths=np.array(lengths),
        successes=np.array(successes),
        collisions=np.array(collisions),
        timesteps=np.array(timesteps),
    )
    print(f"Saved metrics to {metrics_path}")

    # save hparams
    with open(os.path.join(args.log_dir, f"{args.algo}_hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
