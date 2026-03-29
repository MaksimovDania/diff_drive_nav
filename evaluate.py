"""Evaluate trained agents and generate comparison plots."""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from env import DiffDriveNavEnv
from agents import PPOAgent, ReinforceAgent


# ──────────────────────────────────────────────────────────────────────
# Agent loader
# ──────────────────────────────────────────────────────────────────────

def load_agent(algo: str, checkpoint: str, obs_dim: int, act_dim: int, hidden: int = 64):
    if algo == "ppo":
        agent = PPOAgent(obs_dim, act_dim, hidden=hidden)
    else:
        agent = ReinforceAgent(obs_dim, act_dim, hidden=hidden)
    agent.load(checkpoint)
    return agent


# ──────────────────────────────────────────────────────────────────────
# 1. Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_policy(agent, n_episodes: int = 100, seed: int = 0):
    env = DiffDriveNavEnv()
    results = {
        "returns": [], "lengths": [], "successes": [], "collisions": [],
        "trajectories": [], "goals": [], "starts": [],
    }

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        trajectory = [(env.x, env.y)]
        total_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append((env.x, env.y))
            done = terminated or truncated

        results["returns"].append(total_reward)
        results["lengths"].append(len(trajectory) - 1)
        results["successes"].append(info.get("success", False))
        results["collisions"].append(info.get("collision", False))
        results["trajectories"].append(np.array(trajectory))
        results["goals"].append((env.goal_x, env.goal_y))
        results["starts"].append(trajectory[0])

    env.close()

    sr = np.mean(results["successes"]) * 100
    cr = np.mean(results["collisions"]) * 100
    ar = np.mean(results["returns"])
    al = np.mean(results["lengths"])

    print(f"  Success rate:   {sr:.1f}%")
    print(f"  Collision rate: {cr:.1f}%")
    print(f"  Avg return:     {ar:.2f}")
    print(f"  Avg length:     {al:.1f} steps")

    return results


# ──────────────────────────────────────────────────────────────────────
# 2. Training curves
# ──────────────────────────────────────────────────────────────────────

def smooth(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(metrics_path: str, label: str, save_dir: str, color: str):
    data = np.load(metrics_path)
    returns = data["returns"]
    successes = data["successes"].astype(float)
    timesteps = data["timesteps"]
    window = min(50, len(returns) // 5) if len(returns) > 10 else 1
    return returns, successes, timesteps, window


def plot_comparison_curves(metrics_paths: dict, save_dir: str):
    """Plot training curves for multiple algorithms on the same axes."""
    os.makedirs(save_dir, exist_ok=True)
    colors = {"ppo": "steelblue", "reinforce": "orangered"}
    labels = {"ppo": "PPO", "reinforce": "REINFORCE"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for algo, path in metrics_paths.items():
        if not os.path.exists(path):
            continue
        data = np.load(path)
        returns = data["returns"]
        successes = data["successes"].astype(float)
        timesteps = data["timesteps"]
        color = colors.get(algo, "gray")
        label = labels.get(algo, algo)
        window = min(50, len(returns) // 5) if len(returns) > 10 else 1

        # reward
        ax = axes[0]
        ax.plot(timesteps, returns, alpha=0.08, color=color, linewidth=0.5)
        if len(returns) > window:
            sm = smooth(returns, window)
            ax.plot(timesteps[window - 1:], sm, color=color, linewidth=2, label=label)

        # success rate
        ax = axes[1]
        if len(successes) > window:
            sm = smooth(successes, window) * 100
            ax.plot(timesteps[window - 1:], sm, color=color, linewidth=2, label=label)

    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Episode Return")
    axes[0].set_title("Training Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Training Success Rate")
    axes[1].set_ylim(-5, 105)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved comparison curves to {path}")


# ──────────────────────────────────────────────────────────────────────
# 3. Trajectory plots
# ──────────────────────────────────────────────────────────────────────

def plot_trajectories(results: dict, save_dir: str, algo_label: str, n_plots: int = 5):
    os.makedirs(save_dir, exist_ok=True)
    trajectories = results["trajectories"]
    goals = results["goals"]
    successes = results["successes"]

    success_idx = [i for i, s in enumerate(successes) if s]
    fail_idx = [i for i, s in enumerate(successes) if not s]

    n_success = min(max(n_plots - 1, n_plots // 2), len(success_idx))
    n_fail = min(n_plots - n_success, len(fail_idx))
    n_success = min(n_plots - n_fail, len(success_idx))

    rng = np.random.default_rng(42)
    selected = []
    if success_idx:
        selected.extend(rng.choice(success_idx, size=n_success, replace=False).tolist())
    if fail_idx:
        selected.extend(rng.choice(fail_idx, size=n_fail, replace=False).tolist())

    if not selected:
        print("No episodes to plot.")
        return

    # combined plot
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.add_patch(plt.Circle((0, 0), 2.0, color="gray", alpha=0.4))
    ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, edgecolor="dimgray", linewidth=1.5))

    colors_ok = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_success, 1)))
    colors_fail = plt.cm.Reds(np.linspace(0.4, 0.9, max(n_fail, 1)))
    ci_ok, ci_fail = 0, 0

    for idx in selected:
        traj = trajectories[idx]
        gx, gy = goals[idx]
        s = successes[idx]
        color = colors_ok[ci_ok % len(colors_ok)] if s else colors_fail[ci_fail % len(colors_fail)]
        if s:
            ci_ok += 1
        else:
            ci_fail += 1
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.8)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=6)
        ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=6)
        ax.plot(gx, gy, "*", color="green", markersize=12, zorder=5)

    ax.plot([], [], "o", color="steelblue", label="Start")
    ax.plot([], [], "s", color="steelblue", label="End")
    ax.plot([], [], "*", color="green", markersize=10, label="Goal")
    ax.plot([], [], "-", color="steelblue", label="Success")
    ax.plot([], [], "-", color="indianred", label="Fail")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{algo_label} — Agent Trajectories")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, f"trajectories_{algo_label.lower()}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # individual trajectory plots (3)
    for k, idx in enumerate(selected[:3]):
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.add_patch(plt.Circle((0, 0), 2.0, color="gray", alpha=0.4))
        ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, edgecolor="dimgray", linewidth=1.5))
        traj = trajectories[idx]
        gx, gy = goals[idx]
        s = successes[idx]
        color = "steelblue" if s else "indianred"
        label = "Success" if s else "Fail"
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, label=label)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=8, label="Start")
        ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=8, label="End")
        ax.plot(gx, gy, "*", color="green", markersize=14, label="Goal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-12, 12)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{algo_label} — Trajectory {k+1} ({label}, {len(traj)-1} steps)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_dir, f"trajectory_{algo_label.lower()}_{k+1}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 4. Comparison bar chart
# ──────────────────────────────────────────────────────────────────────

def plot_comparison_bar(results_dict: dict, save_dir: str):
    """Bar chart comparing success rate, collision rate, avg length."""
    os.makedirs(save_dir, exist_ok=True)
    algos = list(results_dict.keys())
    metrics = {
        "Success Rate (%)": [np.mean(results_dict[a]["successes"]) * 100 for a in algos],
        "Collision Rate (%)": [np.mean(results_dict[a]["collisions"]) * 100 for a in algos],
        "Avg Episode Length": [np.mean(results_dict[a]["lengths"]) for a in algos],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"PPO": "steelblue", "REINFORCE": "orangered"}
    bar_colors = [colors.get(a, "gray") for a in algos]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(algos, values, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved comparison bar chart to {path}")


# ──────────────────────────────────────────────────────────────────────
# 5. GIF recording
# ──────────────────────────────────────────────────────────────────────

def record_gif(agent, save_path: str, n_episodes: int = 3, fps: int = 20):
    try:
        import imageio
    except ImportError:
        print("imageio not installed, skipping GIF.")
        return

    env = DiffDriveNavEnv(render_mode="rgb_array")
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7)
        done = False
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    env.close()
    if frames:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        imageio.mimsave(save_path, frames, fps=fps, loop=0)
        print(f"Saved GIF ({len(frames)} frames) to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 6. Live pygame rendering
# ──────────────────────────────────────────────────────────────────────

def render_episodes(agent, n_episodes: int = 3):
    import time as _time
    env = DiffDriveNavEnv(render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7)
        done = False
        total_reward = 0.0
        env.render()
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
        status = "SUCCESS" if info.get("success") else ("COLLISION" if info.get("collision") else "TIMEOUT")
        print(f"  Episode {ep+1}: {status} | Return: {total_reward:.2f} | Steps: {info['step']}")
        _time.sleep(1.0)
    env.close()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare RL agents")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # individual mode
    parser.add_argument("--algo", type=str, default=None, choices=["ppo", "reinforce"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    args = parser.parse_args()

    env = DiffDriveNavEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    if args.algo and args.checkpoint:
        # single agent evaluation
        print(f"\nEvaluating {args.algo.upper()} from {args.checkpoint}...")
        agent = load_agent(args.algo, args.checkpoint, obs_dim, act_dim, args.hidden)
        results = evaluate_policy(agent, args.n_episodes, args.seed)
        label = args.algo.upper()
        plot_trajectories(results, args.output_dir, label)
        if args.metrics and os.path.exists(args.metrics):
            plot_comparison_curves({args.algo: args.metrics}, args.output_dir)
        if args.render:
            render_episodes(agent, 3)
        if args.gif:
            record_gif(agent, os.path.join(args.output_dir, f"demo_{args.algo}.gif"))
        return

    # comparison mode: evaluate both
    algos = {
        "ppo": ("checkpoints/ppo_final.pt", "logs/ppo_metrics.npz"),
        "reinforce": ("checkpoints/reinforce_final.pt", "logs/reinforce_metrics.npz"),
    }

    results_dict = {}
    agents_dict = {}
    metrics_paths = {}

    for algo, (ckpt, metrics) in algos.items():
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}, skipping {algo}")
            continue
        label = algo.upper()
        print(f"\n{'='*50}")
        print(f"Evaluating {label}")
        print(f"{'='*50}")
        agent = load_agent(algo, ckpt, obs_dim, act_dim, args.hidden)
        results = evaluate_policy(agent, args.n_episodes, args.seed)
        results_dict[label] = results
        agents_dict[label] = agent
        plot_trajectories(results, args.output_dir, label)
        if os.path.exists(metrics):
            metrics_paths[algo] = metrics

        if args.gif:
            record_gif(agent, os.path.join(args.output_dir, f"demo_{algo}.gif"))

    # comparison plots
    if len(results_dict) == 2:
        plot_comparison_bar(results_dict, args.output_dir)

    if metrics_paths:
        plot_comparison_curves(metrics_paths, args.output_dir)

    if args.render and agents_dict:
        for label, agent in agents_dict.items():
            print(f"\nRendering {label}...")
            render_episodes(agent, 3)


if __name__ == "__main__":
    main()
