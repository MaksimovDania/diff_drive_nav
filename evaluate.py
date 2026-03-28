"""Evaluate a trained PPO agent and generate plots / visualizations."""

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from stable_baselines3 import PPO

from env import DiffDriveNavEnv


# ──────────────────────────────────────────────────────────────────────
# 1. Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_policy(model, n_episodes: int = 100, seed: int = 0):
    """Run episodes and collect metrics."""
    env = DiffDriveNavEnv()
    results = {"returns": [], "lengths": [], "successes": [], "collisions": [],
               "trajectories": [], "goals": []}

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        trajectory = [(env.x, env.y)]
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
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

    env.close()

    success_rate = np.mean(results["successes"])
    collision_rate = np.mean(results["collisions"])
    avg_return = np.mean(results["returns"])
    avg_length = np.mean(results["lengths"])

    print(f"\n{'='*50}")
    print(f"Evaluation over {n_episodes} episodes:")
    print(f"  Success rate:   {success_rate*100:.1f}%")
    print(f"  Collision rate: {collision_rate*100:.1f}%")
    print(f"  Avg return:     {avg_return:.2f}")
    print(f"  Avg length:     {avg_length:.1f} steps")
    print(f"{'='*50}\n")

    return results


# ──────────────────────────────────────────────────────────────────────
# 2. Training curves
# ──────────────────────────────────────────────────────────────────────

def smooth(values, window=50):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(metrics_path: str, save_dir: str):
    """Plot reward and success rate over training."""
    data = np.load(metrics_path)
    returns = data["returns"]
    successes = data["successes"].astype(float)
    timesteps = data["timesteps"]

    os.makedirs(save_dir, exist_ok=True)
    window = min(50, len(returns) // 5) if len(returns) > 10 else 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # reward curve
    ax = axes[0]
    ax.plot(timesteps, returns, alpha=0.15, color="steelblue", linewidth=0.5)
    if len(returns) > window:
        smoothed = smooth(returns, window)
        t_smooth = timesteps[window - 1:]
        ax.plot(t_smooth, smoothed, color="steelblue", linewidth=2, label=f"Smoothed (w={window})")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.set_title("Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # success rate curve
    ax = axes[1]
    if len(successes) > window:
        sr_smooth = smooth(successes, window) * 100
        t_smooth = timesteps[window - 1:]
        ax.plot(t_smooth, sr_smooth, color="green", linewidth=2)
    else:
        ax.plot(timesteps, successes * 100, color="green", linewidth=2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Training Success Rate")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves to {path}")


# ──────────────────────────────────────────────────────────────────────
# 3. Trajectory plots
# ──────────────────────────────────────────────────────────────────────

def plot_trajectories(results: dict, save_dir: str, n_plots: int = 5):
    """Plot agent trajectories on a 2D plane with obstacle and goal."""
    os.makedirs(save_dir, exist_ok=True)

    trajectories = results["trajectories"]
    goals = results["goals"]
    successes = results["successes"]

    # select a mix of successful and failed trajectories
    success_idx = [i for i, s in enumerate(successes) if s]
    fail_idx = [i for i, s in enumerate(successes) if not s]

    selected = []
    # prioritize showing some successes and some failures
    n_success = min(max(n_plots - 1, n_plots // 2), len(success_idx))
    n_fail = min(n_plots - n_success, len(fail_idx))
    n_success = min(n_plots - n_fail, len(success_idx))

    rng = np.random.default_rng(42)
    if success_idx:
        selected.extend(rng.choice(success_idx, size=n_success, replace=False).tolist())
    if fail_idx:
        selected.extend(rng.choice(fail_idx, size=n_fail, replace=False).tolist())

    if not selected:
        print("No episodes to plot.")
        return

    # ── combined plot ──
    fig, ax = plt.subplots(figsize=(8, 10))
    obstacle = plt.Circle((0, 0), 2.0, color="gray", alpha=0.4, label="Obstacle")
    ax.add_patch(obstacle)
    ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, edgecolor="dimgray", linewidth=1.5))

    colors_ok = plt.cm.Blues(np.linspace(0.4, 0.9, n_success))
    colors_fail = plt.cm.Reds(np.linspace(0.4, 0.9, max(n_fail, 1)))
    ci_ok, ci_fail = 0, 0

    for idx in selected:
        traj = trajectories[idx]
        gx, gy = goals[idx]
        s = successes[idx]
        if s:
            color = colors_ok[ci_ok % len(colors_ok)]
            ci_ok += 1
        else:
            color = colors_fail[ci_fail % len(colors_fail)]
            ci_fail += 1

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.8)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=6)
        ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=6)
        ax.plot(gx, gy, "*", color="green", markersize=12, zorder=5)

    # legend
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
    ax.set_title("Agent Trajectories")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "trajectories.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to {path}")

    # ── individual plots ──
    for k, idx in enumerate(selected[:3]):
        fig, ax = plt.subplots(figsize=(6, 8))
        obstacle = plt.Circle((0, 0), 2.0, color="gray", alpha=0.4)
        ax.add_patch(obstacle)
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
        ax.set_title(f"Trajectory {k+1} ({label}, {len(traj)-1} steps)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        path = os.path.join(save_dir, f"trajectory_{k+1}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 4. Pygame live visualization
# ──────────────────────────────────────────────────────────────────────

def render_episodes(model, n_episodes: int = 3):
    """Render episodes in a pygame window."""
    env = DiffDriveNavEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 7)
        done = False
        total_reward = 0.0
        env.render()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()

        status = "SUCCESS" if info.get("success") else ("COLLISION" if info.get("collision") else "TIMEOUT")
        print(f"Episode {ep+1}: {status} | Return: {total_reward:.2f} | Steps: {info['step']}")

        # pause briefly between episodes
        import time
        time.sleep(1.0)

    env.close()


# ──────────────────────────────────────────────────────────────────────
# 5. GIF recording
# ──────────────────────────────────────────────────────────────────────

def record_gif(model, save_path: str, n_episodes: int = 3, fps: int = 20):
    """Record episodes as an animated GIF."""
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Run: pip install imageio")
        return

    env = DiffDriveNavEnv(render_mode="rgb_array")
    frames = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 7)
        done = False
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
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
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DiffDriveNav agent")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_final.zip",
                        help="Path to SB3 model checkpoint")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--metrics", type=str, default="logs/training_metrics.npz",
                        help="Path to training metrics file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output plots")
    parser.add_argument("--render", action="store_true",
                        help="Show live pygame rendering")
    parser.add_argument("--gif", type=str, default=None,
                        help="Path to save GIF (e.g. results/demo.gif)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = PPO.load(args.checkpoint)

    # training curves
    if os.path.exists(args.metrics):
        plot_training_curves(args.metrics, args.output_dir)
    else:
        print(f"Training metrics not found at {args.metrics}, skipping training curves.")

    # evaluation
    results = evaluate_policy(model, n_episodes=args.n_episodes, seed=args.seed)

    # trajectory plots
    plot_trajectories(results, args.output_dir)

    # live rendering
    if args.render:
        render_episodes(model, n_episodes=3)

    # GIF
    if args.gif:
        record_gif(model, args.gif)


if __name__ == "__main__":
    main()
