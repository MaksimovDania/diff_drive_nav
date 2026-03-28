"""Live demo: run the trained agent with pygame visualization."""

from stable_baselines3 import PPO
from env import DiffDriveNavEnv
import time


def main():
    model = PPO.load("checkpoints/ppo_final.zip")
    env = DiffDriveNavEnv(render_mode="human")

    for ep in range(5):
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

        status = "SUCCESS" if info["success"] else ("COLLISION" if info["collision"] else "TIMEOUT")
        print(f"Episode {ep+1}: {status} | Return: {total_reward:.1f} | Steps: {info['step']}")
        time.sleep(1.5)

    env.close()


if __name__ == "__main__":
    main()
