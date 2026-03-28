# Differential Drive Navigation with Reinforcement Learning

## 1. Environment Description and Reward Function Design

### Environment Overview

The environment (`env.py`) simulates a differential-drive robot navigating a 2D plane. A circular obstacle of radius 2 is centered at the origin. The robot must travel from a random start position to a random goal while avoiding collision.

| Parameter | Value |
|---|---|
| Agent spawn region | x in [-2, 0], y in [-10, 0] |
| Goal spawn region | x in [0, 2], y in [0, 10] |
| Obstacle | Circle at (0, 0), radius = 2 |
| Agent radius | 0.2 |
| Collision boundary | distance to origin < 2.2 (obstacle + agent radii) |
| Goal threshold | distance to goal < 0.3 |
| Simulation timestep (dt) | 0.1 s |
| Wheelbase (L) | 0.5 |
| Max episode steps | 800 |

Both spawn regions are filtered via rejection sampling to guarantee a minimum clearance of 0.5 from the obstacle surface, preventing impossible initial conditions.

### Differential Drive Kinematics

The robot state is (x, y, theta). Actions are left/right wheel velocities (v_left, v_right) in [-1, 1]. At each step:

```
v     = (v_right + v_left) / 2
omega = (v_right - v_left) / L

x     += v * cos(theta) * dt
y     += v * sin(theta) * dt
theta += omega * dt
```

### Observation Space (12-dimensional, continuous)

| Index | Feature | Normalization | Rationale |
|---|---|---|---|
| 0 | dx to goal | / 12 | Relative goal direction |
| 1 | dy to goal | / 20 | Relative goal direction |
| 2 | Distance to goal | / 25 | Scalar proximity to objective |
| 3 | cos(theta) | native [-1,1] | Heading without angle wrapping |
| 4 | sin(theta) | native [-1,1] | Heading without angle wrapping |
| 5 | cos(angle to goal) | native [-1,1] | Goal bearing |
| 6 | sin(angle to goal) | native [-1,1] | Goal bearing |
| 7 | Distance to obstacle surface | / 5 | Proximity warning signal |
| 8 | cos(relative angle to obstacle) | native [-1,1] | Obstacle direction in agent frame |
| 9 | sin(relative angle to obstacle) | native [-1,1] | Obstacle direction in agent frame |
| 10 | Agent x position | / 5 | Spatial context for navigation |
| 11 | Agent y position | / 12 | Spatial context for navigation |

Angles are represented as (cos, sin) pairs to avoid discontinuities at +/-pi. All features are pre-normalized with fixed constants derived from the world geometry, avoiding instability from running statistics.

### Action Space (2-dimensional, continuous)

| Index | Feature | Range |
|---|---|---|
| 0 | v_left (left wheel velocity) | [-1, 1] |
| 1 | v_right (right wheel velocity) | [-1, 1] |

### Reward Function Design

The reward function combines potential-based shaping with sparse terminal bonuses:

| Component | Formula | Scale | Justification |
|---|---|---|---|
| **Progress shaping** | (prev_dist - curr_dist) | x 5.0 | Primary learning signal. Potential-based, so it does not alter the optimal policy but provides dense gradient toward the goal. |
| **Heading alignment** | (1 - \|heading_error\| / pi) | x 0.1 | Small bonus for facing the goal. Encourages the robot to orient before moving, which is critical for differential drive (cannot strafe). |
| **Step penalty** | constant | -0.01 | Encourages efficiency. Small enough not to dominate but discourages loitering. |
| **Proximity penalty** | (1 - dist_obs / threshold) when dist_obs < 1.0 | x -1.0 | Smooth gradient pushing the agent away from the obstacle. Activates within 1.0 unit of the surface. |
| **Goal reached** | one-time bonus | +20.0 | Terminal reward for success. |
| **Collision** | one-time penalty | -10.0 | Terminal penalty for hitting the obstacle. |

The progress shaping term is the dominant signal at approximately 0.5 per step when moving toward the goal at full speed, which is an order of magnitude larger than the step penalty and heading bonus. This hierarchy ensures the agent prioritizes reaching the goal while receiving gentle steering guidance.

### Termination Conditions

- **Success**: distance to goal < 0.3 (terminated = True)
- **Collision**: distance to obstacle center < 2.2 (terminated = True)
- **Timeout**: step count >= 800 (truncated = True)

### Rendering

The environment supports Pygame rendering in both `"human"` (live window) and `"rgb_array"` (frame capture for GIF) modes. The visualization shows a 600x600 window mapping world coordinates [-12, 12] x [-12, 12], with grid lines, the obstacle circle, a green goal marker, an oriented blue triangle for the agent, and a fading trajectory trail.

---

## 2. Choice of RL Algorithm and Rationale

### Algorithm: PPO (Proximal Policy Optimization)

**Implementation**: Stable-Baselines3 `PPO` with an MLP policy.

### Why PPO

1. **Continuous action space**: The differential drive requires continuous wheel velocity control. PPO handles this natively with Gaussian policies (learned mean + standard deviation for each action dimension), unlike DQN which requires discretization.

2. **Stability**: PPO's clipped surrogate objective constrains policy updates to a trust region, preventing the large destructive updates that plague vanilla policy gradient methods like REINFORCE. This is especially important here because the reward landscape has sharp transitions (collision penalty, goal bonus) that can destabilize training.

3. **Sample efficiency**: Compared to REINFORCE (which discards data after each update), PPO reuses rollout data across multiple epochs (10 in our configuration), extracting more learning per environment interaction.

4. **Proven track record**: PPO is the standard algorithm for continuous-control robotics tasks in both simulation and sim-to-real transfer. It consistently outperforms TRPO (which it approximates more cheaply) and A2C on locomotion and navigation benchmarks.

### Why not other algorithms

- **DQN**: Requires discretizing the action space. With two continuous wheel velocities, even a coarse 5-level discretization yields 25 action combinations, losing fine control precision.
- **REINFORCE**: High variance, no value baseline in the basic form, single-use data. Would require significantly more environment interactions.
- **TRPO**: Theoretically sound but computationally expensive (requires conjugate gradient + line search). PPO achieves comparable performance with simpler implementation.
- **SAC**: A viable alternative for continuous control, but PPO's on-policy nature makes it more stable with shaped rewards that may change the effective MDP during training.

### Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Learning rate | 3e-4 | Standard for PPO, Adam optimizer |
| Rollout steps | 2048 | Sufficient to capture multiple complete episodes per update |
| Mini-batch size | 256 | Balances gradient noise and compute efficiency |
| Epochs per update | 10 | Reuses data without overfitting (clipping prevents excessive change) |
| Gamma | 0.99 | Long horizon: episodes can last up to 800 steps |
| GAE lambda | 0.95 | Balances bias-variance in advantage estimation |
| Clip range | 0.2 | Standard PPO clipping |
| Entropy coefficient | 0.01 | Mild exploration bonus to prevent premature convergence |
| Network architecture | [64, 64] | Two hidden layers of 64 units. Sufficient capacity for the 12-dim observation space; larger networks showed no improvement. |
| Total timesteps | 2,000,000 | Empirically determined: success rate plateaus around 1.5M steps |

---

## 3. Training Curves

Training was conducted for 2,000,000 timesteps (~7.5 minutes on CPU). Metrics were logged per episode via a custom callback and evaluated every 10,000 steps on 20 held-out episodes.

### Reward per Episode

![Training Curves](results/training_curves.png)

**Left plot (Training Reward)**: Raw episode returns (light) and smoothed with a 50-episode moving average (solid). The reward climbs steeply during the first 500K steps as the agent learns to move toward the goal, then continues to improve gradually as it refines obstacle avoidance. The final smoothed return stabilizes around 90.

**Right plot (Training Success Rate)**: The rolling success rate (fraction of episodes reaching the goal) rises from near 0% to above 90% by 1M steps and approaches 100% by 2M steps. The curve shows some variance in the early phase (200K-600K) as the agent oscillates between reaching the goal and colliding.

---

## 4. Evaluation Results and Trajectory Visualizations

### Quantitative Results

Evaluated over **100 episodes** with random spawn/goal positions using deterministic policy inference:

| Metric | Value |
|---|---|
| **Success rate** | **100.0%** |
| **Collision rate** | **0.0%** |
| **Timeout rate** | **0.0%** |
| **Average return** | 91.98 |
| **Average episode length** | 153.4 steps |

The agent reliably reaches every goal without a single collision or timeout across all 100 test episodes.

### Trajectory Visualizations

#### Combined Trajectory Plot

![Combined Trajectories](results/trajectories.png)

The plot shows five representative trajectories overlaid on the environment. The gray circle is the obstacle (radius 2) at the origin. Green stars mark goal positions. Circles mark start positions and squares mark endpoints. All trajectories (blue) successfully navigate around the obstacle to reach their respective goals.

#### Individual Trajectories

| Trajectory 1 | Trajectory 2 | Trajectory 3 |
|---|---|---|
| ![T1](results/trajectory_1.png) | ![T2](results/trajectory_2.png) | ![T3](results/trajectory_3.png) |

The individual plots show the agent taking smooth, curved paths around the obstacle. The agent has learned to:
- Orient toward a tangent direction when the direct path is blocked
- Maintain safe clearance from the obstacle surface
- Execute efficient arcing trajectories rather than sharp turns

### Animated Demo

An animated GIF of three episodes is available at `results/demo.gif`.

---

## 5. Discussion

### What Worked

1. **Potential-based reward shaping** was the single most impactful design choice. The `(prev_dist - curr_dist) * 5.0` term provides a dense, informative gradient from the first step of training. Without it, the agent would need to randomly stumble into the goal to receive any positive signal, which almost never happens in a continuous 2D space.

2. **Cos/sin angle representation** avoided the discontinuity problem at +/-pi. Early experiments with raw theta values showed the agent struggling when its heading crossed the -pi/+pi boundary, leading to sudden jumps in the observation space.

3. **Obstacle direction in the agent's frame** (observation indices 8-9) gave the agent awareness of where the obstacle is relative to its heading. This is critical for differential drive because the robot can only move forward/backward along its heading, so it needs to know whether the obstacle is ahead-left or ahead-right to choose the correct turning direction.

4. **Agent position in the observation** (indices 10-11) allowed the agent to learn a spatial policy that varies based on location. Near the obstacle, the agent takes wider arcs; far from it, it moves more directly.

5. **Stable-Baselines3** eliminated boilerplate for rollout buffers, GAE computation, logging, and evaluation callbacks, letting us focus entirely on environment and reward design.

### What Didn't Work

1. **Spawn region overlapping the obstacle** was a critical bug that took several training iterations to identify. The assignment specifies x in [-2, 0], y in [-10, 0], but points like (0, -2) or (-1, -1) are inside the obstacle (distance to origin < 2.2). This caused the agent to experience unavoidable collisions on spawn, injecting noise into the reward signal and capping the success rate at ~62-68% regardless of training duration or reward tuning. The fix was rejection sampling to enforce a minimum safe distance of 2.7 from the origin.

2. **Overly aggressive obstacle penalties** backfired. Increasing `COLLISION_PENALTY` from -10 to -20 and `PROXIMITY_SCALE` from -1.0 to -2.0 caused the agent to learn an excessively cautious policy that circled far from the obstacle and timed out instead of reaching the goal (0% success with high reward from shaping alone).

3. **Vectorized training with SubprocVecEnv** degraded performance when using 8 parallel environments with the default `n_steps=2048`. Each environment only collected 256 steps per rollout, resulting in incomplete episodes and poor advantage estimates. The entropy collapsed (0.237 vs ~2.8 at init) and the policy became deterministic but suboptimal. Single-environment training with full 2048-step rollouts performed much better.

4. **Larger networks (128x128)** showed no improvement over 64x64 for this task. The 12-dimensional observation and 2-dimensional action spaces are simple enough that additional capacity provided no benefit and slightly increased training time.

### Bugs Encountered

- **Spawn-obstacle overlap**: The most impactful bug. Points in the spawn region with x^2 + y^2 < (2.0 + 0.2)^2 = 4.84 are inside the collision zone. For example, (0, -2) has distance 2.0 from origin, well inside the 2.2 collision boundary. Fixed with rejection sampling.
- **Goal-obstacle overlap**: Similarly, goal positions like (1, 1) (distance 1.41 from origin) are inside the obstacle. Also fixed with rejection sampling.
- **Proximity penalty scaling**: The original formula `PROXIMITY_SCALE * (1.0 - dist_to_obs)` was not normalized by the threshold, meaning the penalty could exceed the intended scale when the threshold was changed. Fixed to `PROXIMITY_SCALE * (1.0 - dist_to_obs / PROXIMITY_THRESHOLD)`.

### Possible Improvements

1. **Curriculum learning**: Start with nearby goals (short distances) and gradually increase difficulty. This could speed up early training by giving the agent easier problems first.

2. **Multiple obstacles**: The current environment has a single fixed obstacle. Generalizing to randomly placed obstacles would test the agent's ability to plan more complex paths.

3. **Domain randomization on dynamics**: Varying the wheelbase, dt, or max velocity during training could produce a more robust policy that transfers better to real robots.

4. **Recurrent policy (LSTM)**: A recurrent network could maintain memory of recent trajectory, potentially improving path planning around the obstacle without needing explicit position in the observation.

5. **Higher-fidelity physics**: Adding wheel slip, motor dynamics, or sensor noise would make the simulation more realistic for sim-to-real transfer.

---

## Usage

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install stable-baselines3 gymnasium pygame-ce matplotlib imageio tensorboard

# Train (takes ~7 minutes on CPU)
python train.py --total-timesteps 2000000

# Evaluate and generate all plots
python evaluate.py --checkpoint checkpoints/ppo_final.zip --gif results/demo.gif

# Live pygame rendering
python evaluate.py --checkpoint checkpoints/ppo_final.zip --render
```

## Project Structure

```
diff_drive_nav/
  env.py           -- Gymnasium environment (DiffDriveNavEnv)
  train.py         -- PPO training script (Stable-Baselines3)
  evaluate.py      -- Evaluation metrics, plots, and GIF generation
  checkpoints/     -- Saved model weights (ppo_final.zip, best_model.zip)
  logs/            -- Training metrics (training_metrics.npz) and TensorBoard logs
  results/         -- Output plots and demo GIF
    training_curves.png
    trajectories.png
    trajectory_1.png, trajectory_2.png, trajectory_3.png
    demo.gif
```
