"""Hand-written PPO and REINFORCE agents for continuous control."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Shared network
# ──────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Actor-critic network with separate policy and value heads.

    Policy outputs mean for each action dimension; log_std is a learnable
    parameter (state-independent).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_dist(self, obs: torch.Tensor):
        mean, std, value = self(obs)
        return Normal(mean, std), value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.get_dist(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


# ──────────────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores transitions for on-policy algorithms."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def to_tensors(self, device="cpu"):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.actions), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.values), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.dones), dtype=torch.float32, device=device),
        )


# ──────────────────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────────────────

class PPOAgent:
    """Proximal Policy Optimization with clipped surrogate objective.

    Key ideas:
    - Collects rollouts of fixed length, then updates for multiple epochs
    - Clips the policy ratio to prevent destructively large updates
    - Uses GAE (Generalized Advantage Estimation) for variance reduction
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 256,
        hidden: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.net = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value = self.net.act(obs_t, deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, buffer: RolloutBuffer, next_value: float):
        obs, actions, old_log_probs, rewards, values, dones = buffer.to_tensors(self.device)

        # compute GAE
        advantages_np, returns_np = self.compute_gae(
            rewards.cpu().numpy(), values.cpu().numpy(),
            dones.cpu().numpy(), next_value,
        )
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        stats = {"pg_loss": 0, "v_loss": 0, "entropy": 0, "clip_frac": 0, "updates": 0}

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                b_obs = obs[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                dist, values_pred = self.net.get_dist(b_obs)
                new_log_probs = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # policy loss (clipped surrogate)
                ratio = (new_log_probs - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # value loss
                v_loss = 0.5 * (values_pred - b_ret).pow(2).mean()

                loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.item()
                stats["clip_frac"] += clip_frac
                stats["updates"] += 1

        for k in ["pg_loss", "v_loss", "entropy", "clip_frac"]:
            stats[k] /= max(stats["updates"], 1)
        return stats

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))


# ──────────────────────────────────────────────────────────────────────
# REINFORCE Agent
# ──────────────────────────────────────────────────────────────────────

class ReinforceAgent:
    """REINFORCE (Monte-Carlo policy gradient) with learned baseline.

    Key ideas:
    - Collects full episodes, computes discounted returns
    - Uses the return as the gradient weight (no bootstrapping)
    - A learned value baseline reduces variance without introducing bias
    - Simpler than PPO but higher variance and less sample-efficient
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.net = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value = self.net.act(obs_t, deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def compute_returns(self, rewards, dones):
        """Compute discounted returns for each timestep."""
        n = len(rewards)
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            if dones[t]:
                running = 0.0
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def update(self, buffer: RolloutBuffer):
        obs, actions, old_log_probs, rewards, values, dones = buffer.to_tensors(self.device)

        # compute Monte-Carlo returns
        returns_np = self.compute_returns(
            rewards.cpu().numpy(), dones.cpu().numpy(),
        )
        returns = torch.tensor(returns_np, dtype=torch.float32, device=self.device)

        # forward pass
        dist, values_pred = self.net.get_dist(obs)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        # advantage = return - baseline
        advantages = returns - values_pred.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy gradient loss
        pg_loss = -(log_probs * advantages).mean()

        # value loss (train the baseline)
        v_loss = 0.5 * (values_pred - returns).pow(2).mean()

        loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy.item(),
        }

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))
