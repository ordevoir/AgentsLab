# =========================
# Imports
# =========================
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt


# =========================
# Policy Network
# =========================
class Network(nn.Module):
    """
    Небольшая двухслойная сеть: state -> logits(actions).
    Ничего не знает о лог-вероятностях, наградах и пр.
    """
    def __init__(self, in_dim: int, out_dim: int, n_units: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, n_units),
            nn.ReLU(),
            nn.Linear(n_units, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Agent (PG)
# =========================
@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr:    float = 5e-3
    grad_clip: float | None = None  # например, 1.0 или None


class Agent:
    """
    Хранит память эпизода, выбирает действия, считает возвраты и обновляет сеть.
    """
    def __init__(self, network: Network, config: AgentConfig, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = network.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.gamma = config.gamma
        self.grad_clip = config.grad_clip

        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float] = []

    # ---------- взаимодействие ----------
    def reset_episode(self) -> None:
        self.log_probs.clear()
        self.rewards.clear()

    def act(self, state: np.ndarray) -> int:
        """
        Выбирает действие стохастически из Categorical(logits).
        Сохраняет log_prob выбранного действия для градиентного шага.
        """
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.net(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def remember(self, reward: float) -> None:
        self.rewards.append(float(reward))

    # ---------- обучение ----------
    def _returns(self) -> torch.Tensor:
        """
        Ретроспективно считает G_t и нормализует (baseline = 0).
        """
        T = len(self.rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        future = 0.0
        for t in reversed(range(T)):
            future = self.rewards[t] + self.gamma * future
            returns[t] = future
        # Нормализация отдач — тот же прием, что и у вас
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self) -> float:
        """
        Один policy gradient шаг по накопленной траектории.
        Возвращает значение функции потерь (для логов).
        """
        if not self.log_probs:
            return 0.0

        R = self._returns()
        log_probs = torch.stack(self.log_probs)  # [T]
        loss = -(log_probs * R).mean()

        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

        # очистим память эпизода
        self.reset_episode()
        return float(loss.item())

    # ---------- удобный хелпер ----------
    def run_episode(self, env: gym.Env, max_steps: int = 500) -> Tuple[float, float]:
        """
        Полный проход одного эпизода: сбор траектории + апдейт.
        Возвращает (total_reward, loss).
        """
        self.reset_episode()
        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = self.act(obs)
            obs, rew, term, trunc, _ = env.step(action)
            self.remember(rew)
            total_reward += rew
            if term or trunc:
                break

        loss = self.update()
        return total_reward, loss


# =========================
# Training
# =========================
def moving_mean(array: np.ndarray, width: int = 30) -> np.ndarray:
    means = np.zeros_like(array, dtype=float)
    for i in range(len(means)):
        left = max(0, i + 1 - width)
        means[i] = array[left:i + 1].mean()
    return means


def main():
    # --- Env ---
    env = gym.make("CartPole-v1")
    m = env.observation_space.shape[0]
    n = env.action_space.n

    # --- Network + Agent ---
    net = Network(m, n, n_units=16)
    agent = Agent(net, AgentConfig(gamma=0.99, lr=0.005, grad_clip=None), device="cpu")

    # --- Hyperparams ---
    episodes = 900
    max_steps = 500

    # --- Train loop ---
    total_rewards: List[float] = []
    losses: List[float] = []
    for ep in range(episodes):
        ep_reward, loss = agent.run_episode(env, max_steps)
        total_rewards.append(ep_reward)
        losses.append(loss)
        print(f"\repisode={ep:4d} | loss={loss:8.4f} | reward={ep_reward:6.1f}", end="")

    # --- Plot ---
    xs = np.arange(len(total_rewards))
    rewards = np.array(total_rewards, dtype=float)
    means = moving_mean(rewards, 30)

    plt.figure()
    plt.plot(xs, rewards, label="Rewards")
    plt.plot(xs, means,   label="Moving mean (30)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Vanilla Policy Gradient on CartPole-v1")
    plt.show()


if __name__ == "__main__":
    main()

