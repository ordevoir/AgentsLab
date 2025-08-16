# ===============================
# file: network.py
# ===============================
from typing import Sequence
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Простая MLP-сеть для Q(s, a).

    Разбейте ваш прежний Policy на чистую сеть (Network) и объект-агент.
    Network отвечает только за прямой проход.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===============================
# file: buffer.py
# ===============================
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Простой опыт-реплей на deque."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.memory, batch_size)
        states = np.stack([b.state for b in batch], axis=0)
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.stack([b.next_state for b in batch], axis=0)
        dones = np.array([b.done for b in batch], dtype=np.bool_)
        return Transition(states, actions, rewards, next_states, dones)


# ===============================
# file: utils.py
# ===============================
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class LinearEpsilonSchedule:
    """Линейное расписание eps-greedy: от eps_start -> eps_end за decay_steps."""

    def __init__(self, eps_start: float, eps_end: float, decay_steps: int):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = max(1, int(decay_steps))

    def __call__(self, t: int) -> float:
        frac = min(1.0, max(0.0, t / self.decay_steps))
        return self.eps_start + (self.eps_end - self.eps_start) * frac


def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    target.load_state_dict(source.state_dict())


@dataclass
class TrainConfig:
    env_id: str = "CartPole-v1"
    total_episodes: int = 500
    max_steps_per_episode: int = 500

    buffer_capacity: int = 50_000
    batch_size: int = 64

    gamma: float = 0.99
    lr: float = 1e-3
    grad_clip: float = 5.0

    start_learning_after: int = 1_000  # шагов взаимодействия до начала обучения
    train_every: int = 1               # обучаемся каждый шаг (можно увеличить для пропусков)
    target_update_freq: int = 1_000    # как часто копировать веса в target

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    double_dqn: bool = True
    seed: int = 42
    device: Optional[str] = None  # "cuda" / "cpu"; если None — авто


# ===============================
# file: agent.py
# ===============================
from typing import Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import hard_update


class DQNAgent:
    """Agent инкапсулирует сети, оптимизатор и выбор действия."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_fn: Callable[[], nn.Module],
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
        double_dqn: bool = True,
        grad_clip: float = 5.0,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip

        self.policy_net = network_fn().to(self.device)
        self.target_net = network_fn().to(self.device)
        hard_update(self.target_net, self.policy_net)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_dim = action_dim

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def train_step(self, batch) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones = batch

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(-1)

        # Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: a* = argmax_a Q_online(s', a)
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                # Q_target(s', a*)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                # DQN: max_a Q_target(s', a)
                next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values

            target_q = rewards_t + (~dones_t) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        mean_q = q_values.detach().mean().item()
        return float(loss.item()), float(mean_q)

    def update_target(self):
        hard_update(self.target_net, self.policy_net)


# ===============================
# file: trainer.py
# ===============================
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

# Поддержка как gymnasium, так и старого gym
try:
    import gymnasium as gym
    _NEW_API = True
except Exception:
    import gym
    _NEW_API = False

from buffer import ReplayBuffer
from agent import DQNAgent
from utils import LinearEpsilonSchedule, TrainConfig, set_seed


class Trainer:
    """Управляет циклом: взаимодействие с окружением, запись в буфер, обучение, таргет-апдейты."""

    def __init__(self, env, agent: DQNAgent, buffer: ReplayBuffer, cfg: TrainConfig):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.cfg = cfg
        self.eps_schedule = LinearEpsilonSchedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
        self.global_step = 0

    def run(self) -> Dict[str, List[float]]:
        rewards_history: List[float] = []
        losses: List[float] = []

        for episode in range(1, self.cfg.total_episodes + 1):
            if _NEW_API:
                state, _ = self.env.reset(seed=self.cfg.seed if episode == 1 else None)
            else:
                state = self.env.reset()
            ep_reward = 0.0

            for t in range(self.cfg.max_steps_per_episode):
                epsilon = self.eps_schedule(self.global_step)
                action = self.agent.act(state, epsilon)

                if _NEW_API:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = self.env.step(action)

                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                self.global_step += 1

                # Обучение
                if (
                    len(self.buffer) >= self.cfg.start_learning_after
                    and self.global_step % self.cfg.train_every == 0
                ):
                    batch = self.buffer.sample(self.cfg.batch_size)
                    loss, _ = self.agent.train_step(batch)
                    losses.append(loss)

                # Обновление target сети
                if self.global_step % self.cfg.target_update_freq == 0:
                    self.agent.update_target()

                if done:
                    break

            rewards_history.append(ep_reward)

            # Простой лог
            if episode % 10 == 0:
                avg100 = np.mean(rewards_history[-100:])
                last_loss = np.mean(losses[-50:]) if len(losses) else float("nan")
                print(
                    f"Episode {episode:4d} | step {self.global_step:6d} | "
                    f"return {ep_reward:7.2f} | avg100 {avg100:7.2f} | loss {last_loss:.4f} | eps {self.eps_schedule(self.global_step):.3f}"
                )

        return {"episode_returns": rewards_history, "losses": losses}


# ===============================
# file: train.py
# ===============================
import argparse
import torch

# gym/gymnasium импортирован в trainer.py
from network import QNetwork
from buffer import ReplayBuffer
from agent import DQNAgent
from trainer import Trainer
from utils import TrainConfig, set_seed


def make_env(env_id: str):
    # создаем среду в trainer.py (там уже есть поддержка обеих API)
    try:
        import gymnasium as gym
        env = gym.make(env_id)
    except Exception:
        import gym
        env = gym.make(env_id)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=TrainConfig.total_episodes)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(env_id=args.env_id, total_episodes=args.episodes, seed=args.seed, device=args.device)

    set_seed(cfg.seed)

    env = make_env(cfg.env_id)
    # Определяем размерности
    try:
        import gymnasium as gym
        obs_space = env.observation_space
        act_space = env.action_space
        state_dim = obs_space.shape[0]
        action_dim = act_space.n
    except Exception:
        # старый gym: аналогично
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Фабрика для сетей, чтобы у agent были одинаковые policy/target
    def network_factory():
        return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128))

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        network_fn=network_factory,
        lr=cfg.lr,
        gamma=cfg.gamma,
        device=device,
        double_dqn=cfg.double_dqn,
        grad_clip=cfg.grad_clip,
    )

    buffer = ReplayBuffer(cfg.buffer_capacity)
    trainer = Trainer(env, agent, buffer, cfg)

    history = trainer.run()

    # Пример: сохраним веса
    torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pt")
    print("Saved model to dqn_cartpole.pt")


if __name__ == "__main__":
    main()
