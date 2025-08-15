from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from agentslab.rl.agents.reinforce import REINFORCEAgent, Trajectory
from agentslab.networks.policy_mlp import PolicyMLP
from agentslab.rl.environments.factory import make_env
from agentslab.utils.seed import set_seed

@dataclass
class TrainStats:
    episode: int
    ep_return: float
    loss: float

def train(
    cfg,
) -> None:
    set_seed(cfg.seed)

    # --- Env (no wrappers) ---
    env = make_env(cfg.rl.env.id, render_mode=cfg.rl.env.render_mode)
    obs, info = env.reset(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- Model & Optimizer ---
    hidden_dim = cfg.rl.model.hidden_dim
    activation = cfg.rl.model.activation
    policy = PolicyMLP(obs_dim, action_dim, hidden_dim=hidden_dim, activation=activation)
    if cfg.rl.optimizer.name.lower() == "adam":
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.rl.optimizer.lr, weight_decay=cfg.rl.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.rl.optimizer.name}")

    # --- Agent ---
    agent = REINFORCEAgent(
        policy=policy,
        optimizer=optimizer,
        gamma=cfg.rl.training.gamma,
        normalize_returns=cfg.rl.training.normalize_returns,
        device=cfg.device,
    )

    # --- Logging ---
    writer = None
    if cfg.common.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "tb"))

    max_episodes = cfg.rl.training.max_episodes
    log_every = cfg.rl.training.log_every
    eval_every = cfg.rl.training.eval_every
    save_every = cfg.rl.training.save_every

    for ep in range(1, max_episodes + 1):
        obs, info = env.reset()
        done, truncated = False, False
        ep_return = 0.0
        traj_log_probs = []
        rewards = []

        while not (done or truncated):
            action, log_prob = agent.select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            traj_log_probs.append(log_prob)
            rewards.append(float(reward))
            ep_return += float(reward)

        loss = agent.update(Trajectory(log_probs=traj_log_probs, rewards=rewards))

        if writer is not None:
            writer.add_scalar("charts/ep_return", ep_return, ep)
            writer.add_scalar("loss/policy_loss", loss, ep)

        if ep % log_every == 0:
            print(f"[ep {ep:04d}] return={ep_return:.2f} loss={loss:.3f}")

        if save_every and (ep % save_every == 0 or ep == max_episodes):
            ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "rl")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(agent.policy.state_dict(), os.path.join(ckpt_dir, f"reinforce_ep{ep}.pt"))

    env.close()
    if writer is not None:
        writer.close()
