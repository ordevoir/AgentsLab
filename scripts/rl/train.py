from __future__ import annotations

from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from agentslab.rl.agents.pg import PGAgent, PGConfig, PGPolicy
from agentslab.rl.agents.dqn import DQNAgent, DQNConfig
from agentslab.utils.io import CheckpointPaths, save_checkpoint
from agentslab.utils.seed import select_device, set_seed


@main(version_base=None, config_path="../../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    # --- Setup ---
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    env = gym.make(cfg.env.id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    run_dir = Path.cwd()  # Hydra run dir
    paths = CheckpointPaths(run_dir=run_dir, algo=cfg.rl.algo.name, env_id=cfg.env.id)

    writer: SummaryWriter | None = None
    if cfg.logging.tensorboard:
        tb_dir = run_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir, flush_secs=cfg.logging.tb_flush_secs)

    # --- Build agent ---
    if cfg.rl.algo.name == "pg":
        pg_cfg = PGConfig(
            gamma=cfg.rl.algo.agent.gamma,
            lr=cfg.rl.algo.optimizer.lr,
            grad_clip=cfg.rl.algo.agent.grad_clip,
            hidden_units=cfg.rl.algo.network.hidden_units,
        )
        policy = PGPolicy(obs_dim, act_dim, hidden_units=pg_cfg.hidden_units)
        agent = PGAgent(policy, pg_cfg, device=device)

        def train_one_episode(ep_idx: int) -> Dict[str, float]:
            reward, loss = agent.run_episode(env, cfg.train.max_steps)
            return {"reward": reward, "loss": loss}

        def snapshot_state(best_metric: float, last_metrics: Dict[str, float]):
            return {
                "algo": "pg",
                "env_id": cfg.env.id,
                "model_state": agent.net.state_dict(),
                "optimizer_state": agent.opt.state_dict(),
                "metrics": {"best_avg_reward": best_metric, **last_metrics},
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }

    elif cfg.rl.algo.name == "dqn":
        dqn_cfg = DQNConfig(
            gamma=cfg.rl.algo.agent.gamma,
            tau=cfg.rl.algo.agent.tau,
            batch_size=cfg.rl.algo.agent.batch_size,
            buffer_capacity=cfg.rl.algo.agent.buffer_capacity,
            clip_grad_value=cfg.rl.algo.agent.clip_grad_value,
            lr=cfg.rl.algo.optimizer.lr,
            weight_decay=cfg.rl.algo.optimizer.weight_decay,
            eps_start=cfg.rl.algo.agent.epsilon.start,
            eps_end=cfg.rl.algo.agent.epsilon.end,
            eps_decay_steps=cfg.rl.algo.agent.epsilon.decay_steps,
            hidden1=cfg.rl.algo.network.hidden1,
            hidden2=cfg.rl.algo.network.hidden2,
        )
        agent = DQNAgent(obs_dim, act_dim, dqn_cfg, device=device)

        def train_one_episode(ep_idx: int) -> Dict[str, float]:
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward = 0.0
            losses = []
            for t in range(cfg.train.max_steps):
                action = agent.select_action(state, env)
                obs, rew, term, trunc, _ = env.step(action.item())
                reward_t = torch.tensor([rew], device=device)
                done = term or trunc
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                agent.memory.push(state, action, next_state, reward_t)
                state = next_state if next_state is not None else torch.zeros_like(state)
                loss = agent.optimize()
                if loss is not None:
                    losses.append(loss)
                total_reward += rew
                if done:
                    break
            loss_mean = float(np.mean(losses)) if losses else 0.0
            return {"reward": total_reward, "loss": loss_mean}

        def snapshot_state(best_metric: float, last_metrics: Dict[str, float]):
            return {
                "algo": "dqn",
                "env_id": cfg.env.id,
                "online_state": agent.online.state_dict(),
                "target_state": agent.target.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "steps": agent.steps,
                "metrics": {"best_avg_reward": best_metric, **last_metrics},
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
    else:
        raise ValueError(f"Unknown algo: {cfg.rl.algo.name}")

    # --- Train loop ---
    best_avg = float("-inf")
    window = cfg.checkpointing.avg_window
    recent_rewards: list[float] = []

    for ep in range(cfg.train.episodes):
        metrics = train_one_episode(ep)
        recent_rewards.append(metrics["reward"])
        if len(recent_rewards) > window:
            recent_rewards.pop(0)
        avg_reward = float(np.mean(recent_rewards))

        if writer is not None:
            writer.add_scalar("train/reward", metrics["reward"], ep)
            writer.add_scalar("train/avg_reward", avg_reward, ep)
            writer.add_scalar("train/loss", metrics["loss"], ep)

        if cfg.logging.stdout_interval and (ep % cfg.logging.stdout_interval == 0):
            print(f"[ep={ep:04d}] reward={metrics['reward']:.1f} avg@{window}={avg_reward:.1f} loss={metrics['loss']:.4f}")

        # Checkpointing
        if cfg.checkpointing.save_best and avg_reward > best_avg:
            best_avg = avg_reward
            state = snapshot_state(best_avg, metrics)
            save_checkpoint(paths, state, tag="best")
        if cfg.checkpointing.save_last:
            state = snapshot_state(best_avg, metrics)
            save_checkpoint(paths, state, tag="last")
        if cfg.checkpointing.keep_every and (ep + 1) % int(cfg.checkpointing.keep_every) == 0:
            tag = f"ep{ep+1:05d}"
            state = snapshot_state(best_avg, metrics)
            save_checkpoint(paths, state, tag=tag)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    run()
