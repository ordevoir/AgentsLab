
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from collections import deque
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from agentslab.core.logger import TBLogger
from agentslab.rl.environments.factory import make_env
from agentslab.rl.agents.reinforce import ReinforceAgent, ReinforceConfig
from agentslab.rl.agents.dqn import DQNAgent, DQNConfig
from agentslab.rl.agents.ppo import PPOAgent, PPOConfig
from agentslab.utils.checkpointing import save_checkpoint, copy_as_last

@dataclass
class TrainCommonConfig:
    env_id: str = "CartPole-v1"
    seed: int = 42
    total_timesteps: int = 200_000
    eval_interval: int = 10_000
    log_interval: int = 1_000
    checkpoint_path: str = "checkpoints/rl/last.pt"  # legacy
    log_dir: str = "logs/tb"
    # new fields for structured checkpoints
    algo: str = "reinforce"
    ckpt_root: str = "checkpoints/rl"
    run_name: Optional[str] = None  # if None -> auto

def _make_ckpt_dir(common: TrainCommonConfig) -> str:
    run = common.run_name or time.strftime(f"%Y%m%d_%H%M%S_seed{common.seed}")
    ckpt_dir = os.path.join(common.ckpt_root, common.algo, common.env_id, run)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def _episodic_returns_from_rollout(traj: dict) -> list[float]:
    rets: list[float] = []
    acc = 0.0
    for r, d in zip(traj["rewards"], traj["dones"]):
        acc += float(r)
        if d >= 0.5:
            rets.append(acc)
            acc = 0.0
    return rets

class ReinforceTrainer:
    def __init__(self, cfg: TrainCommonConfig, agent_cfg: ReinforceConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.env = make_env(cfg.env_id, seed=cfg.seed)
        obs_shape = self.env.observation_space.shape
        assert obs_shape is not None
        obs_dim = int(np.prod(obs_shape))
        act_dim = self.env.action_space.n
        self.agent = ReinforceAgent(obs_dim, act_dim, agent_cfg)
        self.logger = TBLogger(cfg.log_dir)
        self.ckpt_dir = _make_ckpt_dir(cfg)
        self.agent_cfg = agent_cfg

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        ep_return = 0.0
        ep_len = 0
        logps = []
        rewards = []
        returns_window = deque(maxlen=20)
        pbar = tqdm(total=self.cfg.total_timesteps, desc=f"REINFORCE {self.cfg.env_id}", dynamic_ncols=True)
        last_loss = None
        for step in range(1, self.cfg.total_timesteps + 1):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, logp = self.agent.select_action(obs_t)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            logps.append(logp)
            rewards.append(reward)
            ep_return += reward
            ep_len += 1
            obs = next_obs

            if done:
                loss = self.agent.update(logps, rewards)
                last_loss = loss
                self.logger.log({"train/return": ep_return, "train/length": ep_len, "loss/policy": loss}, step)
                returns_window.append(ep_return)
                logps.clear()
                rewards.clear()
                ep_return = 0.0
                ep_len = 0
                obs, _ = self.env.reset()

            pbar.update(1)
            if step % self.cfg.log_interval == 0:
                mean_ret = float(np.mean(returns_window)) if len(returns_window) else 0.0
                pbar.set_postfix({
                    "R_mean": f"{mean_ret:6.1f}",
                    "Lpi": f"{(last_loss if last_loss is not None else 0.0):.3f}",
                })

            if step % self.cfg.eval_interval == 0:
                meta = {
                    "algorithm": self.cfg.algo,
                    "env_id": self.cfg.env_id,
                    "model": "ReinforceAgent",
                    "seed": self.cfg.seed,
                    "step": step,
                    "agent_cfg": asdict(self.agent_cfg),
                }
                ckpt_file = os.path.join(self.ckpt_dir, f"step_{step}.pt")
                save_checkpoint(ckpt_file, self.agent, self.agent.optimizer, step, meta=meta)
                copy_as_last(ckpt_file)

        pbar.close()
        self.logger.close()

class DQNTrainer:
    def __init__(self, cfg: TrainCommonConfig, agent_cfg: DQNConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.env = make_env(cfg.env_id, seed=cfg.seed)
        obs_shape = self.env.observation_space.shape
        assert obs_shape is not None
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = self.env.action_space.n
        self.agent = DQNAgent(self.obs_dim, self.act_dim, agent_cfg, device)
        self.logger = TBLogger(cfg.log_dir)
        self.ckpt_dir = _make_ckpt_dir(cfg)
        self.agent_cfg = agent_cfg

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        ep_return = 0.0
        returns_window = deque(maxlen=20)
        pbar = tqdm(total=self.cfg.total_timesteps, desc=f"DQN {self.cfg.env_id}", dynamic_ncols=True)
        last_loss = 0.0
        for step in range(1, self.cfg.total_timesteps + 1):
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward

            metrics = self.agent.update()
            if metrics:
                self.logger.log(metrics, step)
                last_loss = float(metrics.get("loss/q", last_loss))

            if done:
                self.logger.log({"train/return": ep_return}, step)
                returns_window.append(ep_return)
                ep_return = 0.0
                obs, _ = self.env.reset()

            pbar.update(1)
            if step % self.cfg.log_interval == 0:
                mean_ret = float(np.mean(returns_window)) if len(returns_window) else 0.0
                eps = self.agent.epsilon()
                pbar.set_postfix({
                    "R_mean": f"{mean_ret:6.1f}",
                    "loss_q": f"{last_loss:.3f}",
                    "eps": f"{eps:.3f}",
                    "buf": len(self.agent.buffer),
                })

            if step % self.cfg.eval_interval == 0:
                meta = {
                    "algorithm": self.cfg.algo,
                    "env_id": self.cfg.env_id,
                    "model": "QNetwork",
                    "seed": self.cfg.seed,
                    "step": step,
                    "agent_cfg": asdict(self.agent_cfg),
                }
                ckpt_file = os.path.join(self.ckpt_dir, f"step_{step}.pt")
                # Save online network weights
                save_checkpoint(ckpt_file, self.agent.q, self.agent.optimizer, step, meta=meta)
                copy_as_last(ckpt_file)

        pbar.close()
        self.logger.close()

class PPOTrainer:
    def __init__(self, cfg: TrainCommonConfig, agent_cfg: PPOConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.env = make_env(cfg.env_id, seed=self.cfg.seed)
        obs_shape = self.env.observation_space.shape
        assert obs_shape is not None
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = self.env.action_space.n
        self.agent = PPOAgent(self.obs_dim, self.act_dim, agent_cfg, device)
        self.logger = TBLogger(cfg.log_dir)
        self.ckpt_dir = _make_ckpt_dir(cfg)
        self.agent_cfg = agent_cfg

    def train(self) -> None:
        step = 0
        pbar = tqdm(total=self.cfg.total_timesteps, desc=f"PPO {self.cfg.env_id}", dynamic_ncols=True)
        while step < self.cfg.total_timesteps:
            traj = self.agent.collect_rollout(self.env)
            metrics: Dict[str, float] = self.agent.update(traj)
            step += len(traj["rewards"])
            if metrics:
                self.logger.log(metrics, step)
            rets = _episodic_returns_from_rollout(traj)
            mean_ret = float(np.mean(rets)) if len(rets) else 0.0
            pbar.update(len(traj["rewards"]))
            pbar.set_postfix({
                "R_mean": f"{mean_ret:6.1f}",
                "Lpi": f"{metrics.get('loss/policy', 0.0):.3f}",
                "Lv": f"{metrics.get('loss/value', 0.0):.3f}",
                "H": f"{metrics.get('loss/entropy', 0.0):.3f}",
            })
            if step % self.cfg.eval_interval < len(traj["rewards"]):
                meta = {
                    "algorithm": self.cfg.algo,
                    "env_id": self.cfg.env_id,
                    "model": "ActorCritic",
                    "seed": self.cfg.seed,
                    "step": step,
                    "agent_cfg": asdict(self.agent_cfg),
                }
                ckpt_file = os.path.join(self.ckpt_dir, f"step_{step}.pt")
                save_checkpoint(ckpt_file, self.agent.ac, self.agent.optimizer, step, meta=meta)
                copy_as_last(ckpt_file)
        pbar.close()
        self.logger.close()
