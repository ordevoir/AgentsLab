from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random
import numpy as np
import torch

from ..utils.metrics import moving_mean


@dataclass
class TrainCfg:
    episodes: int
    lr: float
    log_every: int
    solved_threshold: float
    solved_window: int
    eval_every: int
    eval_episodes: int
    grad_clip: float | None = 1.0


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, policy, env, optimizer, algo_cfg, logger, callbacks: List, eval_fn):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer
        self.algo_cfg = algo_cfg
        self.logger = logger
        self.callbacks = callbacks
        self.eval_fn = eval_fn
        self.stop_training = False

    def train(self, cfg: TrainCfg, seed: int) -> List[float]:
        set_global_seeds(seed)
        returns: List[float] = []
        for ep in range(cfg.episodes):
            if self.stop_training:
                break
            obs, info = self.env.reset(seed=seed + ep)
            self.policy.train()
            self.policy.buffers.clear()
            ep_return = 0.0

            while True:
                action = self.policy.act_train(obs)
                obs, rew, terminated, truncated, info = self.env.step(action)
                self.policy.buffers.rewards.append(float(rew))
                ep_return += float(rew)
                if terminated or truncated:
                    break

            loss, _ = self.policy.compute_loss_and_rtg(self.algo_cfg.gamma)
            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
            self.optimizer.step()

            returns.append(ep_return)
            metrics = {
                "train/loss": loss.item(),
                "train/return": ep_return,
                "train/mean_return@{}".format(min(cfg.solved_window, len(returns))): float(np.mean(returns[-cfg.solved_window:]))
            }
            for cb in self.callbacks:
                cb.on_episode_end(ep=ep, metrics=metrics, returns_history=returns, trainer=self)

            if (ep + 1) % cfg.eval_every == 0:
                eval_rets = self.eval_fn(self.policy, cfg.eval_episodes)
                self.logger.info(f"Eval mean return: {np.mean(eval_rets):.1f}")

        self.logger.info("Training finished.")
        return returns
    
    