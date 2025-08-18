from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from torchrl.objectives.value import GAE
from tensordict import TensorDict

from ..storage.collectors import CollectorConfig, build_sync_collector
from ..storage.buffers import BufferConfig, build_onpolicy_buffer
from ..modules.estimators import GAEConfig, build_gae
from .objectives import PPOLossConfig, build_ppo_loss
from .schedulers import SchedulerConfig, build_scheduler

@dataclass
class PPOTrainerConfig:
    frames_per_batch: int = 2048
    total_frames: int = 1_000_000
    sub_batch_size: int = 64
    num_epochs: int = 10
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 1e-4
    seed: int = 0
    device: Union[str, torch.device] = "cpu"
    eval_every: int = 10

class PPOTrainer:
    def __init__(self, cfg: PPOTrainerConfig, env, actor, critic, run_dirs, logger):
        self.cfg = cfg
        self.env = env
        self.actor = actor
        self.critic = critic
        self.run_dirs = run_dirs
        self.logger = logger

        # Collector
        coll_cfg = CollectorConfig(
            frames_per_batch=cfg.frames_per_batch,
            total_frames=cfg.total_frames,
            split_trajs=False,
            device=cfg.device,
        )
        self.collector = build_sync_collector(coll_cfg, env, actor)

        # Buffer
        self.buffer = build_onpolicy_buffer(BufferConfig(frames_per_batch=cfg.frames_per_batch))

        # GAE and PPO loss
        self.gae = build_gae(GAEConfig(gamma=cfg.gamma, lam=cfg.lam, average_gae=True), self.critic)
        self.loss_module = build_ppo_loss(
            PPOLossConfig(clip_epsilon=cfg.clip_epsilon, entropy_coef=cfg.entropy_coef),
            self.actor,
            self.critic,
        )

        # Optim and scheduler
        self.optimizer = Adam(self.loss_module.parameters(), lr=cfg.lr)
        outer_iters = cfg.total_frames // cfg.frames_per_batch
        self.scheduler = build_scheduler(self.optimizer, SchedulerConfig(name="cosine", T_max=outer_iters))

    def train(self, evaluator=None):
        cfg = self.cfg
        device = cfg.device
        iter_idx = 0

        for tensordict_data in tqdm(self.collector, total=cfg.total_frames // cfg.frames_per_batch, desc="Collect/Train"):
            iter_idx += 1
            # Recompute GAE with current value network
            with torch.no_grad():
                self.gae(tensordict_data)

            # Flatten and move to CPU for buffer
            if hasattr(self.buffer, 'empty'):
                self.buffer.empty()
            elif hasattr(self.buffer, 'clear'):
                self.buffer.clear()
            else:
                # recreate buffer if no clear/empty method is available
                from ..storage.buffers import BufferConfig, build_onpolicy_buffer
                self.buffer = build_onpolicy_buffer(BufferConfig(frames_per_batch=cfg.frames_per_batch))
            data = tensordict_data.reshape(-1).cpu()
            self.buffer.extend(data)

            # epochs over on-policy batch
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0

            for epoch in range(cfg.num_epochs):
                batch_iter = cfg.frames_per_batch // cfg.sub_batch_size
                for _ in range(batch_iter):
                    subdata = self.buffer.sample(cfg.sub_batch_size)
                    loss_vals = self.loss_module(subdata)
                    loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += loss_vals["loss_objective"].item()
                    total_value_loss += loss_vals["loss_critic"].item()
                    total_entropy += (-loss_vals["loss_entropy"].item())  # entropy bonus is negative in loss

            if self.scheduler:
                self.scheduler.step()

            # Train metrics
            mean_reward = tensordict_data.get(("next","reward")).mean().item()
            max_steps = tensordict_data.get(("next","step_count")).max().item()
            lr = self.optimizer.param_groups[0]["lr"]
            metrics = {
                "iter": iter_idx,
                "train_reward_mean": mean_reward,
                "train_steps_max": max_steps,
                "policy_loss": total_policy_loss / (cfg.num_epochs),
                "value_loss": total_value_loss / (cfg.num_epochs),
                "entropy": total_entropy / (cfg.num_epochs),
                "lr": lr,
            }
            
            # ensure eval keys always exist so CSV header includes them from the first row
            metrics.setdefault('eval_return', float('nan'))
            metrics.setdefault('eval_steps', 0)
            # Optional evaluation
            if evaluator and (iter_idx % cfg.eval_every == 0):
                eval_res = evaluator()
                metrics.update(eval_res)

            self.logger.log(metrics)

        self.logger.close()
