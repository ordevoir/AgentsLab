from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from tqdm.auto import tqdm
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

from torch.optim import Adam
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl._utils import logger as torchrl_logger

from ..envs.gym import make_gym_env
from ..models.value import build_qvalue_actor
from ..models.policy import build_dqn_policy
from ..storage.buffers import make_replay_buffer
from ..storage.collectors import make_sync_collector
from .logger import prepare_logging, log_metrics
from .checkpointer import prepare_checkpoint_dir, save_checkpoint

@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    frames_per_batch: int = 256
    init_random_frames: int = 1_000
    optim_steps_per_batch: int = 1
    total_frames: int = 50_000
    buffer_size: int = 100_000
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.995  # for SoftUpdate
    epsilon_steps: int = 100_000
    eps_init: float = 1.0
    eps_end: float = 0.05
    max_steps: Optional[int] = None
    log_root: str = "logs"
    ckpt_root: str = "checkpoints"
    run_name: str = "dqn"

def train_dqn(config: DQNConfig, device: torch.device, record_video: bool=False) -> Dict[str, Any]:
    # Logging / checkpoints
    logs = prepare_logging(config.log_root, config.run_name, with_tensorboard=True, video=record_video)
    ckpt = prepare_checkpoint_dir(config.ckpt_root, config.run_name)

    # Env
    env = make_gym_env(config.env_id, seed=config.seed, record_video=False, max_steps=config.max_steps)
    env.set_seed(config.seed)
    # Specs
    # Action-space size
    n_actions = getattr(getattr(env.action_spec, 'space', None), 'n', None)
    if n_actions is None:
        n_actions = env.action_spec.shape[-1]

    # Policy / value
    value_net, q_head = build_qvalue_actor(n_actions, hidden=(256,256))
    greedy, explore = build_dqn_policy(value_net, q_head, action_spec=env.action_spec,
                                       epsilon_steps=config.epsilon_steps, eps_init=config.eps_init, eps_end=config.eps_end)
    value_net.to(device)
    greedy.to(device)
    explore.to(device)

    # Collector / Replay
    collector = make_sync_collector(env, explore,
                                    frames_per_batch=config.frames_per_batch,
                                    init_random_frames=config.init_random_frames,
                                    total_frames=config.total_frames)
    rb = make_replay_buffer(config.buffer_size, prioritized=False)

    # Loss + Optim + Target updater
    loss_module = DQNLoss(value_network=greedy, action_space=env.action_spec, gamma=config.gamma,
                          delay_value=True, double_dqn=True)
    optimizer = Adam(loss_module.parameters(), lr=config.lr)
    target_updater = SoftUpdate(loss_module, eps=config.tau)

    pbar = tqdm(total=config.total_frames, desc="Training", unit="frame")
    frames = 0
    episodes = 0
    best_eval = float("-inf")

    for batch in collector:
        # batch: TensorDict with [time, env, ...]
        # Push to replay
        rb.extend(batch)

        # Optimization steps
        for _ in range(config.optim_steps_per_batch):
            if len(rb) < config.batch_size:
                break
            td = rb.sample(config.batch_size).to(device)
            loss_td = loss_module(td)
            loss = loss_td["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            target_updater.step()

        # Metrics
        frames += batch.numel()
        info = {
            "loss": float(loss.detach().cpu()) if 'loss' in locals() else float('nan'),
            "frames": frames,
            "episodes": episodes,
            "epsilon": float(explore[-1].eps.item()) if hasattr(explore[-1], "eps") else float('nan'),
        }
        log_metrics(logs, info, step=frames)
        pbar.set_postfix(loss=info["loss"], eps=info["epsilon"])
        pbar.update(batch.numel())

        # stop condition
        if 0 < config.total_frames <= frames:
            break

    pbar.close()
    collector.shutdown()
    env.close()

    # Save checkpoint
    state = {
        "value_net": value_net.state_dict(),
        "loss_module": loss_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(config),
        "frames": frames,
    }
    path = save_checkpoint(ckpt, f"{config.env_id}_DQN_seed{config.seed}_frames{frames}.pt", state)
    return {"log_dir": logs.log_dir, "ckpt_path": path}
