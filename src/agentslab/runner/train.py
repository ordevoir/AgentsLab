from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.optim import Adam
from tensordict.nn import set_composite_lp_aggregate
from torchrl.objectives import DQNLoss, ClipPPOLoss, ValueEstimators, SoftUpdate
from torchrl.envs import TransformedEnv, RewardSum

from agentslab.storage.collectors import make_sync_collector
from agentslab.storage.buffers import make_replay_buffer_dqn, make_replay_buffer_ppo
from agentslab.runner.logger import LabLogger
from agentslab.runner.checkpointer import Checkpointer, default_run_name

@dataclass
class DQNConfig:
    frames_per_batch: int = 32
    total_frames: int = 20_000
    rb_max_size: int = 50_000
    batch_size: int = 32
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.01
    eps_init: float = 1.0
    eps_end: float = 0.01
    eps_anneal: int = 5_000

def train_dqn(
    create_env_fn,
    q_net,
    action_spec,
    device,
    log_dir: str,
    ckpt_dir: str,
    env_name: str,
    seed: int | None = None,
    cfg: DQNConfig = DQNConfig(),
):
    from agentslab.models.policy import build_dqn_policy
    policy = build_dqn_policy(q_net, action_spec, cfg.eps_init, cfg.eps_end, cfg.eps_anneal)

    collector = make_sync_collector(create_env_fn, policy, cfg.frames_per_batch, cfg.total_frames, device=device)
    rb = make_replay_buffer_dqn(cfg.rb_max_size, cfg.batch_size, device=device)
    loss_module = DQNLoss(value_network=policy[0], loss_function="l2", delay_value=True)
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_updater = SoftUpdate(loss_module, tau=cfg.tau)
    optimizer = Adam(q_net.parameters(), lr=cfg.lr)

    run_name = default_run_name("dqn", env_name, seed)
    logger = LabLogger(log_dir, run_name)
    checkpointer = Checkpointer(ckpt_dir, run_name)

    step = 0
    try:
        for i, data in enumerate(collector):
            rb.extend(data)
            policy[1].step(data.numel())
            if len(rb) >= cfg.batch_size:
                batch = rb.sample().to(device)
                loss = loss_module(batch)["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                target_updater.step()
                logger.log_scalar(step, "loss", float(loss.detach().cpu()))
            step += 1
    finally:
        collector.shutdown()
    checkpointer.save(policy=policy, q_net=q_net, optimizer=optimizer, loss=loss_module)
    logger.close()
    return run_name

@dataclass
class PPOConfig:
    frames_per_batch: int = 6_000
    n_iters: int = 5
    num_epochs: int = 30
    minibatch_size: int = 400
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.9
    entropy_eps: float = 1e-4

def train_marl_ppo(
    env,
    policy,
    critic_value_net,
    device,
    log_dir: str,
    ckpt_dir: str,
    env_name: str,
    seed: int | None = None,
    cfg: PPOConfig = PPOConfig(),
):
    set_composite_lp_aggregate(False).set()

    rb = make_replay_buffer_ppo(cfg.frames_per_batch, cfg.minibatch_size)
    try:
        env = TransformedEnv(env, RewardSum(reward_key=env.reward_key))
    except Exception:
        pass
    collector = make_sync_collector(lambda: env, policy, cfg.frames_per_batch, cfg.frames_per_batch * cfg.n_iters, device=device)

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic_value_net,
        clip_epsilon=cfg.clip_epsilon,
        entropy_coef=cfg.entropy_eps,
        critic_coef=1.0,
        normalize_advantage=False,   # как в туториале TorchRL для MARL
    )

    loss_module.set_keys(
        action=env.action_key,
        reward=env.reward_key,
        value=("agents", "state_value"),   # критик пишет сюда
        # sample_log_prob оставляем по умолчанию: "sample_log_prob"
        # advantage / value_target тоже оставляем по умолчанию:
        # "advantage" и "value_target"
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lmbda
    )

    optimizer = Adam(loss_module.parameters(), lr=cfg.lr)

    run_name = default_run_name("mappo", env_name, seed)
    logger = LabLogger(log_dir, run_name)
    checkpointer = Checkpointer(ckpt_dir, run_name)

    episode_reward_means = []

    for it, tensordict_data in enumerate(collector):
        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        rb.extend(tensordict_data)

        for _ in range(cfg.num_epochs):
            for batch in rb:
                loss_vals = loss_module(batch)
                loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if ("next", "episode_reward") in tensordict_data.keys(True, True):
            ep_rew = tensordict_data.get(("next", "episode_reward"))
            rew_mean = float(ep_rew.mean().cpu())
            episode_reward_means.append(rew_mean)
            logger.log_scalar(it, "episode_reward_mean", rew_mean)

        rb.clear()

    checkpointer.save(policy=policy, optimizer=optimizer, loss=loss_module)
    logger.close()
    return run_name, episode_reward_means
