from __future__ import annotations
from dataclasses import dataclass
import os
from tqdm import tqdm
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from tensordict import TensorDictBase

from agentslab.storage.collectors import make_sync_collector
from agentslab.storage.buffers import make_replay_buffer
from agentslab.runner.logger import CSVLogger
from agentslab.runner.checkpointer import Checkpointer

@dataclass
class TrainHandles:
    collector: object
    replay: object
    logger: CSVLogger
    checkpointer: Checkpointer

def _batch_frames(td: TensorDictBase, fallback: int) -> int:
    try:
        bs = td.batch_size
        if len(bs) >= 1:
            return int(bs[0])
    except Exception:
        pass
    try:
        return int(td.get(("next", "reward")).shape[0])
    except Exception:
        return int(fallback)

def train(
    *,
    make_env_fn,
    policy,
    q_net,
    loss_module,
    optimizer,
    target_updater,
    device,
    paths,
    env_cfg,
    collector_cfg,
    rb_cfg,
    optim_cfg,
    train_cfg,
    eval_fn=None,
    eval_cfg=None,
):
    paths.ensure()
    train_log_path = os.path.join(paths.logs_path, "train.csv")
    logger = CSVLogger(train_log_path, fieldnames=["frames", "updates", "loss", "eps"])
    ckpt = Checkpointer(paths.ckpt_path)

    collector = make_sync_collector(
        create_env_fn=lambda: make_env_fn(env_cfg.env_id, env_cfg.seed, env_cfg.render_mode),
        policy=policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        init_random_frames=collector_cfg.init_random_frames,
        reset_at_each_iter=collector_cfg.reset_at_each_iter,
        device=device,
    )
    replay = make_replay_buffer(
        size=rb_cfg.size,
        batch_size=rb_cfg.batch_size,
        prefetch=rb_cfg.prefetch,
        pin_memory=rb_cfg.pin_memory,
    )

    handles = TrainHandles(collector=collector, replay=replay, logger=logger, checkpointer=ckpt)

    pbar = tqdm(total=collector_cfg.total_frames, desc="Training", unit="frame", leave=True)
    frames_done = 0
    updates = 0
    last_logged = 0

    try:
        for batch in collector:
            n_steps = _batch_frames(batch, collector_cfg.frames_per_batch)
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                policy[-1].step(n_steps)

            replay.extend(batch)
            frames_done += n_steps
            pbar.update(n_steps)

            loss_val = None
            if len(replay) >= collector_cfg.init_random_frames:
                optimizer.zero_grad(set_to_none=True)
                for _ in range(train_cfg.utd_ratio):
                    td = replay.sample().to(device)
                    loss = loss_module(td)["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), optim_cfg.grad_clip_norm)
                    optimizer.step()
                    target_updater.step()
                    loss_val = float(loss.detach().cpu().item())
                    updates += 1

            eps = getattr(policy[-1], "eps", None)
            if loss_val is not None and frames_done - last_logged >= train_cfg.log_interval_frames:
                logger.log(frames=frames_done, updates=updates, loss=loss_val, eps=eps)
                last_logged = frames_done
                pbar.set_postfix(loss=f"{loss_val:.4f}", eps=f"{eps:.3f}" if eps is not None else "n/a")

            if eval_fn and eval_cfg and (frames_done % max(train_cfg.eval_interval_frames, 1) == 0):
                avg_ret = eval_fn(policy=policy, make_env_fn=make_env_fn, env_cfg=env_cfg, eval_cfg=eval_cfg, device=device)

            if train_cfg.max_train_steps is not None and updates >= train_cfg.max_train_steps:
                break

        ckpt_path = ckpt.save(
            step=frames_done,
            policy=policy,
            q_net=q_net,
            optimizer=optimizer,
            loss_module=loss_module,
            extra={"frames_done": frames_done, "updates": updates},
        )
    finally:
        pbar.close()
        collector.shutdown()
        logger.close()

    return handles
