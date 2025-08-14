from __future__ import annotations
import hydra
from omegaconf import DictConfig
import torch
import gymnasium as gym

from agentslab.environments.gym_wrapper import EnvConfig, make_env
from agentslab.networks.mlp import MLP
from agentslab.agents.reinforce_agent import ReinforcePolicy
from agentslab.training.trainer import Trainer, TrainCfg
from agentslab.training.callbacks import EarlySolvedStop, TBLogger, CSVLogger, BestCheckpoint
from agentslab.utils.logging import make_logger, make_tb_writer, make_csv_writer
from agentslab.utils.checkpoints import save_checkpoint
from agentslab.training.evaluator import evaluate as eval_loop


def env_factory(env_cfg: EnvConfig):
    def _make():
        return make_env(env_cfg)
    return _make


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = make_logger()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env (determine dims)
    env_cfg = EnvConfig(**cfg.env)
    env = make_env(env_cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Policy + optimizer
    net = MLP(obs_dim, act_dim, hidden=cfg.net.hidden)
    policy = ReinforcePolicy(net, device=device).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.train.lr)

    # Callbacks & logging
    callbacks = []
    writer = None
    if cfg.logging.enable_tensorboard:
        writer = make_tb_writer(cfg.logging.tb_dir)
        callbacks.append(TBLogger(writer=writer, log_every=cfg.train.log_every))
    csv_writer = None
    if cfg.logging.enable_csv:
        csv_writer = make_csv_writer(cfg.logging.csv_path)
        callbacks.append(CSVLogger(csv_writer))

    if cfg.train.save_best:
        def _save(tag: str, **meta):
            save_checkpoint(policy, f"checkpoints/{tag}.pt", **meta)
        callbacks.append(BestCheckpoint(save_fn=_save, key="train/return", mode="max"))

    # Trainer
    trainer = Trainer(
        policy=policy,
        env=env,
        optimizer=optimizer,
        algo_cfg=cfg.algo,
        logger=logger,
        callbacks=callbacks,
        eval_fn=lambda p, n: eval_loop(env_factory(env_cfg), p, episodes=n)
    )

    returns = trainer.train(
        cfg=TrainCfg(
            episodes=cfg.train.episodes,
            lr=cfg.train.lr,
            log_every=cfg.train.log_every,
            solved_threshold=cfg.train.solved_threshold,
            solved_window=cfg.train.solved_window,
            eval_every=cfg.train.eval_every,
            eval_episodes=cfg.train.eval_episodes,
        ),
        seed=cfg.seed,
    )

    # Финальный чекпоинт
    save_checkpoint(policy, "checkpoints/last.pt", episodes=len(returns))

    if writer:
        writer.flush(); writer.close()


if __name__ == "__main__":
    main()