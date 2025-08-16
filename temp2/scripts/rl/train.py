from __future__ import annotations
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from agentslab.core.seeding import set_seed
from agentslab.rl.training.trainers import (
    TrainCommonConfig, ReinforceTrainer, DQNTrainer, PPOTrainer,
)
from agentslab.rl.agents.reinforce import ReinforceConfig
from agentslab.rl.agents.dqn import DQNConfig
from agentslab.rl.agents.ppo import PPOConfig

@hydra.main(config_path='../../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.utils import get_original_cwd
    import os
    # Merge common config into our dataclass
    root = get_original_cwd()
    ckpt = os.path.join(root, cfg.common.train.checkpoint_path)
    log_dir = os.path.join(root, cfg.common.logging.log_dir)
    root = get_original_cwd()
    ckpt_root = os.path.join(root, cfg.common.train.ckpt_root)
    algo = cfg.rl.algorithm
    common = TrainCommonConfig(
        env_id=cfg.common.env.env_id,
        seed=cfg.common.train.seed,
        total_timesteps=cfg.common.train.total_timesteps,
        eval_interval=cfg.common.train.eval_interval,
        log_interval=cfg.common.train.log_interval,
        checkpoint_path=ckpt,
        log_dir=log_dir,
        algo=algo,
        ckpt_root=ckpt_root,
        run_name=cfg.common.train.run_name,
    )
    set_seed(common.seed, deterministic=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if algo == 'reinforce':
        acfg = ReinforceConfig(**OmegaConf.to_container(cfg.rl.reinforce, resolve=True))
        trainer = ReinforceTrainer(common, acfg, device)
    elif algo == 'dqn':
        acfg = DQNConfig(**OmegaConf.to_container(cfg.rl.dqn, resolve=True))
        trainer = DQNTrainer(common, acfg, device)
    elif algo == 'ppo':
        acfg = PPOConfig(**OmegaConf.to_container(cfg.rl.ppo, resolve=True))
        trainer = PPOTrainer(common, acfg, device)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    trainer.train()

if __name__ == '__main__':
    main()
