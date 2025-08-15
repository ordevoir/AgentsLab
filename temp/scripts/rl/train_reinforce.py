#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

from agentslab.rl.training.train_loop import train

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Pretty-print the active config
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)

if __name__ == "__main__":
    main()
