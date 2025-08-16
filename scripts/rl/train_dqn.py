# scripts/rl/train_dqn.py
import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from agentslab.rl.training.dqn_trainer import train as train_dqn


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> Any:
    assert cfg.rl.name == "dqn", "This script trains DQN; set defaults to rl: dqn"
    print("\n=== CONFIG ===\n", OmegaConf.to_yaml(cfg))
    results = train_dqn(cfg)
    print("\n=== RESULTS ===\n", results)
    return results


if __name__ == "__main__":
    main()
