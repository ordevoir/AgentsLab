# scripts/rl/train_reinforce.py
import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from agentslab.rl.training.reinforce_trainer import train as train_reinforce


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> Any:
    assert cfg.rl.name == "reinforce", "This script trains REINFORCE; set defaults to rl: reinforce"
    print("\n=== CONFIG ===\n", OmegaConf.to_yaml(cfg))
    results = train_reinforce(cfg)
    print("\n=== RESULTS ===\n", results)
    return results


if __name__ == "__main__":
    main()
