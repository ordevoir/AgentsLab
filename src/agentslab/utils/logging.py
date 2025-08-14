from __future__ import annotations
import csv
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def make_logger() -> logging.Logger:
    logger = logging.getLogger("rl_lab")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(h)
    return logger


def make_tb_writer(path: str | Path) -> SummaryWriter:
    return SummaryWriter(log_dir=str(path))


def make_csv_writer(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("w", newline="")
    writer = csv.DictWriter(f, fieldnames=["episode", "train/loss", "train/return", "train/mean_return@20"])  # базовые поля
    writer.writeheader()
    return writer