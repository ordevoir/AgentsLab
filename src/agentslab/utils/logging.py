from __future__ import annotations
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
from .paths import logs_dir

def make_tb_writer(run_name: str, subdir: str) -> SummaryWriter:
    """Create a SummaryWriter under <project_root>/logs/<subdir>/<run_name>.
    Safe against Hydra changing CWD via get_original_cwd() used in logs_dir().
    You can override base logs dir by setting AGENTSLAB_LOG_DIR env var.
    """
    base = Path(os.environ.get("AGENTSLAB_LOG_DIR", logs_dir().as_posix()))
    log_dir = base / subdir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
