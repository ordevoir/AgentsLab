
from pathlib import Path
from typing import Optional
from hydra.utils import get_original_cwd

def project_root() -> Path:
    """Return absolute path to repo root (original CWD before Hydra changes it)."""
    return Path(get_original_cwd())

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def logs_dir(*parts: str) -> Path:
    return ensure_dir(project_root() / "logs" / Path(*parts))

def ckpts_dir(*parts: str) -> Path:
    return ensure_dir(project_root() / "checkpoints" / Path(*parts))

def results_dir(*parts: str) -> Path:
    return ensure_dir(project_root() / "results" / Path(*parts))
