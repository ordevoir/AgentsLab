from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import torch
import random, os
import numpy as np


@dataclass
class RunPaths:
    run_name: str
    root: Path
    run_dir: Path
    ckpt_dir: Path
    csv_train: Path
    csv_eval: Path
    tb_train: Path
    tb_eval: Path
    fig_dirs: Path
    meta_yaml: Path

@dataclass
class GeneralConfigs:
    root: Path
    algo_name: str
    env_id: str
    env_name: str = None
    device: str = "cpu"
    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        if self.env_name is None:
            self.env_name = self.env_id


def resolve_device(preferred: str | torch.device = "cuda"):

    if preferred.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device(preferred)
        else:
            print("warning: cuda is not abvailable! cpu is using")
            return torch.device("cpu")
    if preferred.lower() == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError('Select "cuda" or "cpu"!')


def set_global_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_paths(root: Path, algo_name: str, env_name: str) -> RunPaths:
    """
    Формирует пути по схеме:
    root/runs/<algo>_<env>_<YYYYMMDD_HHMMSS>/
    Директории не создаёт — только возвращает объект RunPaths.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo_name}_{env_name}_{ts}"
    run_dir = root / "runs" / run_name

    # На случай редкого совпадения по секунде — гарантируем уникальность
    if run_dir.exists():
        i = 2
        while (run_dir / f"{run_name}__{i}").exists():
            i += 1
        run_dir = run_dir / f"{run_name}__{i}"

    return RunPaths(
        run_name=run_name,
        root=root,
        run_dir=run_dir,
        ckpt_dir=run_dir / "checkpoints",
        csv_train=run_dir / "csv_logs" / "train.csv",
        csv_eval=run_dir / "csv_logs" / "eval.csv",
        tb_train=run_dir / "tb_logs" / "train",
        tb_eval=run_dir / "tb_logs" / "eval",
        fig_dirs=run_dir / "figures",
        meta_yaml=run_dir / "meta_info.yaml",
    )


def restore_paths(root: Path, run_name: str) -> RunPaths:
    """
    По root и run_name воссоздаёт объект RunPaths.
    Директории не создаёт.

    Пример:
        rp = restore_paths(Path("/data/project"), "ppo_CartPole-v1_20250101_120000")
    """
    run_dir = root / "runs" / run_name
    return RunPaths(
        run_name=run_name,
        root=root,
        run_dir=run_dir,
        ckpt_dir=run_dir / "checkpoints",
        csv_train=run_dir / "csv_logs" / "train.csv",
        csv_eval=run_dir / "csv_logs" / "eval.csv",
        tb_train=run_dir / "tb_logs" / "train",
        tb_eval=run_dir / "tb_logs" / "eval",
        fig_dirs=run_dir / "figures",
        meta_yaml=run_dir / "meta_info.yaml",
    )


# -----------------------------------------------------------------


from contextlib import contextmanager
from tqdm.auto import tqdm

@contextmanager
def progress_bar(total_frames: int, desc: str = "train"):
    pbar = tqdm(
        total=total_frames,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        unit="frames",
        unit_scale=True,
        smoothing=0.1,
    )
    try:
        yield pbar
    finally:
        pbar.close()

