from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import torch
import random, os
import numpy as np




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

