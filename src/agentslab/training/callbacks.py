from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


class Callback:
    def on_train_start(self, **kwargs):
        pass
    def on_episode_end(self, **kwargs):
        pass
    def on_eval_end(self, **kwargs):
        pass
    def on_train_end(self, **kwargs):
        pass


@dataclass
class EarlySolvedStop(Callback):
    window: int
    threshold: float
    def on_episode_end(self, returns_history, trainer, **kwargs):
        if len(returns_history) >= self.window:
            if float(np.mean(returns_history[-self.window:])) >= self.threshold:
                trainer.stop_training = True
                trainer.logger.info(
                    f"Solved: mean@{self.window} >= {self.threshold}, stopping early.")


@dataclass
class CSVLogger(Callback):
    csv_writer: any
    def on_episode_end(self, ep, metrics: Dict, **kwargs):
        row = {"episode": ep, **metrics}
        self.csv_writer.writerow(row)


@dataclass
class TBLogger(Callback):
    writer: any
    log_every: int
    def on_episode_end(self, ep, metrics: Dict, **kwargs):
        if (ep + 1) % self.log_every == 0:
            for k, v in metrics.items():
                self.writer.add_scalar(k, float(v), global_step=ep + 1)


@dataclass
class BestCheckpoint(Callback):
    save_fn: any
    key: str = "return"
    mode: str = "max"
    best: Optional[float] = None
    def on_episode_end(self, ep, metrics: Dict, **kwargs):
        cur = float(metrics.get(self.key, float("nan")))
        if np.isnan(cur):
            return
        if self.best is None or (self.mode == "max" and cur > self.best) or (self.mode == "min" and cur < self.best):
            self.best = cur
            self.save_fn(tag="best", episode=ep, metric=self.key, value=cur)


