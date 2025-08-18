from __future__ import annotations
import csv, os, logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class LabLogger:
    def __init__(self, log_dir: str | os.PathLike, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.log_dir / f"{run_name}.csv"
        self.tb_dir = self.log_dir / f"{run_name}_tb"
        self.tb = SummaryWriter(self.tb_dir.as_posix())
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(run_name)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                import csv as _csv
                _csv.writer(f).writerow(["step", "key", "value"])

    def log_scalar(self, step: int, key: str, value: float):
        self.tb.add_scalar(key, value, step)
        with self.csv_path.open("a", newline="") as f:
            import csv as _csv
            _csv.writer(f).writerow([step, key, float(value)])
        self.logger.info(f"{key}={value:.4f} @ {step}")

    def close(self):
        self.tb.flush()
        self.tb.close()
