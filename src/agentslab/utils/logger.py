from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import csv, os, time



@dataclass
class CSVLogger:
    log_dir: str
    filename: str = "train_log.csv"

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.path = os.path.join(self.log_dir, self.filename)
        self._file = None
        self._writer = None
        self._header_written = False

    def log(self, metrics: Dict[str, Any]):
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics.keys()), extrasaction='ignore')
            self._writer.writeheader()
            self._header_written = True
        self._writer.writerow(metrics)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
