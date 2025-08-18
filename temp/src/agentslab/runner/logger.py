from __future__ import annotations
import csv, os

class CSVLogger:
    def __init__(self, filepath: str, fieldnames: list[str]):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._file = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, **kwargs) -> None:
        row = {k: kwargs.get(k) for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
