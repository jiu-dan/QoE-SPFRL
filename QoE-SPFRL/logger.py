from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, log_path: str, append: bool = False):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not append and self.log_path.exists():
            self.log_path.unlink()
        self._header_written = append and self.log_path.exists() and self.log_path.stat().st_size > 0

    def log(self, row: dict) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
