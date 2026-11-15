"""Training logger: TensorBoard (via tensorboardX) + a CSV that survives a
resume without clobbering the previous run's rows."""

import csv
import time
from pathlib import Path

from tensorboardX import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

        csv_path = self.log_dir / f"progress_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer: csv.DictWriter | None = None
        self._start_time = time.perf_counter()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

        row = {"step": step, "wall_time_s": time.perf_counter() - self._start_time, **metrics}
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        self.writer.close()
        self._csv_file.close()
