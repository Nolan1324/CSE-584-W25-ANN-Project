from time import perf_counter_ns
from datetime import timedelta
from logging import Logger
from itertools import product
from pathlib import Path

from dotenv import load_dotenv


ROOT_PATH = Path(__file__).parent.parent.resolve()
WORKLOAD_PATH = ROOT_PATH / "data" / "workload"
DATASET_PATH = ROOT_PATH / "data" / "datasets"
EXPERIMENT_PATH = ROOT_PATH / "data" / "experiments"
BIN_PATH = ROOT_PATH / "bin"
CONFIG_PATH = ROOT_PATH / "config"

load_dotenv(CONFIG_PATH / ".env")


class Timer:
    def __init__(self, logger: Logger = None, message: str = None) -> None:
        self.start_time = None
        self.duration = None
        self.logger = logger
        self.message = message

    def __enter__(self):
        if self.logger and self.message:
            self.logger.info(f"Starting timer {self.message}...")
        self.start_time = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = perf_counter_ns() - self.start_time
        if self.logger and self.message:
            delta = timedelta(microseconds=self.duration / 1_000)
            self.logger.info(f"Timer {self.message} finished in {delta}")


def expand_experiment_grid(grid: dict) -> list[dict]:
    return [
        dict(zip(grid.keys(), values))
        for values in product(*[
            v
            if isinstance(v, list)
            else [v] for v in grid.values()
        ])
    ]
