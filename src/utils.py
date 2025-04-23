from __future__ import annotations

from time import perf_counter_ns
from datetime import timedelta
from logging import Logger
from itertools import product
from pathlib import Path

from dotenv import load_dotenv


# constants that are useful to traverse the directories
ROOT_PATH = Path(__file__).parent.parent.resolve()
WORKLOAD_PATH = ROOT_PATH / "data" / "workload"
DATASET_PATH = ROOT_PATH / "data" / "datasets"
EXPERIMENT_PATH = ROOT_PATH / "data" / "experiments"
BIN_PATH = ROOT_PATH / "bin"
CONFIG_PATH = ROOT_PATH / "config"

# load the user password, which is needed to run trials
load_dotenv(CONFIG_PATH / ".env")


class Timer:
    """A context manager for measuring the execution time of a code block.

    Attributes:
        logger: An optional logger instance for logging timer messages.
        message: An optional message to include in the log output.
        start_time: The start time of the timer in nanoseconds.
        duration: The duration of the timer in nanoseconds.
    """

    def __init__(self, logger: Logger = None, message: str = None) -> None:
        self.start_time = None
        self.duration = None
        self.logger = logger
        self.message = message

    def __enter__(self) -> Timer:
        if self.logger and self.message:
            self.logger.info(f"Starting timer {self.message}...")
        self.start_time = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration = perf_counter_ns() - self.start_time
        if self.logger and self.message:
            delta = timedelta(microseconds=self.duration / 1_000)
            self.logger.info(f"Timer {self.message} finished in {delta}")


def expand_experiment_grid(grid: dict) -> list[dict]:
    """
    Expands a grid of parameters into a list of all possible combinations.
    This function takes a dictionary where the keys represent parameter names
    and the values are either single values or lists of possible values for
    those parameters. It generates a list of dictionaries, where each dictionary
    represents a unique combination of the parameter values.

    Args:
        grid: A dictionary where keys are parameter names and values are either single values or lists of
        possible values.

    Returns:
        A list of dictionaries, where each dictionary represents a unique combination of parameter values
        from the input grid.

    Example (based on sklearn experiment grids):
        >>> grid = {
        ...     "learning_rate": [0.01, 0.1],
        ...     "batch_size": [16, 32],
        ...     "optimizer": "adam"
        ... }
        >>> expand_experiment_grid(grid)
        [
            {'learning_rate': 0.01, 'batch_size': 16, 'optimizer': 'adam'},
            {'learning_rate': 0.01, 'batch_size': 32, 'optimizer': 'adam'},
            {'learning_rate': 0.1, 'batch_size': 16, 'optimizer': 'adam'},
            {'learning_rate': 0.1, 'batch_size': 32, 'optimizer': 'adam'}
        ]
    """

    return [
        dict(zip(grid.keys(), values))
        for values in product(*[v if isinstance(v, list) else [v] for v in grid.values()])
    ]
