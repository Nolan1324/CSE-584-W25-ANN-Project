import random
import logging
from pathlib import Path
from datetime import datetime


EXPERIMENT_PATH = Path(__file__).parent.parent / "experiments"


def configure_logging(name: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("_%H:%M:%S")
    experiment_dir = EXPERIMENT_PATH / f"{name}{timestamp}"
    experiment_dir.mkdir()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("{asctime} | {name} | {levelname:<8} | {message}", style="{")

    file_handler = logging.FileHandler(experiment_dir / f"{name}.log", mode="w")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    
    logger.info(f"Logging to {experiment_dir / f'{name}.log'}")

    return logger, experiment_dir


if __name__ == "__main__":
    name = "test"
    logger, experiment_dir = configure_logging(name)
    
    seed = random.randint(0, 1_000_000)
    logger.info(f"Seed: {seed}")
