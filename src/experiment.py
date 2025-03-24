import random
import logging
import json
from pathlib import Path
from datetime import datetime

from sift import SiftDataset
from create_db import Creator
from partitioner import RangePartitioner
from attributes import uniform_attributes


EXPERIMENT_PATH = Path(__file__).parent.parent / "experiments"
DATASET_PATH = Path(__file__).parent.parent / "data"


def configure_logging(name: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("_%d_%H:%M:%S")
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


def run_test(config: dict) -> dict:
    dataset = SiftDataset(DATASET_PATH / test_config["dataset"], test_config["dataset"])
    attributes = uniform_attributes(dataset.num_base_vecs, 1, test_config["seed"], int, 0, 100).flatten()
    
    partitioner = RangePartitioner([(0, 100), (101, 1000)])
    creator = Creator(partitioner)
    creator.create_collection_schema(test_config["name"])
    creator.populate_collection(test_config["name"], dataset, attributes)
    
    return {}


if __name__ == "__main__":
    seed = random.randint(0, 1_000_000)
    test_config = {
        "name": "test",
        "dataset": "sift",
        "partitioner": "range",
        "n_partitions": 10,
        "vector_index": "hnsw",
        "filter_strategy": "first",
        "query_selectivity": 0.1,
        "seed": seed,
    }
    
    logger, experiment_dir = configure_logging(test_config["name"])
    with open(experiment_dir / "config.json", "w") as config_file:
        json.dump(test_config, config_file, indent=4)
    
    try:
        logger.info("Starting test...")
        results = run_test(test_config)
        logger.info("Test succeeded.")
        with open(experiment_dir / "results.json", "w") as results_file:
            json.dump(results, results_file, indent=4)
        logger.info("Test saved.")
    except Exception as e:
        logger.exception(e)
        logger.info("Test failed.")
