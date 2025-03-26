import random
import logging
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

from sift import SiftDataset
from create_db import Creator
from partitioner import RangePartitioner
from attributes import uniform_attributes
from search import Searcher
from utils import Timer


plt.style.use("ggplot")


ROOT_PATH = Path(__file__).parent.parent.resolve()
EXPERIMENT_PATH = ROOT_PATH / "experiments"
DATASET_PATH = ROOT_PATH / "data"
BIN_PATH = ROOT_PATH / "bin"
CONFIG_PATH = ROOT_PATH / "config"

os.chdir(BIN_PATH)
load_dotenv(CONFIG_PATH / ".env")


def run_docker_command(command: str) -> None:
    process = subprocess.run(
        ["sudo", "-S", BIN_PATH / "standalone_embed.sh", command],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        raise RuntimeError(f"Failed to run command: {process.stdout}")


def configure_logging(name: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_dir = EXPERIMENT_PATH / f"{name} ({timestamp})"
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


def run_test(logger: logging.Logger, config: dict, experiment_dir: Path) -> dict:
    dataset = SiftDataset(DATASET_PATH / config["dataset"], config["dataset"])
    attributes = uniform_attributes(dataset.num_base_vecs, 1, config["seed"], int, 0, 1000).flatten()
    
    logger.info("Creating collection...")
    creator = Creator(RangePartitioner([(0, 100), (101, 1000)]))
    creator.create_collection_schema(config["dataset"])
    creator.populate_collection(config["dataset"], dataset, attributes)
    logger.info("Collection created.")
    
    logger.info("Running search...")
    partitioner = RangePartitioner([(0, 100), (101, 1000)])
    searcher = Searcher(config["dataset"], attributes, partitioner)
    times = []
    indices = np.arange(10_000)
    for index in tqdm(indices):
        if attributes[index] > 100:
            continue
        with Timer() as timer:
            searcher.do_search(index, upper_bound=100)
        times.append(timer.duration)
    logger.info(f"Search times: {times}")
    logger.info(f"Average search time: {np.mean(times)}")
    logger.info(f"Median search time: {np.median(times)}")
    logger.info(f"Max search time: {np.max(times)}")
    logger.info(f"Min search time: {np.min(times)}")
    logger.info(f"Standard deviation of search times: {np.std(times)}")
    plt.scatter(np.arange(len(times)), times)
    plt.savefig(experiment_dir / "plot.png", dpi=300)
    logger.info("Search complete.")
    
    return {}


if __name__ == "__main__":
    seed = random.randint(0, 1_000_000)
    test_config = {
        "name": "temp",
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

    logger.info("Starting docker container...")
    run_docker_command("start")
    run_docker_command("stop")
    logger.info("Copying config files to docker container...")
    p = subprocess.run(
        ["sudo", "cp", "-f", CONFIG_PATH / "user.yaml", BIN_PATH / "user.yaml"],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )
    run_docker_command("start")
    logger.info("Docker container started.")

    try:
        results = run_test(logger, test_config, experiment_dir)
        logger.info("Test succeeded.")
        with open(experiment_dir / "results.json", "w") as results_file:
            json.dump(results, results_file, indent=4)
        logger.info("Test saved.")
    except Exception as e:
        logger.exception(e)
        logger.info("Test failed.")
        
    logger.info("Cleaning up docker container...")
    run_docker_command("stop")
    run_docker_command("delete")
    logger.info("Docker container stopped and deleted.")
