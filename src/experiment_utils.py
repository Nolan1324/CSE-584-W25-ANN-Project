import logging
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sift import SiftDataset
from create_db import Creator
from partitioner import RangePartitioner, ModPartitioner
from attributes import uniform_attributes


plt.style.use("ggplot")


ROOT_PATH = Path(__file__).parent.parent.resolve()
DATASET_PATH = ROOT_PATH / "data"
EXPERIMENT_PATH = DATASET_PATH / "experiments"
BIN_PATH = ROOT_PATH / "bin"
CONFIG_PATH = ROOT_PATH / "config"

os.chdir(BIN_PATH)
load_dotenv(CONFIG_PATH / ".env")


def run_docker_command(command: str, ignore_errors: bool = False) -> None:
    process = subprocess.run(
        ["sudo", "-S", BIN_PATH / "standalone_embed.sh", command],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )

    if process.returncode != 0 and not ignore_errors:
        raise RuntimeError(f"Failed to run command: {process.stdout}")


def configure_logging(name: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_dir = EXPERIMENT_PATH / f"{name} ({timestamp})"
    experiment_dir.mkdir()
    
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("{asctime} | {name} | {levelname:<8} | {message}", style="{")

    file_handler = logging.FileHandler(experiment_dir / f"{name}.log", mode="w")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    
    logger.info(f"Logging to {experiment_dir / f'{name}.log'}")

    return logger, experiment_dir


def setup_db(logger: logging.Logger, config: dict) -> tuple[RangePartitioner, np.ndarray]:
    dataset = SiftDataset(DATASET_PATH / "datasets" / config["dataset"], config["dataset"])
    key_max = config["key_max"]
    attributes = uniform_attributes(dataset.num_base_vecs, 1, config["seed"], int, 0, key_max).flatten()
    
    step = 1000 // config["n_partitions"]
    partitions = [(round(i), round(i + step - 1)) for i in np.linspace(0, key_max, config["n_partitions"], endpoint=False)]
    if (config["partitioner"] == "range"):
        partitioner = RangePartitioner(partitions)
    else:
        partitioner = ModPartitioner(config["n_partitions"])
    logger.info(f"Partitions ({len(partitions)}): {partitions}")
    
    creator = Creator(partitioner)
    creator.create_collection_schema(config["dataset"], index_type=config["vector_index"])
    logger.info("Collection schema created.")
    creator.populate_collection(config["dataset"], dataset, attributes)
    logger.info("Collection created.")
    
    return partitioner, attributes


def run_test(config: dict) -> None:
    plt.close("all")
    
    logger, experiment_dir = configure_logging(config["name"])
    with open(experiment_dir / "config.json", "w") as config_file:
        printable_test_config = config.copy()
        printable_test_config["test_function"] = config["test_function"].__name__
        json.dump(printable_test_config, config_file, indent=4)

    logger.info("Starting docker container...")
    run_docker_command("stop", True)
    run_docker_command("delete", True)
    run_docker_command("start")
    run_docker_command("stop")
    logger.info("Copying config files to docker container...")
    subprocess.run(
        ["sudo", "cp", "-f", CONFIG_PATH / "user.yaml", BIN_PATH / "user.yaml"],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )
    run_docker_command("start")
    logger.info("Docker container started.")
    logger.info("Dashboard: http://127.0.0.1:9091/webui/")

    try:
        results = config["test_function"](logger, config, experiment_dir)
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
