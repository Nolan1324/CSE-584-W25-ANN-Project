import logging
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Callable

import numpy as np

from sift import Dataset, load_sift_1b, load_sift_1m, load_sift_small
from create_db import Creator
from partitioner import RangePartitioner, ModPartitioner
from utils import expand_experiment_grid, BIN_PATH, CONFIG_PATH, EXPERIMENT_PATH, DATASET_PATH


@contextmanager
def docker_container():
    os.chdir(BIN_PATH)
    run_docker_command("stop", True)
    run_docker_command("delete", True)
    run_docker_command("start")
    run_docker_command("stop")
    subprocess.run(
        ["sudo", "cp", "-f", CONFIG_PATH / "user.yaml", BIN_PATH / "user.yaml"],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )
    run_docker_command("start")
    try:
        yield
    except Exception as e:
        print("Interrupted")
        raise e
    finally:
        print("Cleaning up...")
        run_docker_command("stop")
        run_docker_command("delete")


def run_experiment(name: str, config: dict, experiment: Callable) -> None:
    schemas = expand_experiment_grid(config["schemas"])
    workflows = expand_experiment_grid(config["workflows"])
    n_experiments = len(schemas) * len(workflows) * config["trials"]
    
    experiment_dir = EXPERIMENT_PATH / f"{name} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    experiment_dir.mkdir()
    
    with open(experiment_dir / "config.json", "w") as config_file:
        json.dump(config | {
            "expanded_schemas": schemas,
            "expanded_workflows": workflows,
        }, config_file, indent=4)
    
    current_experiment = 0
    for trial_i in range(config["trials"]):
        for schema_i, schema in enumerate(schemas):
            db_logger = configure_logging(experiment_dir, f"trial{trial_i}_schema{schema_i}_db")
            with docker_container():
                print(f"Configuring schema {schema_i + 1}/{len(schemas)} and trial {trial_i + 1}/{config['trials']}...")
                partitioner, attributes = setup_db(db_logger, schema, config["dataset"])
                for workflow_i, workflow in enumerate(workflows):
                    current_experiment += 1
                    print(f"Running workflow {workflow_i + 1}/{len(workflows)} (experiment {current_experiment}/{n_experiments})...")
                    trial_name = f"trial{trial_i}_schema{schema_i}_workflow{workflow_i}"
                    logger = configure_logging(experiment_dir, trial_name)
                    results = experiment(logger, schema, workflow, config["dataset"], partitioner, attributes)
                    with open(experiment_dir / f"{trial_name}.json", "w") as results_file:
                        json.dump(results, results_file, indent=4)


def load_dataset(config: dict, base: bool) -> Dataset:
    if config["name"] == "sift_1b":
        return load_sift_1b(DATASET_PATH / "sift1b", config["size"], base=base)
    elif config["name"] == "sift":
        return load_sift_1m(DATASET_PATH / "sift", base=base)
    elif config["name"] == "siftsmall":
        return load_sift_small(DATASET_PATH / "siftsmall", base=base)


def run_docker_command(command: str, ignore_errors: bool = False) -> None:
    process = subprocess.run(
        ["sudo", "-S", BIN_PATH / "standalone_embed.sh", command],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )

    if process.returncode != 0 and not ignore_errors:
        raise RuntimeError(f"Failed to run command: {process.stdout}")


def configure_logging(experiment_path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("{asctime} | {name:<20} | {levelname:<8} | {message}", style="{")

    file_handler = logging.FileHandler(experiment_path / f"{name}.log", mode="w")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    
    logger.info(f"Logging to {experiment_path / f'{name}.log'}")

    return logger


def setup_db(logger: logging.Logger, schema_config: dict, dataset_config: dict) -> tuple[RangePartitioner, np.ndarray]:
    dataset = load_dataset(dataset_config, base=True)
    attributes = np.random.default_rng().uniform(0, dataset_config["max_attribute"], dataset.num_base_vecs).astype(int)
    logger.info(f"Loaded dataset: {dataset}")
    
    if schema_config["partitioner"] == "range":
        partitions = [
            (round(i), round(i + dataset_config["max_attribute"] // schema_config["n_partitions"] - 1))
            for i in np.linspace(0, dataset_config["max_attribute"], schema_config["n_partitions"], endpoint=False)
        ]
        partitioner = RangePartitioner(partitions)
        logger.info(f"Using range partitioner with partitions: {partitions}")
    elif schema_config["partitioner"] == "mod":
        partitioner = ModPartitioner(schema_config["n_partitions"])
    
    creator = Creator(partitioner, datatype=dataset.datatype, logger=logger)
    creator.create_collection_schema()
    creator.populate_collection(dataset, attributes, index_type=schema_config["index"])
    
    return partitioner, attributes
