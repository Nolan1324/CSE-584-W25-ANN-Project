"""
This module provides utilities for running experiments with synthetic datasets, configurable schemas,
workflows, and includes tools for Docker management, logging, database setup, and automated testing of
partitioning and workload characterization techniques.
"""

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from create_db import Creator
from multirange_partitioner import MultiRangePartitioner
from predicates import Range
from sift import Dataset, load_sift_1b, load_sift_1m, load_sift_small
from tree_partition_algo import TreeAlgoParams, build_tree
from utils import BIN_PATH, CONFIG_PATH, DATASET_PATH, EXPERIMENT_PATH, expand_experiment_grid
from workload import Workload
from workload_char import counter_characterize_workload


@contextmanager
def docker_container():
    """A context manager set up to create and destroy a docker container.

    Will also ensure that the docker container is stopped and deleted if an error occurs to avoid causing
    problems in repeated trials.
    """

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
    """Executes a series of experiments based on the provided configuration and experiment function.

    Each trial will generate a new synthetic dataset. If needed, the inner loop can be modified to preserve
    docker instances between trials (but not schemas).

    Args:
        name: The name of the experiment. This will be used to create a directory for storing results.
        config: A dictionary containing the configuration for the experiment.
            Expected keys include:
                - "schemas": A list or grid of schema configurations to test.
                - "workflows": A list or grid of workflows to execute.
                - "trials": The number of trials to run for each schema and workflow combination.
                - "dataset": A dictionary containing dataset-specific configurations, such as "attributes".
        experiment: A function that defines the experiment logic. It should accept the following parameters:
            - logger: A logging object for recording experiment progress and results.
            - schema: The schema configuration for the current experiment.
            - workflow: The workflow configuration for the current experiment.
            - dataset: The dataset configuration.
            - partitioner: The database partitioner object.
            - attribute_names: A list of attribute names in the dataset.
            - attribute_data: The data corresponding to the attributes.
            - workflow_params: Parameters for generating synthetic workloads.
    """

    schemas = expand_experiment_grid(config["schemas"])
    workflows = expand_experiment_grid(config["workflows"])
    n_experiments = len(schemas) * len(workflows) * config["trials"]

    experiment_dir = EXPERIMENT_PATH / f"{name} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    experiment_dir.mkdir()

    with open(experiment_dir / "config.json", "w") as config_file:
        json.dump(
            config
            | {
                "expanded_schemas": schemas,
                "expanded_workflows": workflows,
            },
            config_file,
            indent=4,
        )

    current_experiment = 0
    for trial_i in range(config["trials"]):
        for schema_i, schema in enumerate(schemas):
            db_logger = configure_logging(experiment_dir, f"trial{trial_i}_schema{schema_i}_db")
            workflow_params = Workload.create_synthetic_workload(config["dataset"]["attributes"])
            with docker_container():
                db_logger.info(
                    f"Configuring schema {schema_i + 1}/{len(schemas)} and trial {trial_i + 1}/{config['trials']}..."
                )
                workload = Workload.sample_synthetic_workload(2000000, *workflow_params)
                db_logger.info("Loaded workload")
                partitioner, attribute_names, attribute_data = setup_db(
                    db_logger,
                    schema,
                    config["dataset"],
                    workload,
                )
                for workflow_i, workflow in enumerate(workflows):
                    current_experiment += 1
                    db_logger.info(
                        f"Running workflow {workflow_i + 1}/{len(workflows)} (experiment {current_experiment}/{n_experiments})..."
                    )
                    trial_name = f"trial{trial_i}_schema{schema_i}_workflow{workflow_i}"
                    logger = configure_logging(experiment_dir, trial_name)
                    results = experiment(
                        logger,
                        schema,
                        workflow,
                        config["dataset"],
                        partitioner,
                        attribute_names,
                        attribute_data,
                        workflow_params,
                    )
                    with open(experiment_dir / f"{trial_name}.json", "w") as results_file:
                        json.dump(results, results_file, indent=4)


def load_dataset(config: dict, base: bool) -> Dataset:
    """Load the sift dataset based on the configuration provided.

    Args:
        config: The configuration dictionary containing the dataset name and size.
        base: Whether to load the base dataset or not.

    Returns:
        The loaded dataset.
    """
    if config["name"] == "sift_1b":
        return load_sift_1b(DATASET_PATH / "sift1b", config["size"], base=base)
    elif config["name"] == "sift":
        return load_sift_1m(DATASET_PATH / "sift", base=base)
    elif config["name"] == "siftsmall":
        return load_sift_small(DATASET_PATH / "siftsmall", base=base)


def run_docker_command(command: str, ignore_errors: bool = False) -> None:
    """
    Helper function to run command with the Milvus docker image.

    Used to programatically create and destroy the docker container to ensure that each test is run on a fresh
    instance.

    Args:
        command: The command to run. Can be one of "start", "stop", "delete".
        ignore_errors: Whether to ignore errors or not, e.g. when deleting a non-existant container.
    """

    process = subprocess.run(
        ["sudo", "-S", BIN_PATH / "standalone_embed.sh", command],
        input=os.getenv("PASSWORD") + "\n",
        capture_output=True,
        text=True,
    )

    if process.returncode != 0 and not ignore_errors:
        raise RuntimeError(f"Failed to run command: {process.stdout}")


def configure_logging(experiment_path: Path, name: str) -> logging.Logger:
    """Configure the logger for the experiment.

    Args:
        experiment_path: Path to write to.
        name: Name of the logger.

    Returns:
        The constructed logger instance.
    """

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


def setup_db(
    logger: logging.Logger,
    schema_config: dict,
    dataset_config: dict,
    workload=None,
) -> tuple[MultiRangePartitioner, list[str], npt.NDArray[np.int32]]:
    """Construct a database with the given schema and synthetic dataset.

    Args:
        logger: Logger instance to log output information.
        schema_config: Configuration dictionary for the schema, including partitioner type, number of partitions, and index type.
        dataset_config: Configuration dictionary for the dataset, including attributes, maximum attribute value, and number of attributes.
        workload: Optional workload information used for tree-based partitioning. Defaults to None.

    Returns:
        A tuple containing:
        - The partitioner instance used to partition the data.
        - A list of attribute names from the dataset configuration.
        - A NumPy array of synthetic attribute data.
    """

    dataset = load_dataset(dataset_config, base=True)
    attributes = (
        np.random.default_rng()
        .uniform(0, dataset_config["max_attribute"], (dataset.num_base_vecs, dataset_config["n_attributes"]))
        .astype(int)
    )
    logger.info(f"Loaded dataset: {dataset}")

    if schema_config["partitioner"] == "range":
        partitions = {}
        for i in np.linspace(
            0, dataset_config["max_attribute"], schema_config["n_partitions"], endpoint=False
        ):
            range_ = Range(
                round(i),
                round(i + dataset_config["max_attribute"] // schema_config["n_partitions"] - 1),
            )
            partitions[f"{range_[0]}_{range_[1]}"] = {dataset_config["attributes"][0]: range_}
        partitioner = MultiRangePartitioner.from_partitions(partitions)
        logger.info(f"Using range partitioner with partitions: {partitions}")
    elif schema_config["partitioner"] == "tree":
        predicates = counter_characterize_workload(workload)
        tree = build_tree(
            attributes,
            dataset_config["attributes"],
            predicates,
            TreeAlgoParams(max_num_partitions=schema_config["n_partitions"]),
        )
        partitioner = MultiRangePartitioner.from_tree(tree)
        logger.info(f"Using tree partitioner with {schema_config['n_partitions']} partitions.")

    creator = Creator(
        partitioner,
        attributes=dataset_config["attributes"],
        datatype=dataset.datatype,
        logger=logger,
    )
    creator.create_collection_schema()
    creator.populate_collection(
        dataset,
        attributes,
        index_type=schema_config["index"],
        flush=schema_config.get("flush", True),
    )

    return partitioner, dataset_config["attributes"], attributes
