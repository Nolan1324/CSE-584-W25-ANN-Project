"""Run the local experiment used in our report and presentation."""

import logging
from typing import Iterable

import numpy as np
from tqdm import tqdm

from experiment_utils import run_experiment, load_dataset
from search import Searcher
from workload import Workload


def confusion(ground_truth: Iterable[int], results: Iterable[int]) -> tuple[int, int, int]:
    """
    Computes the confusion metrics: true positives (TP), false positives (FP),
    and false negatives (FN) based on the ground truth and results.

    Args:
        ground_truth (Iterable[int]): An iterable containing the ground truth labels.
        results (Iterable[int]): An iterable containing the predicted results.

    Returns:
        tuple: A tuple containing three integers:
            - tp (int): The number of true positives (elements in both ground_truth and results).
            - fp (int): The number of false positives (elements in results but not in ground_truth).
            - fn (int): The number of false negatives (elements in ground_truth but not in results).
    """

    ground_truth = set(ground_truth)
    results = set(results)

    tp = len(ground_truth & results)
    fp = len(results - ground_truth)
    fn = len(ground_truth - results)

    return tp, fp, fn


def test(
    logger: logging.Logger,
    schema_config: dict,
    workflow_config: dict,
    dataset_config: dict,
    partitioner,
    attribute_names: list[str],
    attribute_data,
    workload_params: dict,
) -> dict:
    """
    Executes a search experiment using the provided configurations and logs the results. Relies on the

    Args:
        logger: Logger instance for logging information and progress.
        schema_config: Configuration dictionary for the schema.
        workflow_config: Configuration dictionary for the workflow.
        dataset_config: Configuration dictionary for the dataset, including parameters like "k".
        partitioner: Object responsible for partitioning the dataset.
        attribute_names: Names of the attributes used in the search.
        attribute_data: Data corresponding to the attributes used in the search.
        workload_params: Parameters for generating the synthetic workload.

    Returns:
        results to save to a results json file
    """

    logger.info("Running search...")
    dataset = load_dataset(dataset_config, base=False)
    workload = Workload.sample_synthetic_workload(dataset.num_query_vecs, *workload_params)
    logger.info(f"Loaded query dataset: {dataset}")
    searcher = Searcher("data", attribute_names, attribute_data, dataset, partitioner)
    times = []
    filter_used_list = []
    tp_list = []
    fp_list = []
    fn_list = []

    query_indices = np.arange(searcher.dataset.query.shape[0])
    np.random.shuffle(query_indices)
    for index in tqdm(query_indices):
        search_results = searcher.do_search(index, workload[index], limit=dataset_config["k"])
        tp, fp, fn = confusion(search_results.ground_truth, search_results.results)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        times.append(search_results.time)
    logger.info("Search complete.")

    return {
        "search_times": times,
        "filter_used": filter_used_list,
        "tp": tp_list,
        "fp": fp_list,
        "fn": fn_list,
    }


if __name__ == "__main__":
    name = "second_partition_trial"
    function = test

    experiment_grid = {
        "kill_on_fail": False,
        "trials": 3,
        "schemas": {
            "index": ["IVF_FLAT"],
            "partitioner": ["tree", "range"],
            "n_partitions": [10, 100, 500, 1000],
        },
        "workflows": {
            "synthetic": [True],
        },
        "dataset": {
            "name": "sift_1b",
            "size": 10,
            "max_attribute": 1_000,
            "attributes": ["w", "x", "y", "z"],
            "n_attributes": 4,
            "k": 10_000,
        },
    }

    run_experiment(name, experiment_grid, function)
