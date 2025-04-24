import logging
import random
from typing import Iterable

import numpy as np
from tqdm import tqdm

from experiment_utils import run_experiment, load_dataset
from predicates import Atomic, Not, Operator
from search import Searcher


def confusion(ground_truth: Iterable[int], results: Iterable[int]):
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
    attribute_names,
    attribute_data,
) -> dict:
    logger.info("Running search...")
    dataset = load_dataset(dataset_config, base=False)
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
        filter_used = random.random() <= workflow_config["filter_percentage"]
        filter_used_list.append(filter_used)
        if filter_used:
            search_results = searcher.do_search(
                index,
                Not(
                    Atomic(
                        "x",
                        Operator.GTE,
                        int(dataset_config["max_attribute"] * workflow_config["selectivity"]) + 1,
                    )
                ),
                limit=workflow_config["k"],
            )
        else:
            search_results = searcher.do_search(index, limit=workflow_config["k"])
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
        "average_search_time": float(np.mean(times)),
        "median_search_time": float(np.median(times)),
        "max_search_time": float(np.max(times)),
        "min_search_time": float(np.min(times)),
        "std_dev_search_time": float(np.std(times)),
    }


if __name__ == "__main__":
    name = "partition_baseline"
    function = test

    experiment_grid = {
        "kill_on_fail": False,
        "trials": 1,
        "schemas": {
            "index": ["IVF_FLAT"],
            "partitioner": ["range"],
            "n_partitions": [1, 2, 5, 10, 100, 1000],
        },
        "workflows": {
            "selectivity": [0.1, 0.5],
            "filter_percentage": [0.1, 0.5, 0.9],
            "k": 1_000,
        },
        "dataset": {
            "name": "sift_1b",
            "size": 10,
            "max_attribute": 1_000,
        },
    }

    run_experiment(name, experiment_grid, function)
