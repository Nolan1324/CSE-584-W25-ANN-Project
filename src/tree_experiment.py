import logging
import random
from typing import Iterable

import numpy as np
from tqdm import tqdm

from experiment_utils import run_experiment, load_dataset
from predicates import Atomic, Not, Operator
from search import Searcher
from workload import Workload


def confusion(ground_truth: Iterable[int], results: Iterable[int]):
    ground_truth = set(ground_truth)
    results = set(results)

    tp = len(ground_truth & results)
    fp = len(results - ground_truth)
    fn = len(ground_truth - results)

    return tp, fp, fn


def test(logger: logging.Logger, schema_config: dict, workflow_config: dict, dataset_config: dict, partitioner, attribute_names, attribute_data, workload_params) -> dict:
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
    name = "new_partition_baseline"
    function = test
    
    experiment_grid = {
        "kill_on_fail": False,
        "trials": 1,
        "schemas": {
            "index": ["IVF_FLAT"],
            "partitioner": ["tree"],
            "n_partitions": [10, 100, 1000],
        },
        "workflows": {
            "synthetic": [True],
        },
        "dataset": {
            "name": "sift_1b",
            "size": 5,
            "max_attribute": 1_000,
            "attributes": ["w", "x", "y", "z"],
            "n_attributes": 4,
            "k": 10_000,
        }
    }
    
    run_experiment(name, experiment_grid, function)
