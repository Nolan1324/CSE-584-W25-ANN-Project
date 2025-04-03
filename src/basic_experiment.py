import logging
import random
import json
import time
from pathlib import Path
from itertools import product
from typing import Iterable, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiment_utils import run_test, setup_db, load_dataset
from sift import load_sift_1b
from utils import Timer
from search import Searcher


def confusion(ground_truth: Iterable[int], results: Iterable[int]):
    ground_truth = set(ground_truth)
    results = set(results)

    tp = len(ground_truth & results)
    fp = len(results - ground_truth)
    fn = len(ground_truth - results)

    return tp, fp, fn

def test(logger: logging.Logger, config: dict, experiment_dir: Path) -> dict:
    partitioner, attributes = setup_db(logger, config)
    
    logger.info("Waiting...")
    time.sleep(5)
    logger.info("Running search...")
    dataset = load_dataset(config, base=False)
    logger.info(f"Loaded query dataset: {dataset}")
    searcher = Searcher(config["dataset"], attributes, dataset, partitioner)
    times = []
    filter_used_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    
    query_indices = np.arange(searcher.dataset.query.shape[0])
    np.random.shuffle(query_indices)
    for index in tqdm(query_indices):
        with Timer() as timer:
            filter_used = random.random() <= config["filter_percentage"]
            filter_used_list.append(filter_used)
            if filter_used:
                search_results = searcher.do_search(index, upper_bound= int(config["key_max"] * config["selectivity"]))
            else:
                search_results = searcher.do_search(index)
            tp, fp, fn = confusion(search_results.ground_truth, search_results.results)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
        times.append(search_results.time)
    logger.info("Search complete.")
    
    stats = {
        "search_times": times,
        "filter_used": filter_used_list,
        "tp": tp_list,
        "fp": fp_list,
        "fn": fn_list,
        "average_search_time": np.mean(times),
        "median_search_time": np.median(times),
        "max_search_time": np.max(times),
        "min_search_time": np.min(times),
        "std_dev_search_time": np.std(times),
    }
    plt.scatter(np.arange(len(times)), times)
    plt.savefig(experiment_dir / "plot.png", dpi=300)
    
    return stats


if __name__ == "__main__":
    kill_on_fail = True
    trials = 3
    experiment_grid = {
        "vector_index": ["HNSW"],
        "n_partitions": [1],
        "dataset": ["sift_1b"],
        "dataset_size": [2],
        "test_function": [test],
        "selectivity": [1],
        "filter_percentage": [0],
        "key_max": [1_000],
        "partitioner": ["mod", "range"],
        "trial": list(range(trials)),
        "name": ["new_test"],
    }
    experiment_configs = [
        dict(zip(experiment_grid.keys(), values))
        for values in product(*experiment_grid.values())
    ]
    experiment_configs = experiment_configs[:1]
    
    for i, config in enumerate(experiment_configs):
        config["seed"] = random.randint(0, 1_000_000)
        test = config["test_function"]
        config["test_function"] = test.__name__
        print(f"Running experiment {i+1}/{len(experiment_configs)} with config: {json.dumps(config, indent=4)}")
        config["test_function"] = test
        success = run_test(config)
        if kill_on_fail and not success:
            print(f"Experiment {i+1} failed. Killed future experiments.")
    print("All experiments completed.")
