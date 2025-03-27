import logging
import random
import json
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiment_utils import run_test, setup_db
from utils import Timer
from search import Searcher


def test(logger: logging.Logger, config: dict, experiment_dir: Path) -> dict:
    partitioner, attributes = setup_db(logger, config)
    
    logger.info("Running search...")
    searcher = Searcher(config["dataset"], attributes, partitioner)
    times = []
    indices = np.arange(10_000)
    for index in tqdm(indices):
        if attributes[index] >= 10:
            continue
        with Timer() as timer:
            searcher.do_search(index, upper_bound=100)
        times.append(timer.duration)
    logger.info("Search complete.")
    
    stats = {
        "search_times": times,
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
    trials = 3
    experiment_grid = {
        "vector_index": ["HNSW", "FLAT", "IVF_FLAT"],
        "n_partitions": [5, 10, 20],
        "dataset": ["sift"],
        "name": ["basic_experiment"],
        "test_function": [test],
    }
    
    for config in (dict(zip(experiment_grid.keys(), values)) for values in product(*experiment_grid.values())):
        for t in range(trials):
            config["seed"] = random.randint(0, 1_000_000)
            config["trial"] = t
            test = config["test_function"]
            config["test_function"] = test.__name__
            print(f"Running experiment with config: {json.dumps(config, indent=4)}")
            config["test_function"] = test
            run_test(config)
