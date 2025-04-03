from typing import List, Optional
import numpy as np
from dataclasses import dataclass
import time

from sift import Dataset, load_sift_1m
from client import get_client
from partitioner import Partitioner, RangePartitioner
from attributes import uniform_attributes_example


class Searcher():
    @dataclass
    class SearchResults():
        ground_truth: List[int]
        results: List[int]
        time: float

    def __init__(self, collection_name: str, attributes: 'np.ndarray[np.int32]', dataset: Dataset, partitioner: Optional[Partitioner] = None):
        self.client = get_client()
        self.partitioner = partitioner
        self.collection_name = collection_name
        self.dataset = dataset
        self.attributes = attributes

    def do_search(self, search_vector_id, upper_bound: int = None, limit: int = 100):
        if self.partitioner:
            partitions = list(self.partitioner.get_query_partitions(low=None, high=upper_bound))
        else:
            partitions = None

        start_time = time.monotonic()
        if upper_bound is None:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[self.dataset.query[search_vector_id,:]],
                limit=limit,
                output_fields=["id"],
            )
        else:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[self.dataset.query[search_vector_id,:]],
                limit=limit,
                output_fields=["id"],
                filter=f"attribute <= {upper_bound}",
                partition_names=partitions
            )
        end_time = time.monotonic()
        
        if upper_bound is None:
            return self.SearchResults(
                ground_truth=[i for i in self.dataset.ground_truth[search_vector_id]],
                results=[x['id'] for x in res[0]],
                time=end_time-start_time
            )
        return self.SearchResults(
            ground_truth=[i for i in self.dataset.ground_truth[search_vector_id] if self.attributes[i] <= upper_bound],
            results=[x['id'] for x in res[0]],
            time=end_time-start_time
        )

if __name__ == '__main__':
    partitioner = RangePartitioner([(0, 100), (101, 1000)])
    # searcher = Searcher()
    dataset = load_sift_1m("data/datasets/sift1m", base=False)
    searcher = Searcher('sift', uniform_attributes_example(1000000), partitioner)
    searcher.do_search(0, upper_bound=100)
