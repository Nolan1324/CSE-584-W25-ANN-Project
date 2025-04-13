from typing import List, Optional
import numpy as np
from dataclasses import dataclass

from multirange_partitioner import MultiRangePartitioner
from predicates import Atomic, Not, Operator, Predicate, Range
from sift import Dataset, load_sift_1m
from client import get_client
from partitioner import Partitioner, RangePartitioner
from attributes import uniform_attributes_basic
from utils import Timer


class Searcher:
    @dataclass
    class SearchResults:
        ground_truth: List[int]
        results: List[int]
        time: float

    def __init__(
        self,
        collection_name: str,
        attribute_names: List[str],
        attribute_data: "np.ndarray[np.int32]",
        dataset: Dataset,
        partitioner: Optional[MultiRangePartitioner] = None,
    ):
        self.client = get_client()
        self.partitioner = partitioner
        self.collection_name = collection_name
        self.dataset = dataset
        self.attribute_names = attribute_names
        self.attribute_data = attribute_data

    def do_search(
        self, search_vector_id, predicate: Predicate = None, limit: int = 100
    ):
        def _get_attributes_dict(row_index):
            return {
                name: value
                for name, value in zip(self.attribute_names, self.attribute_data[row_index])
            }

        if self.partitioner and predicate:
            partitions = list(self.partitioner.get_query_partitions(predicate))
        else:
            partitions = None

        with Timer() as timer:
            if predicate is None:
                res = self.client.search(
                    collection_name=self.collection_name,
                    data=[self.dataset.query[search_vector_id, :]],
                    limit=limit,
                    output_fields=["id"],
                )
            else:
                res = self.client.search(
                    collection_name=self.collection_name,
                    data=[self.dataset.query[search_vector_id, :]],
                    limit=limit,
                    output_fields=["id"],
                    filter=str(predicate),
                    partition_names=partitions,
                )

        if predicate is None:
            return self.SearchResults(
                ground_truth=[i for i in self.dataset.ground_truth[search_vector_id]],
                results=[x["id"] for x in res[0]],
                time=timer.duration,
            )
        return self.SearchResults(
            ground_truth=[
                i
                for i in self.dataset.ground_truth[search_vector_id]
                if predicate.evaluate(_get_attributes_dict(i))
            ],
            results=[x["id"] for x in res[0]],
            time=timer.duration,
        )


if __name__ == "__main__":
    # partitioner = RangePartitioner([(0, 100), (101, 1000)])
    # partitioner = None
    partitioner = MultiRangePartitioner.from_partitions(
        {"0_100": {"x": Range(0, 100)}, "101_1000": {"x": Range(101, 1000)}}
    )
    # searcher = Searcher()
    dataset = load_sift_1m("../data/datasets/sift", base=False)
    searcher = Searcher("data", ['x'], uniform_attributes_basic(1000000), dataset, partitioner)
    results = searcher.do_search(0, Not(Atomic('x', Operator.GTE, 51)))
    print(results)
