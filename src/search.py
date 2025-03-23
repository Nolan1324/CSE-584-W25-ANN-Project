from pathlib import Path
from typing import Optional
import numpy as np
from pymilvus import MilvusClient
import time

from sift import SiftDataset
from client import get_client
from partitioner import Partitioner
from attributes import uniform_attributes_example

class Searcher():
    def __init__(self, collection_name: str, attributes: 'np.ndarray[np.int32]', partitioner: Optional[Partitioner] = None):
        self.client = get_client()
        self.partitioner = partitioner
        self.collection_name = collection_name
        self.dataset = SiftDataset('../data' / Path(self.collection_name), self.collection_name, with_base=False)
        self.attributes = attributes

    def do_search(self):
        search_vector_id = 0

        if self.partitioner:
            partitions = list(self.partitioner.get_query_partitions(low=None, high=100))
        else:
            partitions = None

        start_time = time.time()
        res = self.client.search(
            collection_name=self.collection_name,
            data=[self.dataset.query[search_vector_id,:]],
            limit=100,
            output_fields=["id"],
            filter="attribute <= 100",
            partition_names=partitions
        )
        end_time = time.time()

        print(f'Search time: {end_time - start_time}')
        print([i for i in self.dataset.ground_truth[search_vector_id] if self.attributes[i] <= 100])
        print([x['id'] for x in res[0]])

if __name__ == '__main__':
    partitioner = Partitioner([(0, 100), (101, 1000)])
    # searcher = Searcher()
    searcher = Searcher('sift', uniform_attributes_example(1000000), partitioner)
    searcher.do_search()