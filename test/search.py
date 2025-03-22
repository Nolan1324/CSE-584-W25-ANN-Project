from typing import Optional
import numpy as np
from pymilvus import MilvusClient
import time

from sift import SiftDataset
from client import get_client
from partitioner import Partitioner

class Searcher():
    def __init__(self, partitioner: Optional[Partitioner] =None):
        self.client = get_client()
        self.partitioner = partitioner
        self.collection_name = 'siftsmall'
        self.dataset = SiftDataset(self.collection_name, self.collection_name, with_base=False)

    def do_search(self):
        search_vector_id = 0

        if self.partitioner:
            partitions = list(self.partitioner.query_partitions(low=101, high=1000))
            print(partitions)
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
        print(self.dataset.ground_truth[search_vector_id])
        print([x['id'] for x in res[0]])

if __name__ == '__main__':
    # searcher = Searcher()
    searcher = Searcher(Partitioner([(0, 100), (101, 1000)]))
    searcher.do_search()