from typing import List, Optional, Tuple

from pymilvus import MilvusClient

class Partitioner():
    def __init__(self, ranges: List[Tuple[int, int]]):
        self.partition_names = []
        self.partition_ranges = []

        for range_ in ranges:
            self.partition_names.append(f'range_{range_[0]}_{range_[1]}')
            self.partition_ranges.append(range_)

    def add_partitions_to_collection(self, client: MilvusClient, collection_name: str):
        for name in self.partition_names:
            client.create_partition(
                collection_name=collection_name,
                partition_name=name
            )

    def query_partition(self, value: int):
        for name, range_ in zip(self.partition_names, self.partition_ranges):
            if range_[0] <= value <= range_[1]:
                return name
        return None

    def query_partitions(self, low: Optional[int] = None, high: Optional[int] = None):
        for name, range_ in zip(self.partition_names, self.partition_ranges):
            if (low is None or low <= range_[0]) and (high is None or high >= range_[1]):
                yield name

if __name__ == '__main__':
    partitioner = Partitioner([(0, 10), (11, 1000)])
    print(list(partitioner.query_partitions(low=None, high=10)))