from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from pymilvus import MilvusClient

class Partitioner(ABC):
    def add_partitions_to_collection(self, client: MilvusClient, collection_name: str):
        for name in self.partition_names:
            client.create_partition(
                collection_name=collection_name,
                partition_name=name
            )
    
    @abstractmethod
    def get_partition(self, value: int):
        pass

    @abstractmethod
    def get_query_partitions(self, low: Optional[int] = None, high: Optional[int] = None):
        pass

    @property
    @abstractmethod
    def partition_names(self):
        pass


class RangePartitioner(Partitioner):
    def __init__(self, ranges: List[Tuple[int, int]]):
        self._partition_names = []
        self.partition_ranges = []

        for range_ in ranges:
            self._partition_names.append(f'range_{range_[0]}_{range_[1]}')
            self.partition_ranges.append(range_)

    def get_partition(self, value: int):
        for name, range_ in zip(self._partition_names, self.partition_ranges):
            if range_[0] <= value <= range_[1]:
                return name
        return None

    def get_query_partitions(self, low: Optional[int] = None, high: Optional[int] = None):
        for name, range_ in zip(self._partition_names, self.partition_ranges):
            r_low, r_high = range_
            if (r_low if low is None else max(r_low, low)) <= (r_high if high is None else min(r_high, high)):
                yield name

    @property
    def partition_names(self):
        return self._partition_names
    

class ModPartitioner(Partitioner):
    def __init__(self, num_partitions: int):
        self._partition_names = []
        self.num_partitions = num_partitions

        for i in range(num_partitions):
            self._partition_names.append(f'_remainder_{i}')

    def get_partition(self, value: int):
        return self.partition_names[value % self.num_partitions]

    def get_query_partitions(self, low: Optional[int] = None, high: Optional[int] = None):
        if low is None or high is None:
            return self.partition_names
        else:
            res = []
            for i in range(low, high+1):
                res.append(self.partition_names[i % self.num_partitions])
                if len(res) == self.num_partitions:
                    break
            return res
        
    @property
    def partition_names(self):
        return self._partition_names

if __name__ == '__main__':
    partitioner = RangePartitioner([(0, 100), (101, 1000)])
    print(list(partitioner.get_query_partitions(low=None, high=10)))
    print(list(partitioner.get_query_partitions(low=0, high=10)))
    print(list(partitioner.get_query_partitions(low=0, high=100)))
    print(list(partitioner.get_query_partitions(low=500, high=None)))
    print(list(partitioner.get_query_partitions(low=10, high=200)))
    print(partitioner.partition_names)
    print()
    partitioner = ModPartitioner(10)
    print(list(partitioner.get_query_partitions(low=25, high=28)))
