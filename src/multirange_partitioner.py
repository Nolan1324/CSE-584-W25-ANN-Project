from typing import Dict, Generator

from pymilvus import MilvusClient

from predicates import Operator, Predicate, Range, Atomic
from tree_partition_algo import Node, get_partitions_from_tree, get_example_tree
from tvl import Maybe


class MultiRangePartitioner:
    def __init__(self, tree: Node = None, partitions: Dict[str, Range] = None):
        self.tree = tree
        if not partitions and tree:
            self.partitions = get_partitions_from_tree(tree)
        else:
            self.partitions = partitions

    @classmethod
    def from_partitions(cls, partitions: Dict[str, Range]):
        return cls(partitions=partitions)

    @classmethod
    def from_tree(cls, tree: Node):
        return cls(tree=tree)

    def get_partition(self, vals: Dict[str, int]) -> str:
        if self.tree:
            return self.tree.find_partition(vals)
        else:
            for partition_name, ranges in self.partitions.items():
                for attr_name, val in vals.items():
                    if attr_name not in ranges:
                        continue
                    start, end = ranges[attr_name]
                    if (start is not None and val < start) or (
                        end is not None and val > end
                    ):
                        break
                else:
                    return partition_name

    def get_query_partitions(self, predicate: Predicate) -> Generator[str, None, None]:
        for name, ranges in self.partitions.items():
            if predicate.range_may_satisfy(ranges) in [Maybe, True]:
                yield name

    def add_partitions_to_collection(self, client: MilvusClient, collection_name: str):
        for name in self.partition_names:
            client.create_partition(
                collection_name=collection_name, partition_name=name
            )

    @property
    def partition_names(self):
        return self.partitions.keys()

    def __str__(self):
        return str(self.partitions)


if __name__ == "__main__":
    # node = get_example_tree()
    # partitioner = MultiRangePartitioner.from_tree(node)
    partitioner = MultiRangePartitioner.from_partitions(
        {
            "part_0": {"x": Range(start=500, end=None), "y": Range(start=300, end=None)},
            "part_1": {"x": Range(start=500, end=None), "y": Range(start=None, end=299)},
            "part_2": {"x": Range(start=100, end=499), "y": Range(start=300, end=None)},
            "part_3": {"x": Range(start=100, end=499), "y": Range(start=None, end=299)},
            "part_4": {"x": Range(start=None, end=99), "y": Range(start=300, end=None)},
            "part_5": {"x": Range(start=None, end=99), "y": Range(start=None, end=299)},
        }
    )
    # print(node)
    print(partitioner.partitions)
    print(partitioner.get_partition({"x": 100, "y": 200}))
    print(list(partitioner.get_query_partitions(Atomic("x", Operator.GTE, 100))))
