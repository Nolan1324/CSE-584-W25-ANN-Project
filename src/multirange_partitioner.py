from typing import Dict, Generator

from predicates import Operator, Predicate, Range, Atomic
from tree_partition_algo import Node, get_partitions_from_tree, get_example_tree
from tvl import Maybe

class MultiRangePartitioner():
    def __init__(self, tree: Node):
        self.tree = tree
        self.partitions = get_partitions_from_tree(tree)

    def get_partition(self, vals: Dict[str, int]) -> str:
        if self.tree:
            return self.tree.find_partition(vals)
        else:
            for partition_name, ranges in self.partitions.items():
                for attr_name, val in vals.items():
                    if attr_name not in ranges:
                        continue
                    start, end = ranges[attr_name]
                    if (start is not None and val < start) or (end is not None and val > end):
                        break
                else:
                    return partition_name

    def get_query_partitions(self, predicate: Predicate) -> Generator[str, None, None]:
        for name, ranges in self.partitions.items():
            if predicate.range_may_satisfy(ranges) in [Maybe, True]:
                yield name

    @property
    def partition_names(self):
        return self.partitions.keys()
    
    def __str__(self):
        return str(self.partitions)
    
if __name__ == '__main__':
    node = get_example_tree()    
    partitioner = MultiRangePartitioner(node)
    print(node)
    print(partitioner.partitions)
    print(partitioner.get_partition({'x': 100, 'y': 200}))
    print(list(partitioner.get_query_partitions(Atomic('x', Operator.GTE, 100))))
