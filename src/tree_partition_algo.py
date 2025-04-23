from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from attributes import uniform_attributes
from predicates import And, Atomic, Not, Operator, Range

import numpy.typing as npt

@dataclass
class TreeAlgoParams():
    """Parameters for the tree partitioning algorithm.

    Attributes:
        min_predicate_frequency (int): The minimum number of times an atomic predicate must appear 
            in the workload to be considered.

        selectivity_threshold (float): The minimum selectivity on the current split of data an atomic 
            predicate must have to be considered. This applies to both the true and false branches; 
            i.e., both branches must include at least (selectivity_threshold * 100)% of the data 
            in the current split.

        min_partition_size (int): The minimum number of vectors that can be in a partition leaf node.

        max_num_partitions (int): The maximum number of partitions that the algorithm can create.
    """

    min_predicate_frequency: int = 10
    selectivity_threshold: float = 0.11
    min_partition_size: int = 1_000
    max_num_partitions: int = 4096


@dataclass
class PartitionTree():
    """
    A node of a partition tree.

    If this is an internal node, `predicate`, `if_true`, and `if_false` will be non-None and `partition_name` will be None.
    If this is a root node, `predicate`, `if_true`, and `if_false` will be None and `partition_name` will be non-None.

    Attributes:
        predicate (Atomic | None): If this is an internal node, the atomic predicate that this node splits on. `None` otherwise.
        if_true (PartitionTree | None): Sub-tree to visit if the predicate evaluates to True.
        if_false (PartitionTree | None): Sub-tree to visit if the predicate evaluates to False.
        partition_name (str | None): If this is a leaf node, the name of the partition represented by this node. `None` otherwise.
    """

    predicate: Atomic = None
    if_true: PartitionTree = None
    if_false: PartitionTree = None
    partition_name: str = None

    def assert_well_formed(self):
        assert(bool(self.predicate and self.if_true and self.if_false) != bool(self.partition_name))

    def find_partition(self, vals: Dict[str, int]) -> str:
        """Given a set of scalar attributes of a vector, recursively search for the partition containing that vector.

        Args:
            vals (Dict[str, int]): Mapping from scalar attribute name to value. Must contain all scalar attributes.
        
        Returns:
            str: Name of the partition containing the vector.
        """

        self.assert_well_formed()
        if self.partition_name:
            return self.partition_name
        else:
            if self.predicate.evaluate(vals):
                return self.if_true.find_partition(vals)
            else:
                return self.if_false.find_partition(vals)
            
    def __str__(self):
        self.assert_well_formed()
        if self.partition_name:
            return self.partition_name
        else:
            out = str(self.predicate) + '\n'
            for line in str(self.if_true).split('\n'):
                out += '  ' + line + '\n'
            out += str(Not(self.predicate)) + '\n'
            for line in str(self.if_false).split('\n'):
                out += '  ' + line + '\n'
            out = out[:-1]
            return out


def build_tree(data: npt.NDArray, attr_names: List[str], atomics: List[Tuple[Atomic, int]], params: TreeAlgoParams) -> PartitionTree:
    """Build a partition tree from a dataset and set of atomic predicates.

    Args:
        data (npt.NDArray): Data array with shape (n,d) and dtype np.int32. Each row should represent a vector, and each column should represent a scalar attribute.
        attr_names (List[str]): Names of the scalar attributes in `data`. `attr_names[i]` should be the name of attribute column `data[:,i]`.
        atomics (List[Tuple[Atomic, int]]): List of atomic predicates, along with how frequently each occured.
        params (TreeAlgoParams): Parameters for the algorithm. Pass TreeAlgoParams() for default parameters.

    Returns:
        PartitionTree: The root node of the partition tree.
    """

    return _build_tree_helper(data, attr_names, list(sorted(atomics, key=lambda x: x[1], reverse=True)), params)


def _build_tree_helper(data: npt.NDArray, attr_names: List[str], atomics: List[Tuple[Atomic, int]], params: TreeAlgoParams, total_leaves=[1], next_partition_index=[0]) -> PartitionTree:
    assert(len(data.shape) == 2)
    assert(data.shape[1] == len(attr_names))
    
    if total_leaves[0] + 1 <= params.max_num_partitions:
        for i, (atomic_pred, freq) in enumerate(atomics):
            if freq < params.min_predicate_frequency:
                continue

            attr_index = attr_names.index(atomic_pred.attr)
            col = data[:,attr_index]
            assert(atomic_pred.op == Operator.GTE)
            bit_mask = col >= atomic_pred.value
            selectivity = np.sum(bit_mask) / data.shape[0]
            if selectivity < params.selectivity_threshold or selectivity > (1 - params.selectivity_threshold):
                continue
            if np.sum(bit_mask) < params.min_partition_size or np.sum(~bit_mask) < params.min_partition_size:
                continue
            
            total_leaves[0] += 1
            if_true = build_tree(data[bit_mask], attr_names, atomics[i+1:], params)
            if_false = build_tree(data[~bit_mask], attr_names, atomics[i+1:], params)

            return PartitionTree(if_true=if_true, if_false=if_false, predicate=atomic_pred) 
    
    partition_name = f"part_{next_partition_index[0]}"
    next_partition_index[0] += 1
    return PartitionTree(partition_name=partition_name)

def get_partitions_from_tree(node: PartitionTree) -> Dict[str, Dict[str, Range]]:
    """Given a partition tree, get each partition summarized as a set of scalar ranges on each attribute.

    Args:
        node (PartitionTree): Root node of the partition tree.

    Returns:
        Dict[str, Dict[str, Range]]: The resulting partitions. An entry of the form `partitions[partition][x] = Range(a, b)` 
        indicates that partition `partition` only includes valuse where attribute `x` is in the range `[a, b]`.
    """

    return _get_partitions_from_tree_helper(node, defaultdict(lambda: (None, None)))

def _get_partitions_from_tree_helper(node: PartitionTree, cur_ranges: Dict[str, Range]) -> Dict[str, Dict[str, Range]]:
    node.assert_well_formed()
    if node.partition_name is None:
        assert(node.predicate.op == Operator.GTE)
        value = node.predicate.value
        start, end = cur_ranges[node.predicate.attr]

        true_ranges = cur_ranges.copy()
        true_ranges[node.predicate.attr] = Range(max(start or value, value), end)
        
        false_ranges = cur_ranges.copy()
        false_ranges[node.predicate.attr] = Range(start, min(end or (value - 1), value - 1))

        true_partitions = get_partitions_from_tree(node.if_true, true_ranges)
        false_partitions = get_partitions_from_tree(node.if_false, false_ranges)

        return true_partitions | false_partitions
    else:
        return {node.partition_name: dict(cur_ranges)}

def get_example_tree() -> PartitionTree:
    atomics = [(Atomic("x", Operator.GTE, 500), 10), (Atomic("x", Operator.GTE, 100), 9), (Atomic("y", Operator.GTE, 300), 8)]

    n = 100000
    x = uniform_attributes(n, 584, np.int32, 0, 1000)
    y = uniform_attributes(n, 585, np.int32, 0, 1000)

    data = np.column_stack((x, y))

    tree = build_tree(data, ['x', 'y'], atomics, TreeAlgoParams(max_num_partitions=5))

    return tree

if __name__ == '__main__':
    # workload = [
    #     And(Atomic("x", Operator.GTE, 500), Not(Atomic("y", Operator.GTE, 100))),
    #     Atomic("x", Operator.GTE, 500)
    # ]
    # atomics = counter_characterize_workload(workload)
    # print(atomics)

    TreeAlgoParams()

    tree = get_example_tree()
    print(tree)

    parts = get_partitions_from_tree(tree)
    print(parts)

    print(tree.find_partition({'x': 100, 'y': 200}))
    