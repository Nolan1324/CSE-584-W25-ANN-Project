from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import uuid

import numpy as np
from attributes import uniform_attributes
from predicates import And, Atomic, Not, Operator, Range
from workload_char import counter_characterize_workload

import numpy.typing as npt

MIN_THRESHOLD = 1
SELECTIVITY_THRESHOLD = 0.01

@dataclass
class Node():
    predicate: Atomic = None
    if_true: Node = None
    if_false: Node = None
    partition_name: str = None

    def assert_well_formed(self):
        assert(bool(self.predicate and self.if_true and self.if_false) != bool(self.partition_name))

def build_tree(data: npt.NDArray, attr_names: List[str], atomics: List[Tuple[Atomic, int]], next_partition_index = [0]) -> Node:
    assert(len(data.shape) == 2)
    assert(data.shape[1] == len(attr_names))
    
    for i, (atomic_pred, freq) in enumerate(atomics):
        if freq < MIN_THRESHOLD:
            continue
        # bit_mask = np.zeros(data.shape[0], dtype=bool)
        # for i, row in enumerate(data):
        #     row_dict = dict(zip(attr_names, row))
        #     if atomic_pred.evaluate(row_dict):
        #         bit_mask[i] = True
        attr_index = attr_names.index(atomic_pred.attr)
        col = data[:,attr_index]
        assert(atomic_pred.op == Operator.GTE)
        bit_mask = col >= atomic_pred.value
        selectivity = np.sum(bit_mask) / data.shape[0]
        if selectivity < SELECTIVITY_THRESHOLD or selectivity > (1 - SELECTIVITY_THRESHOLD):
            continue
        
        if_true = build_tree(data[bit_mask], attr_names, atomics[i+1:])
        if_false = build_tree(data[~bit_mask], attr_names, atomics[i+1:])

        return Node(if_true=if_true, if_false=if_false, predicate=atomic_pred) 
    
    partition_name = f"part_{next_partition_index[0]}"
    next_partition_index[0] += 1
    return Node(partition_name=partition_name)

def get_partitions_from_tree(node: Node, cur_ranges: Dict[str, Range] = None) -> Dict[str, Dict[str, Range]]:
    if cur_ranges is None:
        cur_ranges = defaultdict(lambda: (None, None))

    node.assert_well_formed()
    if node.partition_name is None:
        assert(node.predicate.op == Operator.GTE)
        value = node.predicate.value
        start, end = cur_ranges[node.predicate.attr]

        true_ranges = cur_ranges.copy()
        true_ranges[node.predicate.attr] = (max(start or value, value), end)
        
        false_ranges = cur_ranges.copy()
        false_ranges[node.predicate.attr] = (start, min(end or (value - 1), value - 1))

        true_partitions = get_partitions_from_tree(node.if_true, true_ranges)
        false_partitions = get_partitions_from_tree(node.if_false, false_ranges)

        return true_partitions | false_partitions
    else:
        return {node.partition_name: dict(cur_ranges)}


if __name__ == '__main__':
    workload = [
        And(Atomic("x", Operator.GTE, 500), Not(Atomic("y", Operator.GTE, 100))),
        Atomic("x", Operator.GTE, 500)
    ]
    atomics = counter_characterize_workload(workload)
    print(atomics)

    n = 100000
    x = uniform_attributes(n, 584, np.int32, 0, 1000)
    y = uniform_attributes(n, 585, np.int32, 0, 1000)

    data = np.column_stack((x, y))

    tree = build_tree(data, ['x', 'y'], atomics)
    parts = get_partitions_from_tree(tree)
    print(parts)