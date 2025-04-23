
from collections import Counter
from typing import Iterable, List, Tuple
from predicates import Not, Predicate, And, Atomic, Operator
from workload import Workload


def counter_characterize_workload(workload: List[Predicate]) -> List[Tuple[Predicate, int]]:
    """Characterize a workload by counting the frequencies of atomic predicates.

    Args:
        workload (List[Predicate]): List of filter predicates in the workload.

    Returns:
        List[Tuple[Predicate, int]]: List of (atomic predicate, frequency) pairs, 
            sorted in decreasing order by frequency
    """

    atomics = (atomic for pred in workload for atomic in pred.atomics())
    return list(sorted(Counter(atomics).items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    workload = [
        And(Atomic("x", Operator.GTE, 5), Not(Atomic("y", Operator.GTE, 10))),
        Atomic("x", Operator.GTE, 5)
    ]
    workload = Workload.create_and_sample_synthetic_workload(100, ["w", "x", "y", "z"])
    print(workload)
    print(counter_characterize_workload(workload))
