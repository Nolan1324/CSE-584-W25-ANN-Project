
from collections import Counter
from typing import Iterable, List, Tuple
from predicates import Not, Predicate, And, Atomic, Operator


def counter_characterize_workload(workload: List[Predicate]) -> List[Tuple[Predicate, int]]:
    atomics = (atomic for pred in workload for atomic in pred.atomics())
    return list(sorted(Counter(atomics).items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    workload = [
        And(Atomic("x", Operator.GTE, 5), Not(Atomic("y", Operator.GTE, 10))),
        Atomic("x", Operator.GTE, 5)
    ]
    print(counter_characterize_workload(workload))