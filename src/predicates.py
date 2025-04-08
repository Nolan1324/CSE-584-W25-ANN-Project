from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple


class Operator(Enum):
    # EQ = "=="
    # NEQ = "!="
    # GT = ">"
    # LT = "<"
    GTE = ">="
    LTE = "<="


Range = Tuple[Optional[int], Optional[int]]


class Predicate(ABC):
    @abstractmethod
    def evaluate(self, vals: Dict[str, int]) -> bool:
        pass

    @abstractmethod
    def range_may_satisfy(self, ranges: Dict[str, Range]) -> bool:
        pass

    @abstractmethod
    def atomics(self) -> Generator[Atomic, None, None]:
        pass


@dataclass
class Atomic(Predicate):
    attr: str
    op: Operator
    value: int

    def evaluate(self, vals: Dict[str, int]) -> bool:
        x = vals[self.attr]
        match self.op:
            case Operator.GTE:
                return x >= self.value
            case Operator.LTE:
                return x <= self.value

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> bool:
        start, end = ranges[self.attr]
        match self.op:
            case Operator.GTE:
                return end is None or end >= self.value
            case Operator.LTE:
                return start is None or start <= self.value
            case _:
                assert False

    def atomics(self) -> Generator[Atomic, None, None]:
        yield self


class And(Predicate):
    def __init__(self, *predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return all(predicate.evaluate(vals) for predicate in self.predicates)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> bool:
        return all(predicate.range_may_satisfy(ranges) for predicate in self.predicates)

    def atomics(self) -> Generator[Atomic, None, None]:
        for predicate in self.predicates:
            yield from predicate.atomics()


class Or(Predicate):
    def __init__(self, *predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return any(predicate.evaluate(vals) for predicate in self.predicates)

    def evaluate_range(self, ranges: Dict[str, Range]) -> bool:
        return any(predicate.range_may_satisfy(ranges) for predicate in self.predicates)

    def atomics(self) -> Generator[Atomic, None, None]:
        for predicate in self.predicates:
            yield from predicate.atomics()


if __name__ == "__main__":
    pred = And(Atomic("x", Operator.GTE, 100), Atomic("x", Operator.LTE, 1000))
    print(pred.evaluate({"x": 500}))
    print(pred.evaluate({"x": 50}))
    print(pred.range_may_satisfy({"x": (500, 1000)}))
    print(pred.range_may_satisfy({"x": (50, 200)}))
    print(pred.range_may_satisfy({"x": (0, 10)}))
    print(list(pred.atomics()))
