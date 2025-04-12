from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Literal, Optional, Tuple

from tvl import Maybe, TVL, tvl_not

class Operator(str, Enum):
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
    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        pass

    @abstractmethod
    def atomics(self) -> Generator[Atomic, None, None]:
        pass


@dataclass(frozen=True)
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

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        start, end = ranges[self.attr]
        match self.op:
            case Operator.GTE:
                if end is None or end >= self.value:
                    if start is not None and start >= self.value:
                        return True
                    else:
                        return Maybe
                else:
                    return False
            case Operator.LTE:
                if start is None or start <= self.value:
                    if end is not None and end <= self.value:
                        return True
                    else:
                        return Maybe
                else:
                    return False
            case _:
                assert False

    def atomics(self) -> Generator[Atomic, None, None]:
        yield self

    def __str__(self):
        return f"{self.attr} {str(self.op.value)} {self.value}"


class And(Predicate):
    def __init__(self, *predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return all(predicate.evaluate(vals) for predicate in self.predicates)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        val = True
        for predicate in self.predicates:
            val &= predicate.range_may_satisfy(ranges)
        return val

    def atomics(self) -> Generator[Atomic, None, None]:
        for predicate in self.predicates:
            yield from predicate.atomics()

    def __str__(self):
        return "(" + " and ".join(str(predicate) for predicate in self.predicates) + ")"


class Or(Predicate):
    def __init__(self, *predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return any(predicate.evaluate(vals) for predicate in self.predicates)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        val = False
        for predicate in self.predicates:
            val |= predicate.range_may_satisfy(ranges)
        return val

    def atomics(self) -> Generator[Atomic, None, None]:
        for predicate in self.predicates:
            yield from predicate.atomics()

    def __str__(self):
        return "(" + " or ".join(str(predicate) for predicate in self.predicates) + ")"


class Not(Predicate):
    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return not self.predicate.evaluate(vals)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        val = self.predicate.range_may_satisfy(ranges)
        return tvl_not(val)

    def atomics(self) -> Generator[Atomic, None, None]:
        yield from self.predicate.atomics()

    def __str__(self):
        return f"(not {self.predicate})"


if __name__ == "__main__":
    pred = And(Atomic("x", Operator.GTE, 100), Atomic("x", Operator.LTE, 1000))
    print(pred)
    print(pred.evaluate({"x": 500}))
    print(pred.evaluate({"x": 50}))
    print(pred.range_may_satisfy({"x": (500, 1000)}))
    print(pred.range_may_satisfy({"x": (50, 200)}))
    print(pred.range_may_satisfy({"x": (0, 10)}))
    print(list(pred.atomics()))
