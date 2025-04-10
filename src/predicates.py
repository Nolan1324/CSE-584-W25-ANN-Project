from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Literal, Optional, Tuple

class MaybeCls():
    def __and__(self, other):
        if not self.__is_hol(other):
            raise TypeError("Unsupported operand type(s) for and")
        return {
            True: Maybe,
            False: False,
            Maybe: Maybe,
        }[other]
    
    def __or__(self, other):
        if not self.__is_hol(other):
            raise TypeError("Unsupported operand type(s) for or")
        return {
            True: True,
            False: Maybe,
            Maybe: Maybe,
        }[other]
    
    def __invert__(self):
        return Maybe
    
    def __rand__(self, other):
        return self.__and__(other)
    
    def __ror__(self, other):
        return self.__or__(other)
    
    def __repr__(self):
        return 'Maybe'
    
    def __bool__(self):
        raise ValueError('The truth value of Maybe is undefined')
    
    @staticmethod
    def __is_hol(val):
        return val == Maybe or isinstance(val, bool)

Maybe = MaybeCls()

HOL = Literal[True] | Literal[False] | MaybeCls

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
    def range_may_satisfy(self, ranges: Dict[str, Range]) -> HOL:
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

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> HOL:
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


class And(Predicate):
    def __init__(self, *predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return all(predicate.evaluate(vals) for predicate in self.predicates)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> HOL:
        val = True
        for predicate in self.predicates:
            val &= predicate.range_may_satisfy(ranges)
        return val

    def atomics(self) -> Generator[Atomic, None, None]:
        for predicate in self.predicates:
            yield from predicate.atomics()


class Or(Predicate):
    def __init__(self, *predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return any(predicate.evaluate(vals) for predicate in self.predicates)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> HOL:
        val = False
        for predicate in self.predicates:
            val |= predicate.range_may_satisfy(ranges)
        return val

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
