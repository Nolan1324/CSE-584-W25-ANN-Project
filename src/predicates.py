from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, NamedTuple

from tvl import Maybe, TVL, tvl_not


class Operator(str, Enum):
    """A binary operator in an atomic predicate. Currently only >= is currently supported."""

    GTE = ">="


class Range(NamedTuple):
    """An integer range, inclusive.

    Attributes:
        start (int | None): Start of the range, inclusive. If `None`, interpreted as negative infinity.
        end (int | None): End of the range, inclusive. If `None`, interpreted as positive infinity.
    """

    start: int = None
    end: int = None

    def contains(self, val: int) -> bool:
        """Checks if the range contains an integer value.

        Args:
            val (int): The value to check

        Returns:
            bool: `True` if the range contains `val`, `False` otherwise.
        """
        return self.start <= val <= self.end


class Predicate(ABC):
    """A logical predicate on one or many scalar attributes."""

    @abstractmethod
    def evaluate(self, vals: Dict[str, int]) -> bool:
        """Evaluate the predicate on a set of attributes.

        Args:
            vals (Dict[str, int]): Mapping from attribute name to value of the attribute.

        Returns:
            bool: `true` if the attributes satisfy the predicate, `false` otherwise.
        """
        pass

    @abstractmethod
    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        """Given a range of possible values for each attribute,
        tests if the predicate satisfies **all** values in the ranges, **some** values, or **no** values.

        Args:
            ranges (Dict[str, Range]): Mapping from attribute name to range of possible values of the attribute.

        Returns:
            TVL: A three-value logic (`True`, `Maybe`, or `False`). The ouputs are as follows:
            - `True` - the predicate always satisfies the values in the ranges.
            - `Maybe` - the predicate sometimes satisfies values in the ranges.
            - `False` - the predicate never satisfies values in the ranges.
        """

    @abstractmethod
    def atomics(self) -> Generator[Atomic, None, None]:
        """Breaks a predicate down into all of its atomic predicates.

        Yields:
            Generator[Atomic, None, None]: Each atomic predicate.
        """
        pass

    @abstractmethod
    def to_filter_string(self) -> str:
        """Represent the predicate as a string that can be used as a filter in Milvus.

        Returns:
            str: The filter string.
        """
        pass

    def __repr__(self):
        return self.to_filter_string()


@dataclass(frozen=True)
class Atomic(Predicate):
    """A unary predicate that cannot be further broken down into atomics.
    Is of the form (attr op value) where attr is an attribute name, op is a binary operator (like >=), and `value` is a constant.

    Attributes:
        attr: Name of the attribute.
        op: The operator type.
        value: The value that the attribute is compared against.
    """

    attr: str
    op: Operator
    value: int

    def __post_init__(self):
        object.__setattr__(self, "attr", str(self.attr))

    def evaluate(self, vals: Dict[str, int]) -> bool:
        x = vals[self.attr]
        match self.op:
            case Operator.GTE:
                return x >= self.value

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        if self.attr not in ranges:
            return Maybe
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
            case _:
                assert False

    def atomics(self) -> Generator[Atomic, None, None]:
        """
        A generator method that yields the current instance as an atomic element.

        Yields:
            The current instance as an atomic element.
        """

        yield self

    def to_filter_string(self) -> str:
        return f"{self.attr} {str(self.op.value)} {self.value}"

    def __repr__(self):
        return self.to_filter_string()


class And(Predicate):
    """An "and" of many predicates."""

    def __init__(self, *predicates: List[Predicate]):
        """
        Constructs an "and" predicate.
        Args:
            *predicates (List[Predicate]): The predicates on which "and" applies to.
        """

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

    def to_filter_string(self) -> str:
        return " and ".join(f"({predicate})" for predicate in self.predicates)

    def __repr__(self):
        return self.to_filter_string()


class Or(Predicate):
    """An "or" of many predicates."""

    def __init__(self, *predicates: List[Predicate]):
        """
        Constructs an "or" predicate.
        Args:
            *predicates (List[Predicate]): The predicates on which "or" applies to.
        """
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

    def to_filter_string(self) -> str:
        return " or ".join(f"({predicate})" for predicate in self.predicates)

    def __repr__(self):
        return self.to_filter_string()


class Not(Predicate):
    """The negation of a predicate."""

    def __init__(self, predicate: Predicate):
        """
        Constructs a "not" predicate.
        Args:
            predicates (Predicate): The predicate which is being negated.
        """
        self.predicate = predicate

    def evaluate(self, vals: Dict[str, int]) -> bool:
        return not self.predicate.evaluate(vals)

    def range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL:
        val = self.predicate.range_may_satisfy(ranges)
        return tvl_not(val)

    def atomics(self) -> Generator[Atomic, None, None]:
        yield from self.predicate.atomics()

    def to_filter_string(self) -> str:
        return f"not({self.predicate})"

    def __repr__(self):
        return self.to_filter_string()


if __name__ == "__main__":
    # example use case
    pred = And(Atomic("x", Operator.GTE, 100), Not(Atomic("x", Operator.GTE, 1001)))
    print(pred)
    print(pred.evaluate({"x": 500}))
    print(pred.evaluate({"x": 50}))
    print(pred.range_may_satisfy({"x": Range(500, 1000)}))
    print(pred.range_may_satisfy({"x": Range(50, 200)}))
    print(pred.range_may_satisfy({"x": Range(0, 10)}))
    print(list(pred.atomics()))
