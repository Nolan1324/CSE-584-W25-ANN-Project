from __future__ import annotations

from typing import Literal


__all__ = ["Maybe", "TVL", "is_tvl", "tvl_not"]


def is_tvl(val: TVL) -> bool:
    """Returns whether a given value is a valid TVL value."""
    return val == Maybe or isinstance(val, bool)


def tvl_not(val: TVL) -> TVL:
    """Implements the logical NOT operation for TVL values.

    Due to Python quirks, we can't implement this as a method of the Maybe class.
    """
    if not is_tvl(val):
        raise TypeError("Unsupported operand type(s) for tvl_not")
    return Maybe if val is Maybe else not val


class MaybeCls:
    """
    A class representing the "Maybe" value in a three-valued logic (TVL) system.
    The "Maybe" value is used to represent an indeterminate or unknown state in logical operations.
    This class provides overloaded operators for logical AND (&) and OR (|) operations, as well as
    their reversed counterparts (rand and ror), to work with other TVL values.

    Notes
    - The "Maybe" value is part of a three-valued logic system, where the possible values are:
      True, False, and Maybe.
    - The behavior of logical operations with "Maybe" is defined as follows:
        - AND (&):
            - True & Maybe -> Maybe
            - False & Maybe -> False
            - Maybe & Maybe -> Maybe
        - OR (|):
            - True | Maybe -> True
            - False | Maybe -> Maybe
            - Maybe | Maybe -> Maybe
    """

    def __and__(self, other: TVL) -> TVL:
        if not is_tvl(other):
            raise TypeError("Unsupported operand type(s) for &")
        return {
            True: Maybe,
            False: False,
            Maybe: Maybe,
        }[other]

    def __or__(self, other: TVL) -> TVL:
        if not is_tvl(other):
            raise TypeError("Unsupported operand type(s) for |")
        return {
            True: True,
            False: Maybe,
            Maybe: Maybe,
        }[other]

    def __rand__(self, other):
        return self.__and__(other)

    def __ror__(self, other):
        return self.__or__(other)

    def __repr__(self):
        return "Maybe"

    def __bool__(self):
        raise ValueError("The truth value of Maybe is undefined. Use &, |, tvl_not instead of and, or, not")


# maybe is a singleton instance of MaybeCls, and TVL should always use this alias
Maybe = MaybeCls()
TVL = Literal[True] | Literal[False] | MaybeCls
