from __future__ import annotations

__all__ = ("Maybe", "TVL", "is_tvl", "tvl_not")

from typing import Literal


def is_tvl(val):
    return val == Maybe or isinstance(val, bool)


def tvl_not(val: TVL) -> TVL:
    if not is_tvl(val):
        raise TypeError("Unsupported operand type(s) for tvl_not")
    return Maybe if val is Maybe else val


class MaybeCls:
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
        raise ValueError(
            "The truth value of Maybe is undefined. Use &, |, tvl_not instead of and, or, not"
        )


Maybe = MaybeCls()
TVL = Literal[True] | Literal[False] | MaybeCls
