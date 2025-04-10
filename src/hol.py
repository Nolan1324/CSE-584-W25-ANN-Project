__all__ = ("Maybe", "HOL")

from typing import Literal


class MaybeCls:
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
        return "Maybe"

    def __bool__(self):
        raise ValueError("The truth value of Maybe is undefined")

    @staticmethod
    def __is_hol(val):
        return val == Maybe or isinstance(val, bool)


Maybe = MaybeCls()
HOL = Literal[True] | Literal[False] | MaybeCls
