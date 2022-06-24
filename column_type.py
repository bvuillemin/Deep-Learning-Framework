"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from enum import Enum, auto


class ColumnType(Enum):
    """
    Column type (see https://docs.python.org/3/library/enum.html#flag)

    """
    DATE = auto()
    CASEID = auto()
    BOOLEAN = auto()
    QUALITATIVE = auto()
    QUANTITATIVE = auto()
    ANY = auto()
    JUNK = auto()

    def __add__(self, o):
        if self == o:
            return self
        if self == ColumnType.ANY or o == ColumnType.ANY:
            return ColumnType.ANY
        if self == ColumnType.JUNK or o == ColumnType.JUNK:
            return ColumnType.JUNK
        if self == ColumnType.QUALITATIVE and o == ColumnType.QUANTITATIVE or (
                self == ColumnType.QUANTITATIVE and o == ColumnType.QUALITATIVE):
            return ColumnType.QUALITATIVE
        return ColumnType.JUNK
