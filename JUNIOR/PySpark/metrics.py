"""Metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
import datetime

import pandas as pd


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"lcb": lcb, "ucb": ucb}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...
        return {"today": a, "last_day": b, "lag": lag}
