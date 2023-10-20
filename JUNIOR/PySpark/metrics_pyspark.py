from typing import Any, Dict, Union, List
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pyspark.sql as ps


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: Union[pd.DataFrame, ps.DataFrame]) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count

        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}



@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"
    
    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        mask = df[self.columns[0]].isna()
        if self.aggregation == "any":
            for column in self.columns[1:]:
                mask |= df[column].isna()
        else:
            for column in self.columns[1:]:
                mask &= df[column].isna()
        
        k = sum(mask)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count, isnan

        n = df.count()
        mask = col(self.columns[0]).isNull() |  isnan(col(self.columns[0]))
        if self.aggregation == "any":
            for column in self.columns[1:]:
                c = col(column)
                mask = mask | (c.isNull() |  isnan(c))
        else:
            for column in self.columns[1:]:
                c = col(column)
                mask = mask & (c.isNull() |  isnan(c))
        
        k = df.filter(mask).count()        
                
        return {"total": n, "count": k, "delta": k / n}

    

@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df.duplicated(subset=self.columns))
        return {"total": n, "count": k, "delta": k / n}
    
    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count
        n = df.count()
        k = df.groupby(self.columns).count().where(col('count') > '1').count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == self.value)
        return {"total": n, "count": k, "delta": k / n}
    
    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col
        n = df.count()
        k = df.filter(col(self.column) == self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] < self.value if self.strict else df[self.column] <= self.value)
        return {"total": n, "count": k, "delta": k / n}
    
    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col
        n = df.count()
        if self.strict:   
            k = df.filter(col(self.column) < self.value).count()
        else:
            k = df.filter(col(self.column) <= self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column_x] < df[self.column_y] if self.strict else df[self.column_x] <= df[self.column_y])
        return {"total": n, "count": k, "delta": k / n}
    
    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, isnan
        n = df.count()
        mask_x = ~isnan(col(self.column_x)) & col(self.column_x).isNotNull()
        mask_y = ~isnan(col(self.column_y)) & col(self.column_y).isNotNull()
        if self.strict:   
            k = df.filter(mask_x & mask_y & (col(self.column_x) < col(self.column_y))).count()
        else:
            k = df.filter(mask_x & mask_y & (col(self.column_x) <= col(self.column_y))).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column_x]/df[self.column_y] < df[self.column_z] if self.strict 
                else df[self.column_x]/df[self.column_y] <= df[self.column_z])
        return {"total": n, "count": k, "delta": k / n}
    
    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, isnan
        n = df.count()
        ration = col(self.column_x)/col(self.column_y)
        mask_x = ~isnan(col(self.column_x)) & col(self.column_x).isNotNull()
        mask_y = ~isnan(col(self.column_y)) & col(self.column_y).isNotNull()
        mask_z = ~isnan(col(self.column_z)) & col(self.column_z).isNotNull()
        mask = mask_x & mask_y & mask_z
        if self.strict:   
            k = df.filter(mask & (ration < col(self.column_z))).count()
        else:
            k = df.filter(mask & (ration <= col(self.column_z))).count()
        return {"total": n, "count": k, "delta": k / n}

@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        alpha = 1 - self.conf
        lcb, ucb = df[self.column].quantile([alpha/2, 1-alpha/2])
        return {"lcb": lcb, "ucb": ucb}
    
    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import percentile_approx
        alpha = 1 - self.conf
        q = df.select(percentile_approx(self.column, [alpha/2, 1-alpha/2])).collect()[0][0]
        return {"lcb": q[0], "ucb": q[1]}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""
    
    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        a = datetime.today ()
        b = datetime.strptime(max(df[self.column]), self.fmt)
        lag = a - b
        return {"today": a.strftime(self.fmt), "last_day": b.strftime(self.fmt), "lag": lag.days}

    def _call_pyspark(self, df: pd.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import max
        a = datetime.today ()
        max_time = df.select(max(self.column)).collect()[0][0]
        b = datetime.strptime(max_time, self.fmt)
        lag = a - b
        return {"today": a.strftime(self.fmt), "last_day": b.strftime(self.fmt), "lag": lag.days}
