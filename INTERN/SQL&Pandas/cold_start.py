import pandas as pd
import numpy as np


def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    """Function to fill na"""
    df = df.copy()
    group = df.groupby(group)[target].transform('mean')
    df.loc[df[target].isnull(), target] = group.apply(np.floor)
    return df
