import pandas as pd
import polars as pl
from typing import Union
DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def remove_duplicates(df: DataFrameType) -> DataFrameType:
    """
    Remove duplicate columns and rows from a pandas or polars DataFrame.

    Parameters:
    df (pd.DataFrame | pl.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame | pl.DataFrame: DataFrame with duplicate columns and rows removed.
    """
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.drop_duplicates()
    elif isinstance(df, pl.DataFrame):
        unique_cols = list(dict.fromkeys(df.columns))
        df = df.select(unique_cols)
        df = df.unique()
    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")

    return df

