import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def drop_single_value_columns(df: DataFrameType) -> DataFrameType:
    """
    Drops columns that contain only a single unique value from the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with single-value columns dropped.
    """
    if isinstance(df, pd.DataFrame):
        return df.loc[:, df.nunique() > 1]

    elif isinstance(df, pl.DataFrame):
        cols_to_drop = [col for col in df.columns if df[col].n_unique() <= 1]
        return df.drop(cols_to_drop)

    raise TypeError("Input must be a pandas or polars DataFrame.")

