import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def remove_whitespace(df: DataFrameType) -> DataFrameType:
    """
    Removes leading and trailing whitespace from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with whitespace removed from string entries.
    """
    if isinstance(df, pd.DataFrame):
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        df[str_cols] = df[str_cols].applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df

    elif isinstance(df, pl.DataFrame):
        for col in df.columns:
            if df[col].dtype == pl.String:
                df = df.with_column(pl.col(col).str.strip_chars().alias(col))
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")