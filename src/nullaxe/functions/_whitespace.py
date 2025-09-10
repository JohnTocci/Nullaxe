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
        df_copy = df.copy()
        str_cols = df_copy.select_dtypes(include=['object', 'string']).columns

        for col in str_cols:
            # Only apply strip to actual string values, preserve other types including 'MISSING'
            df_copy[col] = df_copy[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        return df_copy

    elif isinstance(df, pl.DataFrame):
        for col in df.columns:
            if df[col].dtype == pl.String:
                df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")