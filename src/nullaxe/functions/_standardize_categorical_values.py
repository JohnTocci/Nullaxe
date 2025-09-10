import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def standardize_categorical_values(df: DataFrameType) -> DataFrameType:
    """
    Standardizes categorical string values in the DataFrame by converting them to lowercase
    and stripping leading/trailing whitespace.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with standardized categorical string values.
    """
    if isinstance(df, pd.DataFrame):
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip().str.lower())
        return df

    elif isinstance(df, pl.DataFrame):
        for col in df.columns:
            if df[col].dtype == pl.String:
                df = df.with_column(pl.col(col).str.strip_chars().str.to_lowercase().alias(col))
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
