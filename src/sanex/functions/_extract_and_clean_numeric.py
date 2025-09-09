import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def extract_and_clean_numeric(df: DataFrameType, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Extracts numeric values from string entries in the DataFrame and converts them to numeric types.
    Non-numeric entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for numeric extraction.
        Defaults to None (all columns).

    Returns:
    DataFrameType: DataFrame with numeric values extracted and cleaned.
    """
    if isinstance(df, pd.DataFrame):
        if subset is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            df[col] = pd.to_numeric(df[col].str.extract(r'([-+]?\d*\.?\d+)', expand=False), errors='coerce')
        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            columns_to_process = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            df = df.with_column(
                pl.col(col)
                .str.extract(r'([-+]?\d*\.?\d+)', 1)
                .cast(pl.Float64, strict=False)
                .alias(col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")

def clean_numeric(df: DataFrameType, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Cleans numeric columns in the DataFrame by converting non-numeric entries to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for cleaning.
        Defaults to None (all numeric columns).

    Returns:
    DataFrameType: DataFrame with cleaned numeric columns.
    """
    if isinstance(df, pd.DataFrame):
        if subset is None:
            num_cols = df.select_dtypes(include=['number']).columns
        else:
            num_cols = [col for col in subset if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            num_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
        else:
            num_cols = [col for col in subset if col in df.columns and df[col].dtype.is_numeric()]

        for col in num_cols:
            df = df.with_column(
                pl.col(col).cast(pl.Float64, strict=False).alias(col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
