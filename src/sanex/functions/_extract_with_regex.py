import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def extract_with_regex(df: DataFrameType, pattern: str, columns: List[str], new_column: str = None, subset: List[str] = None) -> DataFrameType:
    """
    Extracts substrings matching a given regex pattern from specified columns in the DataFrame
    and places them in new columns. Non-matching entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    pattern (str): Regular expression pattern to match.
    columns (List[str]): List of column names to consider for extraction.
    new_column (str): Name for the new column containing extracted values.
    subset (List[str]): Alternative parameter name for backward compatibility.

    Returns:
    DataFrameType: DataFrame with substrings extracted.
    """
    # Handle parameter compatibility
    if columns is None and subset is not None:
        columns = subset

    if columns is None:
        raise ValueError("'columns' parameter is required")

    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()

        for col in columns:
            if col not in df_copy.columns:
                continue
            if df_copy[col].dtype not in ['object', 'string']:
                continue

            # Create new column name
            col_name = new_column if new_column else f"{col}_extracted"

            # Extract using regex with capture group
            df_copy[col_name] = df_copy[col].str.extract(f'({pattern})', expand=False)

        return df_copy

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()

        for col in columns:
            if col not in df_copy.columns:
                continue
            if df_copy[col].dtype != pl.String:
                continue

            # Create new column name
            col_name = new_column if new_column else f"{col}_extracted"

            # Extract using regex
            df_copy = df_copy.with_columns(
                pl.col(col).str.extract(f'({pattern})', 1).alias(col_name)
            )

        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")
