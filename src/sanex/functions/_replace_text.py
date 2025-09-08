import pandas as pd
import polars as pl
from typing import Union, Optional, List


DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def replace_text(df: DataFrameType, to_replace: str, value: str, subset: Optional[List[str]] = None, regex: bool = False) -> DataFrameType:
    """
    Replaces occurrences of a specified substring with another substring in string columns of the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    to_replace (str): The substring or pattern to be replaced.
    value (str): The string to replace with.
    subset (Optional[List[str]]): List of column names to consider for replacement. Default is None (all string columns).
    regex (bool): Whether to treat 'to_replace' as a regular expression. Default is False.

    Returns:
    DataFrameType: DataFrame with text replaced in specified string columns.
    """
    if isinstance(df, pd.DataFrame):
        # Determine which columns to process
        if subset:
            # Filter subset to only include string columns that exist in the DataFrame
            str_cols = [col for col in subset if col in df.columns]
        else:
            # Use all string columns
            str_cols = df.select_dtypes(include=['object', 'string']).columns

        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Replace text in the selected columns
        for col in str_cols:
            if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
                df_copy[col] = df_copy[col].str.replace(to_replace, value, regex=regex)

        return df_copy

    elif isinstance(df, pl.DataFrame):
        # Determine which columns to process
        if subset:
            # Filter subset to only include columns that exist in the DataFrame
            str_cols = [col for col in subset if col in df.columns]
        else:
            # Use all string columns
            str_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]

        # Replace text in the selected columns
        for col_name in str_cols:
            if col_name in df.columns and df[col_name].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(col_name).str.replace_all(to_replace, value, literal=(not regex)).alias(col_name)
                )

        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")