import pandas as pd
import polars as pl
from typing import Union, Optional, List


DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def replace_text(df: DataFrameType, old: str = None, new: str = None, columns: Optional[List[str]] = None, regex: bool = False, to_replace: str = None, value: str = None, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Replaces occurrences of a specified substring with another substring in string columns of the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    old (str): The substring or pattern to be replaced (preferred parameter name).
    new (str): The string to replace with (preferred parameter name).
    columns (Optional[List[str]]): List of column names to consider for replacement. Default is None (all string columns).
    regex (bool): Whether to treat 'old' as a regular expression. Default is False.
    to_replace (str): Alternative parameter name for backward compatibility.
    value (str): Alternative parameter name for backward compatibility.
    subset (Optional[List[str]]): Alternative parameter name for backward compatibility.

    Returns:
    DataFrameType: DataFrame with text replaced in specified string columns.
    """
    # Handle parameter compatibility
    if old is None and to_replace is not None:
        old = to_replace
    if new is None and value is not None:
        new = value
    if columns is None and subset is not None:
        columns = subset

    if old is None or new is None:
        raise ValueError("Both 'old' and 'new' parameters are required")

    if isinstance(df, pd.DataFrame):
        # Determine which columns to process
        if columns:
            str_cols = [col for col in columns if col in df.columns and df[col].dtype in ['object', 'string']]
        else:
            str_cols = df.select_dtypes(include=['object', 'string']).columns

        df_copy = df.copy()
        for col in str_cols:
            if regex:
                df_copy[col] = df_copy[col].str.replace(old, new, regex=True)
            else:
                df_copy[col] = df_copy[col].str.replace(old, new, regex=False)

        return df_copy

    elif isinstance(df, pl.DataFrame):
        # Determine which columns to process
        if columns:
            str_cols = [col for col in columns if col in df.columns and df[col].dtype == pl.String]
        else:
            str_cols = [col for col in df.columns if df[col].dtype == pl.String]

        df_copy = df.clone()
        for col in str_cols:
            if regex:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(old, new).alias(col)
                )
            else:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(old, new, literal=True).alias(col)
                )

        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")