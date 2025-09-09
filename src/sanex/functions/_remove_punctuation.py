import pandas as pd
import polars as pl
import string
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def remove_punctuation(df: DataFrameType, columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Removes punctuation from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str], optional): List of column names to consider for punctuation removal.
        Defaults to None (all string columns).

    Returns:
    DataFrameType: DataFrame with punctuation removed from string entries.
    """
    if isinstance(df, pd.DataFrame):
        # Determine which columns to process
        if columns:
            str_cols = [col for col in columns if col in df.columns and df[col].dtype in ['object', 'string']]
        else:
            str_cols = df.select_dtypes(include=['object', 'string']).columns

        df_copy = df.copy()
        for col in str_cols:
            # Remove punctuation using string.punctuation
            df_copy[col] = df_copy[col].str.translate(str.maketrans('', '', string.punctuation))

        return df_copy

    elif isinstance(df, pl.DataFrame):
        # Determine which columns to process
        if columns:
            str_cols = [col for col in columns if col in df.columns and df[col].dtype == pl.String]
        else:
            str_cols = [col for col in df.columns if df[col].dtype == pl.String]

        df_copy = df.clone()
        for col in str_cols:
            # Remove punctuation using regex
            df_copy = df_copy.with_columns(
                pl.col(col).str.replace_all(r'[^\w\s]', '').alias(col)
            )

        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")
