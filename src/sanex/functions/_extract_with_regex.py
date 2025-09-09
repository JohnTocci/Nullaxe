import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def extract_with_regex(df: DataFrameType, pattern: str, subset: List[str]) -> DataFrameType:
    """
    Extracts substrings matching a given regex pattern from specified columns in the DataFrame
    and places them in new columns. Non-matching entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    pattern (str): Regular expression pattern to match.
    subset (List[str]): List of column names to consider for extraction.

    Returns:
    DataFrameType: DataFrame with substrings extracted.
    """
    if isinstance(df, pd.DataFrame):
        str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            new_col = f"{col}_extracted"
            df[new_col] = df[col].str.extract(pattern, expand=False)
        return df

    elif isinstance(df, pl.DataFrame):
        columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            new_col = f"{col}_extracted"
            df = df.with_column(
                pl.col(col)
                .str.extract(pattern, 0)
                .alias(new_col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")

