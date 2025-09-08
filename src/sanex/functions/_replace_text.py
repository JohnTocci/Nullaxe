import pandas as pd
import polars as pl
from typing import Union


DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def replace_text(df: DataFrameType, to_replace: str, value: str, regex: bool = False) -> DataFrameType:
    """
    Replaces occurrences of a specified substring with another substring in all string columns of the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    to_replace (str): The substring or pattern to be replaced.
    value (str): The string to replace with.
    regex (bool): Whether to treat 'to_replace' as a regular expression. Default is False.

    Returns:
    DataFrameType: DataFrame with text replaced in string columns.
    """
    if isinstance(df, pd.DataFrame):
        # 2. Use the more efficient, vectorized str.replace method
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            df[col] = df[col].str.replace(to_replace, value, regex=regex)
        return df

    elif isinstance(df, pl.DataFrame):
        # 3. Use replace_all for consistency
        str_cols = [col.name for col in df.select(pl.col(pl.Utf8))]
        for col_name in str_cols:
            df = df.with_columns(
                pl.col(col_name).str.replace_all(to_replace, value, literal=(not regex)).alias(col_name)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")