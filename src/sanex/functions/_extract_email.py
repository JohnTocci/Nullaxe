import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

def extract_email(df: DataFrameType, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Extracts email addresses from string entries in the DataFrame and places them in a new column.
    Non-email entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for email extraction.
        Defaults to None (all columns).

    Returns:
    DataFrameType: DataFrame with email addresses extracted.
    """
    if isinstance(df, pd.DataFrame):
        if subset is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            new_col = f"{col}_email"
            df[new_col] = df[col].str.extract(EMAIL_REGEX, expand=False)
        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            columns_to_process = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            new_col = f"{col}_email"
            df = df.with_column(
                pl.col(col)
                .str.extract(EMAIL_REGEX, 0)
                .alias(new_col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
