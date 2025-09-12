import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

URL_REGEX = r'(https?://[^\s]+)'  # Regex pattern to match URLs with a capture group

def extract_urls(df: DataFrameType, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Extracts URLs from string entries in the DataFrame and places them in new columns.
    Non-URL entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for URL extraction.
        Defaults to None (all columns).

    Returns:
    DataFrameType: DataFrame with URLs extracted.
    """
    if isinstance(df, pd.DataFrame):
        if subset is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            new_col = f"{col}_url"
            df[new_col] = df[col].str.extract(URL_REGEX, expand=False)
        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            columns_to_process = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            new_col = f"{col}_url"
            df = df.with_columns(
                pl.col(col)
                .str.extract(URL_REGEX, 1)  # Extract first capture group
                .alias(new_col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
