import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]
PHONE_REGEX = re.compile(r'\+?(\d[\d-. ]+)?(\([\d-. ]+\))?[\d-. ]+\d')
# This regex matches various phone number formats, including optional country codes and area codes.

# Helper function to reduce code duplication
def _get_string_columns(df, subset=None):
    if isinstance(df, pd.DataFrame):
        if subset is None:
            return df.select_dtypes(include=['object', 'string']).columns
        else:
            return [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]
    elif isinstance(df, pl.DataFrame):
        if subset is None:
            return [col for col in df.columns if df[col].dtype == pl.String]
        else:
            return [col for col in subset if col in df.columns and df[col].dtype == pl.String]
    return []

def extract_phone_numbers(df: DataFrameType, subset: List[str] = None) -> DataFrameType:
    """
    Extracts phone numbers from string entries in the DataFrame and places them in new columns.
    Non-phone number entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for phone number extraction.
        Defaults to None (all columns).

    Returns:
    DataFrameType: DataFrame with phone numbers extracted.
    """
    if isinstance(df, pd.DataFrame):
        str_cols = _get_string_columns(df, subset)
        for col in str_cols:
            new_col = f"{col}_phone"
            df[new_col] = df[col].str.extract(PHONE_REGEX, expand=False)
        return df

    elif isinstance(df, pl.DataFrame):
        str_cols = _get_string_columns(df, subset)
        for col in str_cols:
            new_col = f"{col}_phone"
            # Use regex pattern string for polars, not the compiled pattern
            df = df.with_column(
                pl.col(col)
                .str.extract(PHONE_REGEX.pattern)
                .alias(new_col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
