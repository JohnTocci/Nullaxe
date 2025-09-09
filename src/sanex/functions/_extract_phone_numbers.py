import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]
PHONE_REGEX = re.compile(r'\+?(\d[\d-. ]+)?(\([\d-. ]+\))?[\d-. ]+\d')
# This regex matches various phone number formats, including optional country codes and area codes.

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
        if subset is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            new_col = f"{col}_phone"
            df[new_col] = df[col].str.extract(PHONE_REGEX, expand=False)[0]
        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            columns_to_process = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            new_col = f"{col}_phone"
            df = df.with_column(
                pl.col(col)
                .str.extract(PHONE_REGEX, 0)
                .alias(new_col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
