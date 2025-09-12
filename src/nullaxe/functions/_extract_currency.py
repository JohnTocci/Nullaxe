import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def extract_currency(df: DataFrameType, subset: List[str]) -> DataFrameType:
    """
    Extracts currency values from string entries in the DataFrame and places them in new columns.
    Non-currency entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str]): List of column names to consider for currency extraction.

    Returns:
    DataFrameType: DataFrame with currency values extracted.
    """
    # Regex to match currency patterns - simplified and correct
    CURRENCY_REGEX = r'(\$|€|£|¥|₹)?\s?(\d+(?:,\d{3})*(?:\.\d{1,2})?)'

    if isinstance(df, pd.DataFrame):
        for col in subset:
            if col in df.columns and df[col].dtype in ['object', 'string']:
                new_col = f"{col}_currency"
                # Use str.findall to get the full match, then take first occurrence
                matches = df[col].str.findall(CURRENCY_REGEX)
                
                def extract_first_match(match_list):
                    if match_list and len(match_list) > 0:
                        # Join the captured groups (symbol and number)
                        symbol, number = match_list[0]
                        return f"{symbol}{number}".strip()
                    return pd.NA
                
                df[new_col] = matches.apply(extract_first_match)
        return df

    elif isinstance(df, pl.DataFrame):
        for col in subset:
            if col in df.columns and df[col].dtype == pl.Utf8:
                new_col = f"{col}_currency"
                # Use extract with group 0 to get the full match
                df = df.with_columns(
                    pl.col(col)
                    .str.extract(CURRENCY_REGEX, 0)  # Extract full match (group 0)
                    .alias(new_col)
                )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
