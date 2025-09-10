import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]
SPECIAL_CHAR_REGEX = r'[^a-zA-Z0-9\s]'  # Regex
PII_REGEX = r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'  # Simple SSN pattern
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Email pattern
PHONE_REGEX = r'\b(?:\+?1[-.\s]?|0)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'  # US Phone number pattern
URL_REGEX = r'https?://\S+|www\.\S+'  # URL pattern

def remove_pii(df: DataFrameType, subset: List[str]) -> DataFrameType:
    """
    Removes personally identifiable information (PII) from specified string columns in the DataFrame.
    PII patterns include email addresses, phone numbers, social security numbers, and URLs.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str]): List of column names to consider for PII removal.

    Returns:
    DataFrameType: DataFrame with PII removed from specified columns.
    """
    if isinstance(df, pd.DataFrame):
        for col in subset:
            if col in df.columns and df[col].dtype in ['object', 'string']:
                # Use vectorized string ops; preserve NA values
                df[col] = (
                    df[col]
                    .str.replace(EMAIL_REGEX, '[REDACTED_EMAIL]', regex=True)
                    .str.replace(PHONE_REGEX, '[REDACTED_PHONE]', regex=True)
                    .str.replace(PII_REGEX, '[REDACTED_SSN]', regex=True)
                    .str.replace(URL_REGEX, '[REDACTED_URL]', regex=True)
                )
        return df

    elif isinstance(df, pl.DataFrame):
        for col in subset:
            if col in df.columns and df[col].dtype == pl.String:
                df = df.with_columns(
                    pl.col(col)
                    .str.replace_all(EMAIL_REGEX, '[REDACTED_EMAIL]')
                    .str.replace_all(PHONE_REGEX, '[REDACTED_PHONE]')
                    .str.replace_all(PII_REGEX, '[REDACTED_SSN]')
                    .str.replace_all(URL_REGEX, '[REDACTED_URL]')
                    .alias(col)
                )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
