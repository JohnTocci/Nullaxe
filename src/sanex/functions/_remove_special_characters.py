import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def remove_special_characters(df: DataFrameType, columns: List[str], characters: str = None) -> DataFrameType:
    """
    Removes specified special characters from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str]): List of column names to consider for special character removal.
    characters (str): String of special characters to remove. Defaults to common special characters.

    Returns:
    DataFrameType: DataFrame with specified special characters removed from string entries.
    """
    if characters is None:
        # Default set of special characters
        characters = r"""!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

    # Create regex pattern to match any of the specified characters
    pattern = f"[{re.escape(characters)}]"

    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
                df_copy[col] = df_copy[col].str.replace(pattern, '', regex=True)
        return df_copy

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype == pl.String:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(pattern, '').alias(col)
                )
        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_emojis(df: DataFrameType, columns: List[str]) -> DataFrameType:
    """
    Removes emojis from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str]): List of column names to consider for emoji removal.

    Returns:
    DataFrameType: DataFrame with emojis removed from string entries.
    """
    # Emoji regex pattern as string (usable in both pandas and polars)
    emoji_pattern_str = (
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\u2702-\u27B0"
        "\u24C2-\U0001F251"
        "]+"
    )

    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
                df_copy[col] = df_copy[col].astype(str).str.replace(emoji_pattern_str, '', regex=True)
        return df_copy

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype == pl.String:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(emoji_pattern_str, '').alias(col)
                )
        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_non_alphanumeric(df: DataFrameType, columns: List[str]) -> DataFrameType:
    """
    Removes all non-alphanumeric characters from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str]): List of column names to consider for non-alphanumeric character removal.

    Returns:
    DataFrameType: DataFrame with non-alphanumeric characters removed from string entries.
    """
    pattern = r'[^a-zA-Z0-9\s]'

    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
                df_copy[col] = df_copy[col].str.replace(pattern, '', regex=True)
        return df_copy

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype == pl.String:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(pattern, '').alias(col)
                )
        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_non_numeric(df: DataFrameType, columns: List[str]) -> DataFrameType:
    """
    Removes all non-numeric characters from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str]): List of column names to consider for non-numeric character removal.

    Returns:
    DataFrameType: DataFrame with non-numeric characters removed from string entries.
    """
    pattern = r'[^0-9]'

    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
                df_copy[col] = df_copy[col].str.replace(pattern, '', regex=True)
        return df_copy

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype == pl.String:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(pattern, '').alias(col)
                )
        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_non_ascii(df: DataFrameType, columns: List[str]) -> DataFrameType:
    """
    Removes all non-ASCII characters from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str]): List of column names to consider for non-ASCII character removal.

    Returns:
    DataFrameType: DataFrame with non-ASCII characters removed from string entries.
    """
    pattern = r'[^\x00-\x7F]'

    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
                df_copy[col] = df_copy[col].str.replace(pattern, '', regex=True)
        return df_copy

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype == pl.String:
                df_copy = df_copy.with_columns(
                    pl.col(col).str.replace_all(pattern, '').alias(col)
                )
        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")
