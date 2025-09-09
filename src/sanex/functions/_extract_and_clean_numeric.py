import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def extract_and_clean_numeric(df: DataFrameType, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Extracts numeric values from string entries in the DataFrame and converts them to numeric types.
    Non-numeric entries are set to NaN.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for numeric extraction.
        Defaults to None (all columns).

    Returns:
    DataFrameType: DataFrame with numeric values extracted and cleaned.
    """
    if isinstance(df, pd.DataFrame):
        if subset is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            new_col = f"{col}_numeric"
            # First clean the text by removing currency symbols and commas
            cleaned_text = df[col].astype(str).str.replace('[$€£¥]', '', regex=True).str.replace(',', '')
            # Handle 'k' suffix for thousands
            cleaned_text = cleaned_text.str.replace('k', '000', case=False)

            # Extract numeric values using regex - handles decimals and negatives
            numeric_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
            extracted = cleaned_text.str.extract(f'({numeric_pattern})', expand=False)

            # Convert to numeric
            df[new_col] = pd.to_numeric(extracted, errors='coerce')

        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            columns_to_process = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            df = df.with_column(
                pl.col(col)
                .str.extract(r'([-+]?\d*\.?\d+)', 1)
                .cast(pl.Float64, strict=False)
                .alias(col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")

def clean_numeric(df: DataFrameType, columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Cleans and converts string columns to numeric values by removing currency symbols, commas, etc.

    Parameters:
    df (DataFrameType): Input DataFrame.
    columns (List[str], optional): List of column names to clean. Defaults to None (all string columns).

    Returns:
    DataFrameType: DataFrame with cleaned numeric columns.
    """
    if isinstance(df, pd.DataFrame):
        if columns is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in columns if col in df.columns]

        df_copy = df.copy()
        for col in str_cols:
            # Clean numeric values by removing currency symbols and formatting
            cleaned = df_copy[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('€', '').str.replace('£', '').str.replace('¥', '')
            # Handle 'k' suffix for thousands
            cleaned = cleaned.str.replace('k', '000', case=False)
            df_copy[col] = pd.to_numeric(cleaned, errors='coerce')

        return df_copy

    elif isinstance(df, pl.DataFrame):
        if columns is None:
            str_cols = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            str_cols = [col for col in columns if col in df.columns]

        df_copy = df.clone()
        for col in str_cols:
            df_copy = df_copy.with_columns(
                pl.col(col).str.replace_all(',', '').str.replace_all('[$€£¥]', '').str.replace_all('k', '000').cast(pl.Float64, strict=False).alias(col)
            )

        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")
