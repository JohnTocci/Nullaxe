import pandas as pd
import polars as pl
from typing import Union, Dict, Any, Optional, List

# Define constants for recognized boolean and missing values
DataFrameType = Union[pd.DataFrame, pl.DataFrame]
DEFAULT_TRUE_VALUES = {'true', '1', 't', 'yes', 'y', 'on'}
DEFAULT_FALSE_VALUES = {'false', '0', 'f', 'no', 'n', 'off'}
DEFAULT_MISSING_VALUES = {'', 'nan', 'null', 'none'}


def standardize_booleans(
    df: DataFrameType,
    true_values: List[str] = None,
    false_values: List[str] = None,
    subset: List[str] = None
) -> DataFrameType:
    """
    Standardizes boolean-like columns in the DataFrame to actual boolean types.
    Recognizes various representations of true, false, and missing values.

    Parameters:
    df (DataFrameType): Input DataFrame.
    true_values (List[str], optional): List of values to be considered as True.
                                       Default is ['yes', 'y', 'true', 't', '1', 'on'].
    false_values (List[str], optional): List of values to be considered as False.
                                        Default is ['no', 'n', 'false', 'f', '0', 'off'].
    subset (List[str], optional): List of column names to consider for boolean standardization.
                                  Default is None (all columns).

    Returns:
    DataFrameType: DataFrame with standardized boolean columns.
    """
    # Use default values if not provided
    true_set = set(v.lower() for v in true_values) if true_values else DEFAULT_TRUE_VALUES
    false_set = set(v.lower() for v in false_values) if false_values else DEFAULT_FALSE_VALUES
    missing_set = DEFAULT_MISSING_VALUES
    all_values = true_set | false_set | missing_set

    if isinstance(df, pd.DataFrame):
        # Create a mapping dictionary for efficient conversion
        mapping: Dict[str, Optional[bool]] = {v: True for v in true_set}
        mapping.update({v: False for v in false_set})
        mapping.update({v: None for v in missing_set})

        # Work on a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Determine which columns to process
        columns_to_process = subset if subset else df_copy.select_dtypes(include=['object', 'string']).columns

        # Iterate only over specified columns or object/string columns
        for col in columns_to_process:
            if col not in df_copy.columns:
                continue

            # Skip non-string columns if they're in the subset
            if subset and df_copy[col].dtype not in ['object', 'string']:
                continue

            # Clean the series to check if all values are boolean-like
            cleaned_series = df_copy[col].astype(str).str.lower().str.strip()

            # If all values are in our defined set, perform the conversion
            if cleaned_series.isin(all_values).all():
                df_copy[col] = cleaned_series.map(mapping).astype('boolean')

        return df_copy

    elif isinstance(df, pl.DataFrame):
        # Determine which columns to process
        columns_to_process = subset if subset else df.columns

        # Iterate through selected columns
        for col in columns_to_process:
            if col not in df.columns:
                continue

            # Skip columns that are already boolean
            if df[col].dtype == pl.Boolean:
                continue

            # For string columns, check if they can be standardized
            if df[col].dtype == pl.Utf8:
                # Clean the column values
                lower_col = df[col].str.to_lowercase().str.strip_chars()

                # If all values are boolean-like, convert the column
                if lower_col.is_in(list(all_values)).all():
                    df = df.with_column(
                        pl.when(lower_col.is_in(list(true_set))).then(True)
                        .when(lower_col.is_in(list(false_set))).then(False)
                        .otherwise(None).alias(col)
                    )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")