import pandas as pd
import polars as pl
from typing import Union, Optional, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def drop_missing(df: DataFrameType, axis: str = 'rows',
                how: str = 'any',
                thresh: Optional[int] = None,
                subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Drops rows or columns with missing values from the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    axis (str): The axis to drop from. 'rows' to drop rows, 'columns' to drop columns. Default is 'rows'.
    how (str): 'any' to drop if any NA values are present, 'all' to drop if all values are NA. Default is 'any'.
    thresh (int): Require that many non-NA values to avoid dropping. Default is None.
    subset (list): List of column names to consider for dropping when axis is 'rows'. Default is None.

    Returns:
    DataFrameType: DataFrame with missing values dropped.
    """
    # Validate the axis input
    if axis not in ['rows', 'columns']:
        raise ValueError("Axis must be either 'rows' or 'columns'.")

    # Validate the how input
    if how not in ['any', 'all']:
        raise ValueError("how must be either 'any' or 'all'.")

    if isinstance(df, pd.DataFrame):
        # Map 'rows' to 0 and 'columns' to 1 for pandas
        pandas_axis = 0 if axis == 'rows' else 1

        # Handle parameter conflicts - pandas doesn't allow both how and thresh
        if thresh is not None:
            return df.dropna(axis=pandas_axis, thresh=thresh, subset=subset)
        else:
            return df.dropna(axis=pandas_axis, how=how, subset=subset)

    elif isinstance(df, pl.DataFrame):
        if axis == 'rows':
            if how == 'any':
                return df.drop_nulls(subset=subset)
            elif how == 'all':
                # For 'all', only drop rows where ALL values in the row (or subset) are null
                if subset is None:
                    # Check all columns - only drop if all values in row are null
                    mask = df.select([
                        pl.fold(acc=pl.lit(False), function=lambda acc, x: acc | x.is_not_null(), exprs=pl.all()).alias("has_non_null")
                    ])["has_non_null"]
                    return df.filter(mask)
                else:
                    # Check only subset columns - only drop if all values in subset are null
                    mask = df.select([
                        pl.fold(acc=pl.lit(False), function=lambda acc, x: acc | x.is_not_null(),
                               exprs=[pl.col(col) for col in subset if col in df.columns]).alias("has_non_null")
                    ])["has_non_null"]
                    return df.filter(mask)
        elif axis == 'columns':
            # Polars does not have a direct method to drop columns with nulls based on a condition like pandas.
            # We need to identify columns with any nulls and then drop them.
            if how == 'any':
                cols_to_drop = [col for col in df.columns if df[col].is_null().any()]
            elif how == 'all':
                cols_to_drop = [col for col in df.columns if df[col].is_null().all()]

            # The 'thresh' and 'subset' parameters are not directly applicable to dropping columns in Polars in the same way as pandas.
            # You would typically select the columns you want to keep instead.
            if thresh is not None:
                print(
                    "Warning: The 'thresh' parameter is not supported for dropping columns in Polars and will be ignored.")
            if subset is not None and axis == 'columns':
                print("Warning: The 'subset' parameter is not applicable when dropping columns and will be ignored.")

            return df.drop(cols_to_drop)

    # Ensure we always return a DataFrame for all paths
    raise TypeError("Input must be a pandas or polars DataFrame.")

def fill_missing(df: DataFrameType, value: Union[int, float, str] = 0, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Fills missing values in the DataFrame with a specified value.

    Parameters:
    df (DataFrameType): Input DataFrame.
    value (Union[int, float, str]): The value to replace missing values with. Default is 0.
    subset (list): List of column names to consider for filling. Default is None (all columns).

    Returns:
    DataFrameType: DataFrame with missing values filled.
    """
    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()

        if subset is None:
            # Fill all columns
            columns_to_fill = df_copy.columns
        else:
            columns_to_fill = [col for col in subset if col in df_copy.columns]

        for col in columns_to_fill:
            if df_copy[col].isna().any():
                # If we're filling with a string but column is numeric, convert to object type first
                if isinstance(value, str) and pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].astype('object')
                df_copy[col] = df_copy[col].fillna(value)

        return df_copy

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            # Handle different data types properly
            df_result = df.clone()
            for col in df.columns:
                if df[col].dtype == pl.String:
                    df_result = df_result.with_columns(
                        pl.col(col).fill_null(str(value))
                    )
                else:
                    df_result = df_result.with_columns(
                        pl.col(col).fill_null(value)
                    )
            return df_result
        else:
            df_copy = df.clone()
            for col in subset:
                if col in df_copy.columns:
                    if df_copy[col].dtype == pl.String:
                        df_copy = df_copy.with_columns(
                            pl.col(col).fill_null(str(value))
                        )
                    else:
                        df_copy = df_copy.with_columns(
                            pl.col(col).fill_null(value)
                        )
            return df_copy

    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")
