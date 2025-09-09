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
            return df.drop_nulls(subset=subset)
        elif axis == 'columns':
            # Polars does not have a direct method to drop columns with nulls based on a condition like pandas.
            # We need to identify columns with any nulls and then drop them.
            if how == 'any':
                cols_to_drop = [col for col in df.columns if df[col].is_null().any()]
            elif how == 'all':
                cols_to_drop = [col for col in df.columns if df[col].is_null().all()]
            else:
                raise ValueError("For polars, 'how' must be 'any' or 'all' when dropping columns.")

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
        if subset is None:
            return df.fillna(value=value)
        else:
            df_copy = df.copy()
            for col in subset:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].fillna(value)
            return df_copy

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            return df.fill_null(value)
        else:
            df_copy = df.clone()
            for col in subset:
                if col in df_copy.columns:
                    df_copy = df_copy.with_column(
                        pl.when(pl.col(col).is_null()).then(value).otherwise(pl.col(col)).alias(col)
                    )
            return df_copy

    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")
