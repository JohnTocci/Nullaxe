import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def remove_unwanted_rows_and_cols(df: DataFrameType, unwanted_values: Optional[List[Union[str, int, float]]] = None) -> DataFrameType:
    """
    Removes rows and columns that contain only unwanted values from the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    unwanted_values (List[Union[str, int, float]], optional): List of values considered unwanted.
        Defaults to [None, '', 'NA', 'N/A', 'null', 'NULL', 'NaN'].

    Returns:
    DataFrameType: DataFrame with unwanted rows and columns removed.
    """
    if unwanted_values is None:
        unwanted_values = [None, '', 'NA', 'N/A', 'null', 'NULL', 'NaN']

    if isinstance(df, pd.DataFrame):
        # Remove rows where all values are in unwanted_values
        df = df[~df.apply(lambda row: all(item in unwanted_values for item in row), axis=1)]
        # Remove columns where all values are in unwanted_values
        df = df.loc[:, ~df.apply(lambda col_data: all(item in unwanted_values for item in col_data))]
        return df

    elif isinstance(df, pl.DataFrame):
        # For polars, handle None separately
        null_values = [v for v in unwanted_values if v is None]
        non_null_values = [v for v in unwanted_values if v is not None]

        # Remove rows where all values are unwanted
        row_expressions = []
        for column_name in df.columns:
            expr = ~(pl.col(column_name).is_in(non_null_values))
            if null_values:
                expr = expr & ~pl.col(column_name).is_null()
            row_expressions.append(expr)

        # Keep rows where at least one value is not unwanted
        df = df.filter(pl.any_horizontal(row_expressions))

        # Remove columns where all values are unwanted
        cols_to_keep = []
        for column_name in df.columns:
            # Check if column has at least one non-unwanted value
            if (df.select(
                (~pl.col(column_name).is_in(non_null_values) &
                 (not null_values or ~pl.col(column_name).is_null())).any()
            ).item()):
                cols_to_keep.append(column_name)

        if cols_to_keep:
            df = df.select(cols_to_keep)

        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
