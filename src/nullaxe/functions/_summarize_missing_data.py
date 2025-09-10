import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def missing_data_summary(df: DataFrameType) -> DataFrameType:
    """
    Generates a summary of missing data in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame summarizing missing data for each column.
    """
    if isinstance(df, pd.DataFrame):
        total_missing = df.isnull().sum()
        percent_missing = (total_missing / len(df)) * 100
        summary_df = pd.DataFrame({
            'column_name': df.columns,
            'total_missing': total_missing,
            'percent_missing': percent_missing
        })
        return summary_df

    elif isinstance(df, pl.DataFrame):
        total_missing = {col: df[col].is_null().sum() for col in df.columns}
        percent_missing = {col: (total / len(df)) * 100 for col, total in total_missing.items()}
        summary_df = pl.DataFrame({
            'column_name': list(df.columns),
            'total_missing': list(total_missing.values()),
            'percent_missing': list(percent_missing.values())
        })
        return summary_df

    raise TypeError("Input must be a pandas or polars DataFrame.")
