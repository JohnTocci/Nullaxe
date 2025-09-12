import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def remove_html(df: DataFrameType, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Removes HTML tags from string entries in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame.
    subset (List[str], optional): List of column names to consider for HTML removal.
        Defaults to None (all string columns).

    Returns:
    DataFrameType: DataFrame with HTML tags removed from specified columns.
    """
    html_tag_regex = r'<[^>]+>'  # Regex pattern to match HTML tags

    if isinstance(df, pd.DataFrame):
        if subset is None:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
        else:
            str_cols = [col for col in subset if col in df.columns and df[col].dtype in ['object', 'string']]

        for col in str_cols:
            df[col] = df[col].str.replace(html_tag_regex, '', regex=True)
        return df

    elif isinstance(df, pl.DataFrame):
        if subset is None:
            columns_to_process = [col for col in df.columns if df[col].dtype == pl.String]
        else:
            columns_to_process = [col for col in subset if col in df.columns and df[col].dtype == pl.String]

        for col in columns_to_process:
            df = df.with_columns(
                pl.col(col)
                .str.replace_all(html_tag_regex, '')  # Remove HTML tags
                .alias(col)
            )
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")