import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]


def handle_outliers(df: DataFrameType, method: str = 'zscore', factor: float = 3.0, subset: list = None) -> DataFrameType:
    """
    Handles outliers in the DataFrame by removing rows containing outliers.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'zscore'.
    factor (float): Threshold for identifying outliers. For 'zscore', it's the number of standard deviations.
                   For 'iqr', it's the multiplier for the interquartile range. Default is 3.0.
    subset (list): List of column names to consider for outlier handling. Default is None (all numeric columns).

    Returns:
    DataFrameType: DataFrame with outlier rows removed.
    """
    if isinstance(df, pd.DataFrame):
        # Determine which columns to process
        if subset:
            # Filter subset to only include numeric columns that exist in the DataFrame
            numeric_cols = [col for col in subset if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                # Return the original DataFrame if no valid numeric columns were found
                return df
        else:
            numeric_cols = df.select_dtypes(include='number').columns

        if method == 'zscore':
            # Calculate Z-scores for all numeric columns at once
            z_scores = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
            # A row is kept if all its z-scores are within the threshold
            keep_rows = (z_scores.abs() <= factor).all(axis=1)
            return df[keep_rows]

        elif method == 'iqr':
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (factor * IQR)
            upper_bound = Q3 + (factor * IQR)

            # A row is kept if its values in all numeric columns are within the bounds
            keep_rows = ((df[numeric_cols] >= lower_bound) & (df[numeric_cols] <= upper_bound)).all(axis=1)
            return df[keep_rows]

        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'.")

    elif isinstance(df, pl.DataFrame):
        # Use Polars selectors to get numeric columns
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

        if subset:
            # Filter subset to only include numeric columns that exist in the DataFrame
            numeric_cols = [col for col in subset if col in numeric_cols]

        if not numeric_cols:
            return df  # Return original if no valid numeric columns in subset

        conditions = []
        if method == 'zscore':
            for col_name in numeric_cols:
                mean = df[col_name].mean()
                std = df[col_name].std()
                condition = ((df[col_name] - mean) / std).abs() <= factor
                conditions.append(condition)

        elif method == 'iqr':
            for col_name in numeric_cols:
                Q1 = df[col_name].quantile(0.25)
                Q3 = df[col_name].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (factor * IQR)
                upper_bound = Q3 + (factor * IQR)
                condition = (df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)
                conditions.append(condition)

        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'.")

        # Combine all conditions: a row is kept if it meets the condition for all numeric columns.
        # .is_null() is included to keep rows with missing values in that column.
        final_condition = pl.all_horizontal([(c | pl.col(c.meta.output_name()).is_null()) for c in conditions])
        return df.filter(final_condition)

    raise TypeError("Input must be a pandas or polars DataFrame.")