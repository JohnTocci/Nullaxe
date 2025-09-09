import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]


def handle_outliers(df: DataFrameType, method: str = 'zscore', factor: float = 3.0, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Handles outliers in the DataFrame by removing rows containing outliers.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'zscore'.
    factor (float): Threshold for identifying outliers. For 'zscore', it's the number of standard deviations.
                   For 'iqr', it's the multiplier for the interquartile range. Default is 3.0.
    subset (List[str], optional): List of column names to consider for outlier handling. Default is None (all numeric columns).

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
                # Skip columns with zero standard deviation
                if std == 0:
                    continue
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

        # If no valid conditions were created, return the original DataFrame
        if not conditions:
            return df

        # Combine all conditions using logical AND
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition & condition

        return df.filter(final_condition)

    raise TypeError("Input must be a pandas or polars DataFrame.")

def cap_outliers(df: DataFrameType, method: str = 'zscore', factor: float = 3.0, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Caps outliers in the DataFrame by replacing them with threshold values.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'zscore'.
    factor (float): Threshold for identifying outliers. For 'zscore', it's the number of standard deviations.
                   For 'iqr', it's the multiplier for the interquartile range. Default is 3.0.
    subset (List[str], optional): List of column names to consider for outlier handling. Default is None (all numeric columns).

    Returns:
    DataFrameType: DataFrame with outliers capped at threshold values.
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

        df_copy = df.copy()

        if method == 'zscore':
            for col in numeric_cols:
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                upper_bound = mean + (factor * std)
                lower_bound = mean - (factor * std)

                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

            return df_copy

        elif method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (factor * IQR)
                upper_bound = Q3 + (factor * IQR)

                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

            return df_copy

        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'.")

    elif isinstance(df, pl.DataFrame):
        # Determine which columns to process
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

        if subset:
            # Filter subset to only include numeric columns that exist in the DataFrame
            numeric_cols = [col for col in subset if col in numeric_cols]

        if not numeric_cols:
            return df  # Return original if no valid numeric columns in subset

        df_copy = df.clone()

        if method == 'zscore':
            for col in numeric_cols:
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                # Skip columns with zero standard deviation
                if std == 0:
                    continue
                upper_bound = mean + (factor * std)
                lower_bound = mean - (factor * std)

                df_copy = df_copy.with_column(
                    pl.col(col).clip(lower_bound, upper_bound)
                )

            return df_copy

        elif method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (factor * IQR)
                upper_bound = Q3 + (factor * IQR)

                df_copy = df_copy.with_column(
                    pl.col(col).clip(lower_bound, upper_bound)
                )

            return df_copy

        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'.")

    raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_outliers(df: DataFrameType, method: str = 'zscore', factor: float = 3.0, subset: Optional[List[str]] = None) -> DataFrameType:
    """
    Removes outliers from the DataFrame by dropping rows containing outliers.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'zscore'.
    factor (float): Threshold for identifying outliers. For 'zscore', it's the number of standard deviations.
                   For 'iqr', it's the multiplier for the interquartile range. Default is 3.0.
    subset (List[str], optional): List of column names to consider for outlier handling. Default is None (all numeric columns).

    Returns:
    DataFrameType: DataFrame with outlier rows removed.
    """
    return handle_outliers(df, method=method, factor=factor, subset=subset)
