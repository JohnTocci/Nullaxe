import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def handle_outliers(df: DataFrameType, method: str = 'iqr', columns: Optional[List[str]] = None, action: str = 'remove', threshold: float = 3.0) -> DataFrameType:
    """
    Handles outliers in the DataFrame by removing rows containing outliers.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'iqr'.
    columns (List[str], optional): List of column names to consider for outlier handling. Default is None (all numeric columns).
    action (str): Action to take on outliers ('remove', 'cap'). Default is 'remove'.
    threshold (float): Threshold for outlier detection. Default is 3.0.

    Returns:
    DataFrameType: DataFrame with outliers handled.
    """
    if isinstance(df, pd.DataFrame):
        # Determine which columns to process
        if columns:
            numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return df
        else:
            numeric_cols = df.select_dtypes(include='number').columns

        if method == 'iqr':
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            keep_rows = ((df[numeric_cols] >= lower_bound) & (df[numeric_cols] <= upper_bound)).all(axis=1)
            return df[keep_rows]

        elif method == 'zscore':
            z_scores = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
            keep_rows = (z_scores.abs() <= threshold).all(axis=1)
            return df[keep_rows]

        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'.")

    elif isinstance(df, pl.DataFrame):
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]
        if not numeric_cols:
            return df

        conditions = []
        if method == 'iqr':
            for col_name in numeric_cols:
                Q1 = df[col_name].quantile(0.25)
                Q3 = df[col_name].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                condition = (df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)
                conditions.append(condition)

        elif method == 'zscore':
            for col_name in numeric_cols:
                mean = df[col_name].mean()
                std = df[col_name].std()
                if std == 0:
                    continue
                condition = ((df[col_name] - mean) / std).abs() <= threshold
                conditions.append(condition)

        if not conditions:
            return df

        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition & condition

        return df.filter(final_condition)

    raise TypeError("Input must be a pandas or polars DataFrame.")

def cap_outliers(df: DataFrameType, method: str = 'iqr', columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Caps outliers in the DataFrame by replacing them with threshold values.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'iqr'.
    columns (List[str], optional): List of column names to consider for outlier handling. Default is None (all numeric columns).

    Returns:
    DataFrameType: DataFrame with outliers capped at threshold values.
    """
    if isinstance(df, pd.DataFrame):
        if columns:
            numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return df
        else:
            numeric_cols = df.select_dtypes(include='number').columns

        df_copy = df.copy()

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'zscore':
            for col in numeric_cols:
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                if std == 0:
                    continue
                upper_bound = mean + (3 * std)
                lower_bound = mean - (3 * std)
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

        return df_copy

    elif isinstance(df, pl.DataFrame):
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]
        if not numeric_cols:
            return df

        df_copy = df.clone()

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                df_copy = df_copy.with_columns(
                    pl.col(col).clip(lower_bound, upper_bound)
                )

        elif method == 'zscore':
            for col in numeric_cols:
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                if std == 0:
                    continue
                upper_bound = mean + (3 * std)
                lower_bound = mean - (3 * std)
                df_copy = df_copy.with_columns(
                    pl.col(col).clip(lower_bound, upper_bound)
                )

        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_outliers(df: DataFrameType, method: str = 'zscore', threshold: float = 2, columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Removes outliers from the DataFrame by dropping rows containing outliers.

    Parameters:
    df (DataFrameType): Input DataFrame.
    method (str): Method to identify outliers. Options are 'zscore' or 'iqr'. Default is 'zscore'.
    threshold (float): Threshold for identifying outliers. Default is 2.
    columns (List[str], optional): List of column names to consider for outlier handling. Default is None (all numeric columns).

    Returns:
    DataFrameType: DataFrame with outlier rows removed.
    """
    return handle_outliers(df, method=method, columns=columns, threshold=threshold)
