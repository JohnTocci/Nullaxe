import pandas as pd
import polars as pl
import numpy as np
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def handle_outliers(df: DataFrameType, method: str = 'iqr', threshold: float = 1.5,
                   action: str = 'cap', columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Handle outliers in numeric columns using various methods.

    Parameters:
    - method: 'iqr' or 'zscore'
    - threshold: IQR multiplier or z-score threshold
    - action: 'cap', 'remove', or 'flag'
    - columns: list of columns to process
    """
    if action == 'cap':
        return cap_outliers(df, method, threshold, columns)
    elif action == 'remove':
        return remove_outliers(df, method, threshold, columns)
    else:
        raise ValueError("Action must be 'cap' or 'remove'")

def cap_outliers(df: DataFrameType, method: str = 'iqr', threshold: float = 1.5,
                columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Cap outliers in numeric columns.
    """
    if isinstance(df, pd.DataFrame):
        result_df = df.copy()
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        if columns:
            numeric_cols = [col for col in numeric_cols if col in columns]

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            elif method == 'zscore':
                mean = result_df[col].mean()
                std = result_df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")

            result_df[col] = result_df[col].clip(lower_bound, upper_bound)

        return result_df

    elif isinstance(df, pl.DataFrame):
        result_df = df.clone()
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        if columns:
            numeric_cols = [col for col in numeric_cols if col in columns]

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")

            result_df = result_df.with_columns(
                pl.col(col).clip(lower_bound, upper_bound)
            )

        return result_df

    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")

def remove_outliers(df: DataFrameType, method: str = 'iqr', threshold: float = 1.5,
                   columns: Optional[List[str]] = None) -> DataFrameType:
    """
    Remove rows containing outliers in numeric columns.
    """
    if isinstance(df, pd.DataFrame):
        result_df = df.copy()
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        if columns:
            numeric_cols = [col for col in numeric_cols if col in columns]

        mask = pd.Series(True, index=result_df.index)

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                # Create mask for values that are NOT outliers (within bounds)
                col_mask = (result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)
            elif method == 'zscore':
                mean = result_df[col].mean()
                std = result_df[col].std(ddof=1)  # Use sample standard deviation
                if std == 0:  # Handle case where std is 0
                    continue
                # Calculate z-scores for the column
                z_scores = np.abs((result_df[col] - mean) / std)
                # Create mask for values within threshold (NOT outliers)
                col_mask = z_scores <= threshold
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")

            # Update the overall mask to keep only rows that are not outliers in any column
            mask = mask & col_mask

        return result_df[mask]

    elif isinstance(df, pl.DataFrame):
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        if columns:
            numeric_cols = [col for col in numeric_cols if col in columns]

        filters = []

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                # Filter for values that are NOT outliers (within bounds)
                filters.append((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    continue
                # Filter for values within z-score threshold
                filters.append((pl.col(col) - mean).abs() / std <= threshold)
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")

        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
            return df.filter(combined_filter)

        return df

    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")
