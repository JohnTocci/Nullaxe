import pandas as pd
import polars as pl
from typing import Union, Optional, List, Callable, Dict, Any

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def impute_values(
    df: DataFrameType,
    strategy: str = 'mean',
    fill_value: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    custom_imputers: Optional[Dict[str, Callable[[pd.Series], Any]]] = None
) -> DataFrameType:
    """
    Imputes missing values in the DataFrame using specified strategies.

    Parameters:
    df (DataFrameType): Input DataFrame.
    strategy (str): Imputation strategy - 'mean', 'median', 'mode', or 'constant'.
    fill_value (Any): Value to use for 'constant' strategy.
    columns (List[str], optional): List of columns to impute. If None, all columns are considered.
    custom_imputers (Dict[str, Callable], optional): Custom imputation functions for specific columns.

    Returns:
    DataFrameType: DataFrame with imputed values.
    """
    if isinstance(df, pd.DataFrame):
        if columns is None:
            columns = df.columns.tolist()

        for col in columns:
            if col in df.columns:
                if custom_imputers and col in custom_imputers:
                    df[col] = df[col].fillna(custom_imputers[col](df[col]))
                else:
                    if strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == 'mode':
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif strategy == 'constant':
                        if fill_value is not None:
                            df[col] = df[col].fillna(fill_value)
                        else:
                            raise ValueError("fill_value must be provided for 'constant' strategy.")
                    else:
                        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'.")
        return df

    elif isinstance(df, pl.DataFrame):
        if columns is None:
            columns = df.columns

        for col in columns:
            if col in df.columns:
                if custom_imputers and col in custom_imputers:
                    imputed_value = custom_imputers[col](df[col].to_pandas())
                    df = df.with_column(
                        pl.when(pl.col(col).is_null())
                        .then(imputed_value)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                else:
                    if strategy == 'mean':
                        mean_value = df[col].mean()
                        df = df.with_column(
                            pl.when(pl.col(col).is_null())
                            .then(mean_value)
                            .otherwise(pl.col(col))
                            .alias(col)
                        )
                    elif strategy == 'median':
                        median_value = df[col].median()
                        df = df.with_column(
                            pl.when(pl.col(col).is_null())
                            .then(median_value)
                            .otherwise(pl.col(col))
                            .alias(col)
                        )
                    elif strategy == 'mode':
                        mode_value = df[col].mode()[0, 0] if not df[col].mode().is_empty() else None
                        df = df.with_column(
                            pl.when(pl.col(col).is_null())
                            .then(mode_value)
                            .otherwise(pl.col(col))
                            .alias(col)
                        )
                    elif strategy == 'constant':
                        if fill_value is not None:
                            df = df.with_column(
                                pl.when(pl.col(col).is_null())
                                .then(fill_value)
                                .otherwise(pl.col(col))
                                .alias(col)
                            )
                        else:
                            raise ValueError("fill_value must be provided for 'constant' strategy.")
                    else:
                        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'.")
        return df
    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")




