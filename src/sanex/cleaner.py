from .functions import (
    snakecase, camelcase, pascalcase,
    kebabcase, titlecase, lowercase,
    screaming_snakecase, clean_column_names,
    remove_duplicates, fill_missing, drop_missing,
    remove_whitespace, replace_text, drop_single_value_columns,
    handle_outliers, cap_outliers, remove_outliers,
    standardize_booleans)
import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

class Sanex:
    def __init__(self, df):
        if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError("Input must be a pandas or polars DataFrame.")
        self._df = df


    def clean_column_names(self, case: str = 'snake'):
                """
                Cleans the column names of the DataFrame.

                Args:
                    case (str): The desired case format for the column names.
                                Defaults to 'snake'. Supported formats include:
                                'snake', 'camel', 'pascal', 'kebab', 'title',
                                'lower', and 'screaming_snake'.

                Returns:
                    Sanex: The instance of the class to allow method chaining.

                This is a chainable method.
                """
                self._df = clean_column_names(self._df, case=case)
                return self

    def remove_duplicates(self):
        """
        Removes duplicate rows and columns from the DataFrame.

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_duplicates(self._df)
        return self

    def snakecase(self):
        """
        Converts all column names in the DataFrame to snake_case.

        This is a chainable method.
        """
        self._df = snakecase(self._df)
        return self

    def camelcase(self):
        """
        Converts all column names in the DataFrame to camelCase.

        This is a chainable method.
        """
        self._df = camelcase(self._df)
        return self

    def pascalcase(self):
        """
        Converts all column names in the DataFrame to PascalCase.

        This is a chainable method.
        """
        self._df = pascalcase(self._df)
        return self

    def kebabcase(self):
        """
        Converts all column names in the DataFrame to kebab-case.

        This is a chainable method.
        """
        self._df = kebabcase(self._df)
        return self

    def titlecase(self):
        """
        Converts all column names in the DataFrame to Title Case.

        This is a chainable method.
        """
        self._df = titlecase(self._df)
        return self

    def lowercase(self):
        """
        Converts all column names in the DataFrame to lowercase.

        This is a chainable method.
        """
        self._df = lowercase(self._df)
        return self

    def screaming_snakecase(self):
        """
        Converts all column names in the DataFrame to SCREAMING_SNAKE_CASE.

        This is a chainable method.
        """
        self._df = screaming_snakecase(self._df)
        return self

    def fill_missing(self, value: Union[int, float, str] = 0, subset: list = None):
        """
        Fills missing values in the DataFrame with a specified value.

        Parameters:
        value (Union[int, float, str]): The value to replace missing values with. Default is 0.
        subset (list): List of column names to consider for filling. Default is None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = fill_missing(self._df, value=value, subset=subset)
        return self

    def drop_missing(self, how: str = 'any', thresh: int = None, subset: list = None, axis: str = 'rows'):
        """
        Drops rows or columns with missing values from the DataFrame.

        Parameters:
        how (str): 'any' to drop if any NA values are present, 'all' to drop if all values are NA. Default is 'any'.
        thresh (int): Require that many non-NA values to avoid dropping. Default is None.
        subset (list): List of column names to consider for dropping. Default is None (all columns).
        axis (str): 'rows' to drop rows, 'columns' to drop columns. Default is 'rows'.

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = drop_missing(self._df, how=how, thresh=thresh, subset=subset, axis=axis)
        return self

    def remove_whitespace(self):
        """
        Removes leading and trailing whitespace from string entries in the DataFrame.

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_whitespace(self._df)
        return self

    def replace_text(self, to_replace: str, value: str, subset: list = None):
        """
        Replaces occurrences of a specified substring with another substring in the DataFrame.

        Parameters:
        to_replace (str): The substring to be replaced.
        value (str): The substring to replace with.
        subset (list): List of column names to consider for replacement. Default is None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = replace_text(self._df, to_replace=to_replace, value=value, subset=subset)
        return self

    def drop_single_value_columns(self):
        """
        Drops columns that contain only a single unique value from the DataFrame.

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = drop_single_value_columns(self._df)
        return self

    def handle_outliers(self, method: str = 'iqr', factor: float = 1.5, subset: list = None):
        """
        Handles outliers in the DataFrame using the specified method.

        Parameters:
        method (str): The method to use for handling outliers. Default is 'iqr'.
                      Supported methods include 'iqr' (Interquartile Range) and 'zscore' (Z-Score).
        factor (float): The factor to use for determining outlier thresholds. Default is 1.5.
                        For 'iqr', this is the multiplier for the IQR. For 'zscore', this is the Z-Score threshold.
        subset (list): List of column names to consider for outlier handling. Default is None (all numeric columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = handle_outliers(self._df, method=method, factor=factor, subset=subset)
        return self

    def cap_outliers(self, method: str = 'iqr', factor: float = 1.5, subset: list = None):
        """
        Caps outliers in the DataFrame using the specified method.

        Parameters:
        method (str): The method to use for capping outliers. Default is 'iqr'.
                      Supported methods include 'iqr' (Interquartile Range) and 'zscore' (Z-Score).
        factor (float): The factor to use for determining outlier thresholds. Default is 1.5.
                        For 'iqr', this is the multiplier for the IQR. For 'zscore', this is the Z-Score threshold.
        subset (list): List of column names to consider for outlier capping. Default is None (all numeric columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = cap_outliers(self._df, method=method, factor=factor, subset=subset)
        return self

    def remove_outliers(self, method: str = 'iqr', factor: float = 1.5, subset: list = None):
        """
        Removes outliers from the DataFrame using the specified method.

        Parameters:
        method (str): The method to use for removing outliers. Default is 'iqr'.
                      Supported methods include 'iqr' (Interquartile Range) and 'zscore' (Z-Score).
        factor (float): The factor to use for determining outlier thresholds. Default is 1.5.
                        For 'iqr', this is the multiplier for the IQR. For 'zscore', this is the Z-Score threshold.
        subset (list): List of column names to consider for outlier removal. Default is None (all numeric columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_outliers(self._df, method=method, factor=factor, subset=subset)
        return self

    def standardize_booleans(self, true_values: list = None, false_values: list = None, subset: list = None):
        """
        Standardizes boolean-like values in the DataFrame to actual boolean types.

        Parameters:
        true_values (list): List of values to be considered as True. Default is ['yes', 'y', 'true', 't', '1'].
        false_values (list): List of values to be considered as False. Default is ['no', 'n', 'false', 'f', '0'].
        subset (list): List of column names to consider for boolean standardization. Default is None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = standardize_booleans(self._df, true_values=true_values, false_values=false_values, subset=subset)
        return self

    def to_df(self) -> DataFrameType:
        """
        Returns the final, cleaned DataFrame.
        """
        return self._df
