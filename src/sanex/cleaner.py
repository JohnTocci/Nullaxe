from .functions import (
    snakecase, camelcase, pascalcase,
    kebabcase, titlecase, lowercase,
    screaming_snakecase, clean_column_names,
    remove_duplicates, fill_missing, drop_missing,
    remove_whitespace, replace_text, drop_single_value_columns,
    handle_outliers, cap_outliers, remove_outliers,
    standardize_booleans, remove_unwanted_rows_and_cols,
    extract_and_clean_numeric, clean_numeric, extract_email,
    extract_with_regex, extract_phone_numbers, remove_punctuation,
    remove_special_characters, remove_emojis, remove_non_ascii,
    remove_non_alphanumeric, remove_non_numeric, remove_pii,
    remove_stopwords, flag_for_review, format_for_display)
import pandas as pd
import polars as pl
from typing import Union, List, Optional

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

    def standardize_booleans(self, true_values: list = None, false_values: list = None, columns: list = None):
        """
        Standardizes boolean-like values in the DataFrame to actual boolean types.

        Parameters:
        true_values (list): List of values to be considered as True. Default is ['yes', 'y', 'true', 't', '1'].
        false_values (list): List of values to be considered as False. Default is ['no', 'n', 'false', 'f', '0'].
        columns (list): List of column names to consider for boolean standardization. Default is None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = standardize_booleans(self._df, true_values=true_values, false_values=false_values, columns=columns)
        return self

    def remove_unwanted_rows_and_cols(self, unwanted_values: Optional[List[Union[str, int, float]]] = None):
        """
        Removes rows and columns that contain only unwanted values from the DataFrame.

        Parameters:
        unwanted_values (List[Union[str, int, float]], optional): List of values considered unwanted.
            Defaults to [None, '', 'NA', 'N/A', 'null', 'NULL', 'NaN'].

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_unwanted_rows_and_cols(self._df, unwanted_values=unwanted_values)
        return self

    def extract_and_clean_numeric(self, subset: Optional[List[str]] = None):
        """
        Extracts and cleans numeric data from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for numeric extraction.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = extract_and_clean_numeric(self._df, subset=subset)
        return self

    def clean_numeric(self, method: str = 'iqr', factor: float = 1.5, subset: list = None):
        """
        Cleans numeric columns in the DataFrame by extracting numeric values from strings
        and handling outliers.

        Parameters:
        method (str): The method to use for handling outliers. Default is 'iqr'.
                      Supported methods include 'iqr' (Interquartile Range) and 'zscore' (Z-Score).
        factor (float): The factor to use for determining outlier thresholds. Default is 1.5.
                        For 'iqr', this is the multiplier for the IQR. For 'zscore', this is the Z-Score threshold.
        subset (list): List of column names to consider for cleaning. Default is None (all numeric columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = clean_numeric(self._df, subset=subset)
        self._df = handle_outliers(self._df, method=method, factor=factor, subset=subset)
        return self

    def extract_email(self, subset: Optional[List[str]] = None):
        """
        Extracts email addresses from string entries in the DataFrame and places them in new columns.

        Parameters:
        subset (List[str], optional): List of column names to consider for email extraction.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = extract_email(self._df, subset=subset)
        return self

    def extract_with_regex(self, subset: Optional[List[str]] = None):
        """
        Extracts substrings matching a given regex pattern from specified columns in the DataFrame
        and places them in new columns.

        Parameters:
        subset (List[str], optional): List of column names to consider for extraction.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        if subset is None:
            # Handle different DataFrame types correctly
            if isinstance(self._df, pd.DataFrame):
                subset = list(self._df.columns)  # Use list constructor instead of tolist()
            else:  # polars DataFrame
                subset = self._df.columns  # Already a list in polars
        pattern = input("Enter the regex pattern to extract: ")
        self._df = extract_with_regex(self._df, pattern=pattern, subset=subset)
        return self

    def extract_phone_numbers(self, subset: Optional[List[str]] = None):
        """
        Extracts phone numbers from string entries in the DataFrame and places them in new columns.

        Parameters:
        subset (List[str], optional): List of column names to consider for phone number extraction.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        if subset is None:
            # Handle different DataFrame types correctly
            if isinstance(self._df, pd.DataFrame):
                subset = list(self._df.columns)  # Use list constructor instead of tolist()
            else:  # polars DataFrame
                subset = self._df.columns  # Already a list in polars
        self._df = extract_phone_numbers(self._df, subset=subset)
        return self  # Add missing return statement

    def remove_punctuation(self, subset: Optional[List[str]] = None):
        """
        Removes punctuation from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for punctuation removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_punctuation(self._df, subset=subset)
        return self

    def remove_special_characters(self, subset: Optional[List[str]] = None):
        """
        Removes special characters from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for special character removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_special_characters(self._df, subset=subset)
        return self

    def remove_emojis(self, subset: Optional[List[str]] = None):
        """
        Removes emojis from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for emoji removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_emojis(self._df, subset=subset)
        return self

    def remove_non_ascii(self, subset: Optional[List[str]] = None):
        """
        Removes non-ASCII characters from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for non-ASCII character removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_non_ascii(self._df, subset=subset)
        return self

    def remove_non_alphanumeric(self, subset: Optional[List[str]] = None):
        """
        Removes non-alphanumeric characters from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for non-alphanumeric character removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_non_alphanumeric(self._df, subset=subset)
        return self

    def remove_non_numeric(self, subset: Optional[List[str]] = None):
        """
        Removes non-numeric characters from string entries in the DataFrame.

        Parameters:
        subset (List[str], optional): List of column names to consider for non-numeric character removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_non_numeric(self._df, subset=subset)
        return self

    def remove_pii(self, subset: Optional[List[str]] = None):
        """
        Removes personally identifiable information (PII) from specified string columns in the DataFrame.
        PII patterns include email addresses, phone numbers, social security numbers, and URLs.

        Parameters:
        subset (List[str], optional): List of column names to consider for PII removal.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        if subset is None:
            # Handle different DataFrame types correctly
            if isinstance(self._df, pd.DataFrame):
                subset = list(self._df.columns)  # Use list constructor instead of tolist()
            else:  # polars DataFrame
                subset = self._df.columns  # Already a list in polars
        self._df = remove_pii(self._df, subset=subset)
        return self

    def remove_stopwords(self, subset: Optional[List[str]] = None, language: str = 'english'):
        """Removes stopwords from specified text columns.

        Parameters:
            subset: Columns to process (default all columns).
            language: Stopword language key (default 'english').
        """
        if subset is None:
            if isinstance(self._df, pd.DataFrame):
                subset = list(self._df.columns)
            else:
                subset = self._df.columns
        self._df = remove_stopwords(self._df, subset=subset, language=language)
        return self

    def flag_for_review(self, condition: str, subset: Optional[List[str]] = None):
        """
        Flags rows in the DataFrame for review based on a specified condition.

        Parameters:
        condition (str): The condition to evaluate for flagging rows.
                         This should be a valid expression that can be evaluated.
        subset (List[str], optional): List of column names to consider for flagging.
            Defaults to None (all columns).

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = flag_for_review(self._df, condition=condition, subset=subset)
        return self

    def format_for_display(self, rules: dict, column_case: Optional[str] = 'title'):
        """
        Applies various formatting rules to DataFrame columns for presentation.

        This function is designed to be the final step in a cleaning pipeline,
        converting a clean DataFrame into a human-readable format for reports,
        dashboards, or other displays. Note that this will often convert
        numeric columns to string/object types.

        Parameters:
        rules (Dict[str, Dict]): A dictionary where each key is a column name
            and the value is another dictionary specifying the formatting rule.
            The supported rule types are:
            - {'type': 'currency', 'symbol': '$', 'decimals': 2}
            - {'type': 'percentage', 'decimals': 1}
            - {'type': 'thousands'}
            - {'type': 'truncate', 'length': 50}
            - {'type': 'datetime', 'format': '%B %d, %Y'}
        column_case (Optional[str]): The desired case for the column headers in the
            final output. Currently supports 'title'. Defaults to 'title'.
            Set to None to leave column names as they are.

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = format_for_display(self._df, rules=rules, column_case=column_case)
        return self

    def to_df(self) -> DataFrameType:
        """
        Returns the final, cleaned DataFrame.
        """
        return self._df
