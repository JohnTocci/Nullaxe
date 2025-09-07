import pandas as pd
import polars as pl
import re
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def _convert_to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()  # CamelCase to snake_case
    name = re.sub(r'\s+', '_', name)  # Spaces to underscores
    name = re.sub(r'\W', '', name)  # Remove non-alphanumeric characters
    name = re.sub(r'__+', '_', name)  # Replace multiple underscores with single underscore
    return name.strip('_')  # Remove leading/trailing underscores

def _convert_to_camel_case(name: str) -> str:
    """Convert a string to camelCase."""
    name = re.sub(r'[_\s]+', ' ', name).title().replace(' ', '')  # snake_case or spaces to CamelCase
    return name[0].lower() + name[1:] if name else name  # Lowercase first letter

def _convert_to_pascal_case(name: str) -> str:
    """Convert a string to PascalCase."""
    name = re.sub(r'[_\s]+', ' ', name).title().replace(' ', '')  # snake_case or spaces to PascalCase
    return name  # First letter is already uppercase

def _convert_to_kebab_case(name: str) -> str:
    """Convert a string to kebab-case."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()  # CamelCase to kebab-case
    name = re.sub(r'\s+', '-', name)  # Spaces to hyphens
    name = re.sub(r'[^\w-]', '', name)  # Remove non-alphanumeric characters except hyphen
    name = re.sub(r'--+', '-', name)  # Replace multiple hyphens with single hyphen
    return name.strip('-')  # Remove leading/trailing hyphens

def _convert_to_title_case(name: str) -> str:
    """Convert a string to Title Case."""
    name = re.sub(r'[_\s]+', ' ', name).title()  # snake_case or spaces to Title Case
    return name.strip()  # Remove leading/trailing spaces

def _convert_to_upper_case(name: str) -> str:
    """Convert a string to UPPER_CASE."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).upper()  # CamelCase to UPPER_CASE
    name = re.sub(r'\s+', '_', name)  # Spaces to underscores
    name = re.sub(r'\W', '', name)  # Remove non-alphanumeric characters
    name = re.sub(r'__+', '_', name)  # Replace multiple underscores with single underscore
    return name.strip('_')  # Remove leading/trailing underscores

def _convert_to_lower_case(name: str) -> str:
    """Convert a string to LOWER_CASE."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()  # CamelCase to lower_case
    name = re.sub(r'\s+', '_', name)  # Spaces to underscores
    name = re.sub(r'\W', '', name)  # Remove non-alphanumeric characters
    name = re.sub(r'__+', '_', name)  # Replace multiple underscores with single underscore
    return name.strip('_')  # Remove leading/trailing underscores

def _screaming_snake_case(name: str) -> str:
    """Convert a string to SCREAMING_SNAKE_CASE."""
    return _convert_to_snake_case(name).upper()

def _apply_column_case(df: DataFrameType, case_func) -> DataFrameType:
    """
    Apply a case conversion function to all column names in the DataFrame.

    Parameters:
    df (DataFrameType): Input DataFrame (pandas or polars).
    case_func (Callable[[str], str]): Function that converts a single column name.

    Returns:
    DataFrameType: DataFrame with converted column names.
    """
    if isinstance(df, pd.DataFrame):
        df.columns = [case_func(col) for col in df.columns]
    elif isinstance(df, pl.DataFrame):
        df.columns = [case_func(col) for col in df.columns]
    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")

    return df

def snakecase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to snake_case.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in snake_case.
    """
    return _apply_column_case(df, _convert_to_snake_case)

def camelcase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to camelCase.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in camelCase.
    """
    return _apply_column_case(df, _convert_to_camel_case)

def pascalcase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to PascalCase.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in PascalCase.
    """
    return _apply_column_case(df, _convert_to_pascal_case)

def kebabcase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to kebab-case.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in kebab-case.
    """
    return _apply_column_case(df, _convert_to_kebab_case)

def titlecase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to Title Case.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in Title Case.
    """
    return _apply_column_case(df, _convert_to_title_case)

def uppercase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to UPPER_CASE.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in UPPER_CASE.
    """
    return _apply_column_case(df, _convert_to_upper_case)

def lowercase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to LOWER_CASE.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in LOWER_CASE.
    """
    return _apply_column_case(df, _convert_to_lower_case)

def screaming_snakecase(df: DataFrameType) -> DataFrameType:
    """
    Convert all column names in the DataFrame to SCREAMING_SNAKE_CASE.

    Parameters:
    df (DataFrameType): Input DataFrame.

    Returns:
    DataFrameType: DataFrame with column names in SCREAMING_SNAKE_CASE.
    """
    return _apply_column_case(df, _screaming_snake_case)

def clean_column_names(df: DataFrameType, case: str = 'snake') -> DataFrameType:
    """
    Clean and standardize column names in the DataFrame to the specified case format.

    Parameters:
    df (DataFrameType): Input DataFrame.
    case (str): Desired case format. Supported options include both short and stylistic forms:
                'snake', 'snake_case', 'camel', 'camelCase', 'pascal', 'PascalCase',
                'kebab', 'kebab-case', 'title', 'Title Case', 'upper', 'UPPER_CASE',
                'lower', 'lower_case', 'screaming_snake', 'SCREAMING_SNAKE_CASE'.
                Default is 'snake'.

    Returns:
    DataFrameType: DataFrame with cleaned column names.
    """
    case_functions = {
        'snake': snakecase,
        'snake_case': snakecase,
        'camel': camelcase,
        'camelCase': camelcase,
        'pascal': pascalcase,
        'PascalCase': pascalcase,
        'kebab': kebabcase,
        'kebab-case': kebabcase,
        'title': titlecase,
        'Title Case': titlecase,
        'upper': uppercase,
        'UPPER_CASE': uppercase,
        'lower': lowercase,
        'lower_case': lowercase,
        'screaming_snake': screaming_snakecase,
        'SCREAMING_SNAKE_CASE': screaming_snakecase,
    }

    if case not in case_functions:
        raise ValueError(f"Invalid case option '{case}'. Valid options are: {list(case_functions.keys())}")

    return case_functions[case](df)
