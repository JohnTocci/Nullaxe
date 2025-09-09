import pandas as pd
import polars as pl
import re
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def _convert_to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    if not name:
        return name

    # Handle CamelCase and PascalCase by inserting underscores before capital letters
    name = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', name)

    # Handle spaces and hyphens
    name = re.sub(r'[-\s]+', '_', name)

    # Remove non-alphanumeric characters except underscores
    name = re.sub(r'[^\w]', '_', name)

    # Clean up multiple underscores
    name = re.sub(r'_+', '_', name)

    # Convert to lowercase and remove leading/trailing underscores
    name = name.lower().strip('_')

    # If name is empty after cleaning, return a fallback
    if not name:
        return "column"

    return name

def _convert_to_snake_case_for_dataframe(name: str) -> str:
    """Convert a string to snake_case for DataFrame operations (removes leading digits)."""
    result = _convert_to_snake_case(name)

    # Remove leading digits for DataFrame operations
    result = re.sub(r'^[\d_]+', '', result)

    # If result is empty after removing digits, use fallback
    if not result:
        return "column"

    return result

def _convert_to_camel_case(name: str) -> str:
    """Convert a string to camelCase."""
    if not name:
        return name

    # First normalize to snake_case, then convert
    snake_name = _convert_to_snake_case(name)
    # Split on underscores
    words = snake_name.split('_')
    # Filter out empty strings
    words = [word for word in words if word]
    if not words:
        return name
    # First word lowercase, rest title case
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def _convert_to_pascal_case(name: str) -> str:
    """Convert a string to PascalCase."""
    if not name:
        return name

    # First normalize to snake_case, then convert
    snake_name = _convert_to_snake_case(name)
    # Split on underscores
    words = snake_name.split('_')
    # Filter out empty strings
    words = [word for word in words if word]
    if not words:
        return name
    # All words title case
    return ''.join(word.capitalize() for word in words)

def _convert_to_kebab_case(name: str) -> str:
    """Convert a string to kebab-case."""
    if not name:
        return name

    # First convert to snake_case to normalize
    snake_name = _convert_to_snake_case(name)
    # Convert underscores to hyphens
    return snake_name.replace('_', '-')

def _convert_to_title_case(name: str) -> str:
    """Convert a string to Title Case."""
    if not name:
        return name

    # First convert to snake case to normalize, then split and capitalize
    snake_name = _convert_to_snake_case(name)
    # Split on underscores
    words = snake_name.split('_')
    # Clean and capitalize each word
    words = [word.capitalize() for word in words if word]
    return ' '.join(words)

def _convert_to_lower_case(name: str) -> str:
    """Convert a string to lowercase."""
    return name.lower()

def _screaming_snake_case(name: str) -> str:
    """Convert a string to SCREAMING_SNAKE_CASE."""
    return _convert_to_snake_case(name).upper()

def _apply_column_case(df: DataFrameType, case_func) -> DataFrameType:
    """
    Apply a case conversion function to all column names in the DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        df.columns = [case_func(col) for col in df.columns]
    elif isinstance(df, pl.DataFrame):
        df.columns = [case_func(col) for col in df.columns]
    else:
        raise TypeError("Input must be a pandas or polars DataFrame.")
    return df

def snakecase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to snake_case."""
    return _apply_column_case(df, _convert_to_snake_case_for_dataframe)

def camelcase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to camelCase."""
    def _camel_for_df(name):
        snake_name = _convert_to_snake_case_for_dataframe(name)
        words = snake_name.split('_')
        words = [word for word in words if word]
        if not words:
            return name
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    return _apply_column_case(df, _camel_for_df)

def pascalcase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to PascalCase."""
    def _pascal_for_df(name):
        snake_name = _convert_to_snake_case_for_dataframe(name)
        words = snake_name.split('_')
        words = [word for word in words if word]
        if not words:
            return name
        return ''.join(word.capitalize() for word in words)
    return _apply_column_case(df, _pascal_for_df)

def kebabcase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to kebab-case."""
    def _kebab_for_df(name):
        snake_name = _convert_to_snake_case_for_dataframe(name)
        return snake_name.replace('_', '-')
    return _apply_column_case(df, _kebab_for_df)

def titlecase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to Title Case."""
    def _title_for_df(name):
        snake_name = _convert_to_snake_case_for_dataframe(name)
        words = snake_name.split('_')
        words = [word.capitalize() for word in words if word]
        return ' '.join(words)
    return _apply_column_case(df, _title_for_df)

def lowercase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to lowercase."""
    return _apply_column_case(df, _convert_to_lower_case)

def screaming_snakecase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to SCREAMING_SNAKE_CASE."""
    return _apply_column_case(df, lambda name: _convert_to_snake_case_for_dataframe(name).upper())

def clean_column_names(df: DataFrameType, case: str = 'snake') -> DataFrameType:
    """
    Clean and standardize column names in the DataFrame to the specified case format.
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
        'lower': lowercase,
        'screaming_snake': screaming_snakecase,
        'SCREAMING_SNAKE_CASE': screaming_snakecase,
    }

    if case not in case_functions:
        raise ValueError(f"Unsupported case format: {case}")

    return case_functions[case](df)
