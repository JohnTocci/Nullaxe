import pandas as pd
import polars as pl
import re
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def _convert_to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    # Remove leading digits and non-alphabetic characters
    name = re.sub(r'^[\d\W]+', '', name)

    # Handle CamelCase and PascalCase by inserting underscores before capital letters
    name = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', name)

    # Handle spaces and hyphens
    name = re.sub(r'[-\s]+', '_', name)

    # Remove non-alphanumeric characters except underscores
    name = re.sub(r'[^\w]', '_', name)

    # Clean up multiple underscores
    name = re.sub(r'_+', '_', name)

    # Convert to lowercase and remove leading/trailing underscores
    return name.lower().strip('_')

def _convert_to_camel_case(name: str) -> str:
    """Convert a string to camelCase."""
    # Remove leading digits and non-alphabetic characters
    name = re.sub(r'^[\d\W]+', '', name)

    # Split on underscores, hyphens, and spaces
    words = re.split(r'[-_\s]+', name.lower())
    # Filter out empty strings and clean each word
    words = [re.sub(r'[^\w]', '', word) for word in words if word and word.isalpha()]
    if not words:
        return name
    # First word lowercase, rest title case
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def _convert_to_pascal_case(name: str) -> str:
    """Convert a string to PascalCase."""
    # Remove leading digits and non-alphabetic characters
    name = re.sub(r'^[\d\W]+', '', name)

    # Split on underscores, hyphens, and spaces
    words = re.split(r'[-_\s]+', name.lower())
    # Filter out empty strings and clean each word
    words = [re.sub(r'[^\w]', '', word) for word in words if word and word.isalpha()]
    if not words:
        return name
    # All words title case
    return ''.join(word.capitalize() for word in words)

def _convert_to_kebab_case(name: str) -> str:
    """Convert a string to kebab-case."""
    # If it's already snake_case or SCREAMING_SNAKE_CASE, convert underscores to hyphens
    if '_' in name and not re.search(r'[A-Z][a-z]', name):
        return name.lower().replace('_', '-')

    # Handle CamelCase and PascalCase
    name = re.sub(r'(?<!^)(?=[A-Z][a-z])', '-', name)
    # Handle spaces
    name = re.sub(r'\s+', '-', name)
    # Remove non-alphanumeric characters except hyphens
    name = re.sub(r'[^\w-]', '-', name)
    # Clean up multiple hyphens and convert to lowercase
    name = re.sub(r'-+', '-', name).lower()
    # Remove leading/trailing hyphens
    return name.strip('-')

def _convert_to_title_case(name: str) -> str:
    """Convert a string to Title Case."""
    # Split on underscores, hyphens
    words = re.split(r'[-_]+', name)
    # Clean and capitalize each word
    words = [re.sub(r'[^\w]', '', word).capitalize() for word in words if word]
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
    return _apply_column_case(df, _convert_to_snake_case)

def camelcase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to camelCase."""
    return _apply_column_case(df, _convert_to_camel_case)

def pascalcase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to PascalCase."""
    return _apply_column_case(df, _convert_to_pascal_case)

def kebabcase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to kebab-case."""
    return _apply_column_case(df, _convert_to_kebab_case)

def titlecase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to Title Case."""
    return _apply_column_case(df, _convert_to_title_case)

def lowercase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to lowercase."""
    return _apply_column_case(df, _convert_to_lower_case)

def screaming_snakecase(df: DataFrameType) -> DataFrameType:
    """Convert all column names in the DataFrame to SCREAMING_SNAKE_CASE."""
    return _apply_column_case(df, _screaming_snake_case)

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
