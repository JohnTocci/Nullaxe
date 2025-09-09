import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add the src directory to the path to import sanex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import sanex
from sanex.cleaner import Sanex


class TestSanexClass:
    """Test the main Sanex class functionality."""

    def test_sanex_init_pandas(self, sample_pandas_df):
        """Test Sanex initialization with pandas DataFrame."""
        sx = Sanex(sample_pandas_df)
        assert isinstance(sx._df, pd.DataFrame)
        assert sx._df.shape == sample_pandas_df.shape

    def test_sanex_init_polars(self, sample_polars_df):
        """Test Sanex initialization with polars DataFrame."""
        sx = Sanex(sample_polars_df)
        assert isinstance(sx._df, pl.DataFrame)
        assert sx._df.shape == sample_polars_df.shape

    def test_sanex_init_invalid_input(self):
        """Test Sanex initialization with invalid input."""
        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            Sanex("not a dataframe")

        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            Sanex([1, 2, 3])

    def test_method_chaining_pandas(self, sample_pandas_df):
        """Test method chaining with pandas DataFrame."""
        sx = Sanex(sample_pandas_df)
        result = sx.clean_column_names().remove_duplicates().fill_missing()
        assert isinstance(result, Sanex)
        assert isinstance(result._df, pd.DataFrame)

    def test_method_chaining_polars(self, sample_polars_df):
        """Test method chaining with polars DataFrame."""
        sx = Sanex(sample_polars_df)
        result = sx.clean_column_names().remove_duplicates().fill_missing()
        assert isinstance(result, Sanex)
        assert isinstance(result._df, pl.DataFrame)

    def test_clean_column_names_default(self, messy_column_names_df):
        """Test clean_column_names with default snake_case."""
        sx = Sanex(messy_column_names_df)
        result = sx.clean_column_names()

        expected_columns = [
            'first_name', 'last_name', 'email_address', 'phone_number',
            'some_weird_column', 'numeric', 'camel_case_column', 'screaming_snake_case'
        ]

        assert list(result._df.columns) == expected_columns

    def test_clean_column_names_camel(self, messy_column_names_df):
        """Test clean_column_names with camelCase."""
        sx = Sanex(messy_column_names_df)
        result = sx.clean_column_names(case='camel')

        expected_columns = [
            'firstName', 'lastName', 'emailAddress', 'phoneNumber',
            'someWeirdColumn', 'numeric', 'camelCaseColumn', 'screamingSnakeCase'
        ]

        assert list(result._df.columns) == expected_columns

    def test_case_conversion_methods(self, messy_column_names_df):
        """Test individual case conversion methods."""
        sx = Sanex(messy_column_names_df)

        # Test snakecase
        snake_result = sx.snakecase()
        assert 'first_name' in snake_result._df.columns
        assert 'camel_case_column' in snake_result._df.columns

        # Test camelcase
        sx2 = Sanex(messy_column_names_df)
        camel_result = sx2.camelcase()
        assert 'firstName' in camel_result._df.columns
        assert 'camelCaseColumn' in camel_result._df.columns

        # Test pascalcase
        sx3 = Sanex(messy_column_names_df)
        pascal_result = sx3.pascalcase()
        assert 'FirstName' in pascal_result._df.columns
        assert 'CamelCaseColumn' in pascal_result._df.columns

    def test_missing_data_methods(self, sample_pandas_df):
        """Test missing data handling methods."""
        sx = Sanex(sample_pandas_df)
        original_shape = sx._df.shape

        # Test fill_missing
        filled = sx.fill_missing(value='FILLED')
        assert filled._df.isna().sum().sum() == 0  # No missing values after filling

        # Test drop_missing
        sx2 = Sanex(sample_pandas_df)
        dropped = sx2.drop_missing()
        assert dropped._df.shape[0] <= original_shape[0]  # Fewer or equal rows

    def test_remove_duplicates(self):
        """Test remove_duplicates method."""
        # Create DataFrame with duplicates
        df_with_dups = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [4, 5, 5, 6],
            'C': [7, 8, 8, 9]
        })

        sx = Sanex(df_with_dups)
        result = sx.remove_duplicates()

        assert result._df.shape[0] == 3  # One duplicate row removed
        assert not result._df.duplicated().any()
