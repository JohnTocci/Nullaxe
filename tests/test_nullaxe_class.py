import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add the src directory to the path to import nullaxe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import nullaxe as nlx
from nullaxe.cleaner import Nullaxe


class TestNullaxeClass:
    """Test the main Nullaxe class functionality."""

    def test_nullaxe_init_pandas(self, sample_pandas_df):
        """Test Nullaxe initialization with pandas DataFrame."""
        nlx_instance = Nullaxe(sample_pandas_df)
        assert isinstance(nlx_instance._df, pd.DataFrame)
        assert nlx_instance._df.shape == sample_pandas_df.shape

    def test_nullaxe_init_polars(self, sample_polars_df):
        """Test Nullaxe initialization with polars DataFrame."""
        nlx_instance = Nullaxe(sample_polars_df)
        assert isinstance(nlx_instance._df, pl.DataFrame)
        assert nlx_instance._df.shape == sample_polars_df.shape

    def test_nullaxe_init_invalid_input(self):
        """Test Nullaxe initialization with invalid input."""
        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            Nullaxe("not a dataframe")

        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            Nullaxe([1, 2, 3])

    def test_method_chaining_pandas(self, sample_pandas_df):
        """Test method chaining with pandas DataFrame."""
        nlx_instance = Nullaxe(sample_pandas_df)
        result = nlx_instance.clean_column_names().remove_duplicates().fill_missing()
        assert isinstance(result, Nullaxe)
        assert isinstance(result._df, pd.DataFrame)

    def test_method_chaining_polars(self, sample_polars_df):
        """Test method chaining with polars DataFrame."""
        nlx_instance = Nullaxe(sample_polars_df)
        result = nlx_instance.clean_column_names().remove_duplicates().fill_missing()
        assert isinstance(result, Nullaxe)
        assert isinstance(result._df, pl.DataFrame)

    def test_clean_column_names_default(self, messy_column_names_df):
        """Test clean_column_names with default snake_case."""
        nlx_instance = Nullaxe(messy_column_names_df)
        result = nlx_instance.clean_column_names()

        expected_columns = [
            'first_name', 'last_name', 'email_address', 'phone_number',
            'some_weird_column', 'numeric', 'camel_case_column', 'screaming_snake_case'
        ]

        assert list(result._df.columns) == expected_columns

    def test_clean_column_names_camel(self, messy_column_names_df):
        """Test clean_column_names with camelCase."""
        nlx_instance = Nullaxe(messy_column_names_df)
        result = nlx_instance.clean_column_names(case='camel')

        expected_columns = [
            'firstName', 'lastName', 'emailAddress', 'phoneNumber',
            'someWeirdColumn', 'numeric', 'camelCaseColumn', 'screamingSnakeCase'
        ]

        assert list(result._df.columns) == expected_columns

    def test_case_conversion_methods(self, messy_column_names_df):
        """Test individual case conversion methods."""
        nlx_instance = Nullaxe(messy_column_names_df)

        # Test snakecase
        snake_result = nlx_instance.snakecase()
        assert 'first_name' in snake_result._df.columns
        assert 'camel_case_column' in snake_result._df.columns

        # Test camelcase
        nlx_instance2 = Nullaxe(messy_column_names_df)
        camel_result = nlx_instance2.camelcase()
        assert 'firstName' in camel_result._df.columns
        assert 'camelCaseColumn' in camel_result._df.columns

        # Test pascalcase
        nlx_instance3 = Nullaxe(messy_column_names_df)
        pascal_result = nlx_instance3.pascalcase()
        assert 'FirstName' in pascal_result._df.columns
        assert 'CamelCaseColumn' in pascal_result._df.columns

    def test_missing_data_methods(self, sample_pandas_df):
        """Test missing data handling methods."""
        nlx_instance = Nullaxe(sample_pandas_df)
        original_shape = nlx_instance._df.shape

        # Test fill_missing
        filled = nlx_instance.fill_missing(value='FILLED')
        assert filled._df.isna().sum().sum() == 0  # No missing values after filling

        # Test drop_missing
        nlx_instance2 = Nullaxe(sample_pandas_df)
        dropped = nlx_instance2.drop_missing()
        assert dropped._df.shape[0] <= original_shape[0]  # Fewer or equal rows

    def test_remove_duplicates(self):
        """Test remove_duplicates method."""
        df_with_dups = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [4, 5, 5, 6],
            'C': [7, 8, 8, 9]
        })

        nlx_instance = Nullaxe(df_with_dups)
        result = nlx_instance.remove_duplicates()

        assert result._df.shape[0] == 3  # One duplicate row removed
        assert not result._df.duplicated().any()
