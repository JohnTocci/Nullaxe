import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add the src directory to the path to import nullaxe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._remove_duplicates import remove_duplicates
from nullaxe.functions._drop_single_value_columns import drop_single_value_columns
from nullaxe.functions._handle_outliers import handle_outliers, cap_outliers, remove_outliers
from nullaxe.functions._standardize_booleans import standardize_booleans
from nullaxe.functions._whitespace import remove_whitespace


class TestDataCleaningFunctions:
    """Test data cleaning and validation functions."""

    def test_remove_duplicates_pandas(self):
        """Test duplicate removal with pandas DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3, 1],
            'B': [4, 5, 5, 6, 4],
            'C': [7, 8, 8, 9, 7]
        })

        result = remove_duplicates(df)

        assert result.shape[0] == 3  # Only unique rows remain
        assert not result.duplicated().any()

    def test_remove_duplicates_polars(self):
        """Test duplicate removal with polars DataFrame."""
        df = pl.DataFrame({
            'A': [1, 2, 2, 3, 1],
            'B': [4, 5, 5, 6, 4],
            'C': [7, 8, 8, 9, 7]
        })

        result = remove_duplicates(df)

        assert result.shape[0] == 3  # Only unique rows remain

    def test_drop_single_value_columns_pandas(self):
        """Test dropping single value columns with pandas DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],           # Multiple values
            'B': [5, 5, 5, 5],           # Single value
            'C': [6, 7, 8, 9],           # Multiple values
            'D': ['same', 'same', 'same', 'same']  # Single value
        })

        result = drop_single_value_columns(df)

        assert result.shape[1] == 2  # Only columns A and C remain
        assert list(result.columns) == ['A', 'C']

    def test_drop_single_value_columns_polars(self):
        """Test dropping single value columns with polars DataFrame."""
        df = pl.DataFrame({
            'A': [1, 2, 3, 4],           # Multiple values
            'B': [5, 5, 5, 5],           # Single value
            'C': [6, 7, 8, 9]            # Multiple values
        })

        result = drop_single_value_columns(df)

        assert result.shape[1] == 2  # Only columns A and C remain
        assert 'A' in result.columns
        assert 'C' in result.columns
        assert 'B' not in result.columns

    def test_handle_outliers_pandas_iqr(self):
        """Test outlier handling with IQR method and pandas DataFrame."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'other': [10, 20, 30, 40, 50, 60]
        })

        result = handle_outliers(df, columns=['values'], method='iqr', action='remove')

        assert result.shape[0] < df.shape[0]  # Outlier row removed
        assert 100 not in result['values'].values

    def test_cap_outliers_pandas(self):
        """Test outlier capping with pandas DataFrame."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })

        result = cap_outliers(df, columns=['values'], method='iqr')

        assert result['values'].max() < 100  # Outlier capped
        assert result.shape[0] == df.shape[0]  # No rows removed

    def test_remove_outliers_pandas(self):
        """Test outlier removal with pandas DataFrame."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, 200],  # 100, 200 are outliers
            'other': [10, 20, 30, 40, 50, 60, 70]
        })

        result = remove_outliers(df, columns=['values'], method='zscore', threshold=0.5)

        assert result.shape[0] < df.shape[0]  # Outlier rows removed
        assert 100 not in result['values'].values
        assert 200 not in result['values'].values

    def test_standardize_booleans_pandas(self):
        """Test boolean standardization with pandas DataFrame."""
        df = pd.DataFrame({
            'bool_col': ['yes', 'no', 'true', 'false', 'Y', 'N', '1', '0'],
            'text_col': ['keep', 'this', 'as', 'is', 'text', 'data', 'here', 'now']
        })

        result = standardize_booleans(df, columns=['bool_col'])

        expected_values = [True, False, True, False, True, False, True, False]
        assert list(result['bool_col']) == expected_values
        assert result['text_col'].tolist() == df['text_col'].tolist()  # Unchanged

    def test_standardize_booleans_polars(self):
        """Test boolean standardization with polars DataFrame."""
        df = pl.DataFrame({
            'bool_col': ['yes', 'no', 'true', 'false', 'Y', 'N']
        })

        result = standardize_booleans(df, columns=['bool_col'])

        expected_values = [True, False, True, False, True, False]
        assert result['bool_col'].to_list() == expected_values

    def test_remove_whitespace_pandas(self):
        """Test whitespace removal with pandas DataFrame."""
        df = pd.DataFrame({
            'text_col': ['  leading', 'trailing  ', '  both  ', 'none', ''],
            'numeric_col': [1, 2, 3, 4, 5]
        })

        result = remove_whitespace(df)

        expected_values = ['leading', 'trailing', 'both', 'none', '']
        assert result['text_col'].tolist() == expected_values
        assert result['numeric_col'].tolist() == df['numeric_col'].tolist()  # Unchanged

    def test_remove_whitespace_polars(self):
        """Test whitespace removal with polars DataFrame."""
        df = pl.DataFrame({
            'text_col': ['  leading', 'trailing  ', '  both  ', 'none']
        })

        result = remove_whitespace(df)

        expected_values = ['leading', 'trailing', 'both', 'none']
        assert result['text_col'].to_list() == expected_values

    def test_outlier_methods_comparison(self):
        """Test different outlier detection methods."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is clear outlier
        })

        # Test IQR method
        result_iqr = handle_outliers(df, columns=['values'], method='iqr', action='remove')

        # Test Z-score method
        result_zscore = handle_outliers(df, columns=['values'], method='zscore', action='remove', threshold=2)

        # Both methods should identify and remove the outlier
        assert result_iqr.shape[0] < df.shape[0]
        assert result_zscore.shape[0] < df.shape[0]
        assert 100 not in result_iqr['values'].values
        assert 100 not in result_zscore['values'].values

    def test_boolean_variations(self):
        """Test various boolean representations."""
        df = pd.DataFrame({
            'bools': [
                'yes', 'no', 'YES', 'NO',
                'true', 'false', 'TRUE', 'FALSE',
                'y', 'n', 'Y', 'N',
                '1', '0', 'on', 'off'
            ]
        })

        result = standardize_booleans(df, columns=['bools'])

        expected = [
            True, False, True, False,    # yes/no variants
            True, False, True, False,    # true/false variants
            True, False, True, False,    # y/n variants
            True, False, True, False     # 1/0 and on/off variants
        ]

        assert result['bools'].tolist() == expected

    def test_edge_cases_empty_dataframe(self):
        """Test functions with empty DataFrames."""
        empty_df = pd.DataFrame()

        # Functions should handle empty DataFrames gracefully
        result_dup = remove_duplicates(empty_df)
        assert result_dup.shape == (0, 0)

        result_single = drop_single_value_columns(empty_df)
        assert result_single.shape == (0, 0)

        result_whitespace = remove_whitespace(empty_df)
        assert result_whitespace.shape == (0, 0)

    def test_invalid_input_types(self):
        """Test functions with invalid input types."""
        with pytest.raises(TypeError):
            remove_duplicates("not a dataframe")

        with pytest.raises(TypeError):
            drop_single_value_columns([1, 2, 3])

        with pytest.raises(TypeError):
            remove_whitespace({"not": "dataframe"})
