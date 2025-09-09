import pytest
import pandas as pd
import polars as pl
import numpy as np
import sys
import os

# Add the src directory to the path to import sanex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanex.functions._missing_data import fill_missing, drop_missing


class TestMissingDataFunctions:
    """Test missing data handling functions."""

    def test_fill_missing_pandas_default(self):
        """Test fill_missing with pandas DataFrame using default value."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ['a', 'b', None, 'd']
        })

        result = fill_missing(df)

        assert result.isna().sum().sum() == 0  # No missing values
        assert result.loc[2, 'A'] == 0  # Default fill value
        assert result.loc[0, 'B'] == 0
        assert result.loc[2, 'C'] == 0

    def test_fill_missing_pandas_custom_value(self):
        """Test fill_missing with custom fill value."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4]
        })

        result = fill_missing(df, value='MISSING')

        assert result.loc[2, 'A'] == 'MISSING'
        assert result.loc[0, 'B'] == 'MISSING'
        assert result.isna().sum().sum() == 0

    def test_fill_missing_pandas_subset(self):
        """Test fill_missing with column subset."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ['a', None, 'c', 'd']
        })

        result = fill_missing(df, value='FILLED', subset=['A', 'C'])

        assert result.loc[2, 'A'] == 'FILLED'
        assert result.loc[1, 'C'] == 'FILLED'
        assert pd.isna(result.loc[0, 'B'])  # B column not filled

    def test_fill_missing_polars_default(self):
        """Test fill_missing with polars DataFrame using default value."""
        df = pl.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ['a', 'b', None, 'd']
        })

        result = fill_missing(df)

        assert result.null_count().sum_horizontal().sum() == 0  # No null values

    def test_fill_missing_polars_custom_value(self):
        """Test fill_missing with polars and custom value."""
        df = pl.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4]
        })

        result = fill_missing(df, value=999)

        assert result[2, 'A'] == 999
        assert result[0, 'B'] == 999
        assert result.null_count().sum_horizontal().sum() == 0

    def test_drop_missing_pandas_rows_any(self):
        """Test drop_missing with pandas DataFrame, dropping rows with any missing."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, None, 3, 4],
            'C': [1, 2, 3, 4]
        })

        result = drop_missing(df, axis='rows', how='any')

        assert result.shape[0] == 2  # Only 2 complete rows
        assert not result.isna().any().any()  # No missing values

    def test_drop_missing_pandas_rows_all(self):
        """Test drop_missing with pandas DataFrame, dropping rows with all missing."""
        df = pd.DataFrame({
            'A': [1, None, None, 4],
            'B': [1, None, None, 4],
            'C': [1, 2, 3, 4]
        })

        result = drop_missing(df, axis='rows', how='all')

        assert result.shape[0] == 3  # Only row with all NaN dropped

    def test_drop_missing_pandas_columns_any(self):
        """Test drop_missing with pandas DataFrame, dropping columns with any missing."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, 2, 3, 4],  # No missing values
            'C': [1, None, 3, 4]
        })

        result = drop_missing(df, axis='columns', how='any')

        assert result.shape[1] == 1  # Only column B remains
        assert list(result.columns) == ['B']

    def test_drop_missing_pandas_thresh(self):
        """Test drop_missing with threshold parameter."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, None, None, 4],
            'C': [1, 2, 3, 4]
        })

        result = drop_missing(df, axis='rows', thresh=2)

        # Keep rows with at least 2 non-null values
        assert result.shape[0] == 3

    def test_drop_missing_pandas_subset(self):
        """Test drop_missing with column subset."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, 2, 3, 4],
            'C': [1, None, 3, 4]
        })

        result = drop_missing(df, axis='rows', subset=['A'])

        # Only drop rows where column A has missing values
        assert result.shape[0] == 3
        assert not result['A'].isna().any()

    def test_drop_missing_polars_rows(self):
        """Test drop_missing with polars DataFrame."""
        df = pl.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, None, 3, 4],
            'C': [1, 2, 3, 4]
        })

        result = drop_missing(df, axis='rows')

        assert result.shape[0] == 2  # Only complete rows remain
        assert result.null_count().sum_horizontal().sum() == 0

    def test_drop_missing_polars_columns_any(self):
        """Test drop_missing with polars DataFrame, dropping columns."""
        df = pl.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, 2, 3, 4],  # No missing values
            'C': [1, None, 3, 4]
        })

        result = drop_missing(df, axis='columns', how='any')

        assert result.shape[1] == 1
        assert 'B' in result.columns

    def test_drop_missing_invalid_axis(self):
        """Test drop_missing with invalid axis parameter."""
        df = pd.DataFrame({'A': [1, 2, 3]})

        with pytest.raises(ValueError, match="Axis must be either 'rows' or 'columns'"):
            drop_missing(df, axis='invalid')

    def test_drop_missing_invalid_how(self):
        """Test drop_missing with invalid how parameter."""
        df = pd.DataFrame({'A': [1, 2, 3]})

        with pytest.raises(ValueError, match="how must be either 'any' or 'all'"):
            drop_missing(df, how='invalid')

    def test_fill_missing_invalid_dataframe(self):
        """Test fill_missing with invalid DataFrame type."""
        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            fill_missing("not a dataframe")

    def test_drop_missing_invalid_dataframe(self):
        """Test drop_missing with invalid DataFrame type."""
        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            drop_missing([1, 2, 3])
