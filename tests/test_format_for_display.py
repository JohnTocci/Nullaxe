import sys
import os
import math
import pandas as pd
import polars as pl
import pytest

# Ensure src on path (mirrors style in existing tests)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._format_for_display import format_for_display


@pytest.fixture()
def sample_pandas_df():
    return pd.DataFrame({
        'price': [1234.5, None, 1000000],
        'growth': [0.1234, 0.0, 0.9876],
        'volume': [1000, 2500000, None],
        'description': ['Short', 'A longer description that will be truncated', None],
        'sale_date': ['2024-01-15', '2024-06-30', None],
        'other': [1, 2, 3]
    })


@pytest.fixture()
def sample_polars_df():
    return pl.DataFrame({
        'price': [1234.5, None, 1000000.0],
        'growth': [0.1234, 0.0, 0.9876],
        'volume': [1000, 2500000, None],
        'description': ['Short', 'A longer description that will be truncated', None],
        'sale_date': ["2024-01-15", "2024-06-30", None],
        'other': [1, 2, 3]
    })


@pytest.fixture()
def rules():
    return {
        'price': {'type': 'currency', 'symbol': '$', 'decimals': 2},
        'growth': {'type': 'percentage', 'decimals': 1},
        'volume': {'type': 'thousands'},
        'description': {'type': 'truncate', 'length': 10},
        'sale_date': {'type': 'datetime', 'format': '%B %d, %Y'},
        # Column not present to ensure it's safely ignored
        'missing_col': {'type': 'currency'}
    }


def test_format_for_display_pandas_basic(sample_pandas_df, rules):
    result = format_for_display(sample_pandas_df, rules=rules, column_case='title')

    # Column name casing
    assert 'Price' in result.columns
    assert 'Sale Date' in result.columns
    assert 'Missing Col' not in result.columns  # not added

    # Price formatting
    assert result.loc[0, 'Price'] == '$1,234.50'
    assert result.loc[2, 'Price'] == '$1,000,000.00'

    # Growth percentage
    assert result.loc[0, 'Growth'] == '12.3%'
    assert result.loc[1, 'Growth'] == '0.0%'

    # Thousands separator
    assert result.loc[0, 'Volume'] == '1,000'
    assert result.loc[1, 'Volume'] == '2,500,000'

    # Truncation (length 10 -> add ellipsis only if exceeded)
    assert result.loc[0, 'Description'] == 'Short'  # not truncated
    assert result.loc[1, 'Description'].endswith('...')
    assert len(result.loc[1, 'Description']) == 13  # 10 chars + '...'

    # Datetime formatting
    assert result.loc[0, 'Sale Date'] == 'January 15, 2024'
    assert result.loc[1, 'Sale Date'] == 'June 30, 2024'

    # None/NaN preservation (should remain None/NaN not string 'None' or 'nan')
    assert pd.isna(result.loc[1, 'Price'])


def test_format_for_display_pandas_no_column_case(sample_pandas_df, rules):
    result = format_for_display(sample_pandas_df, rules=rules, column_case=None)
    # Original column names preserved
    assert 'price' in result.columns and 'Price' not in result.columns


def test_format_for_display_polars_basic(sample_polars_df, rules):
    result = format_for_display(sample_polars_df, rules=rules, column_case='title')
    assert isinstance(result, pl.DataFrame)

    # Column names titlecased
    assert 'Price' in result.columns
    assert 'Sale Date' in result.columns

    # Inspect first row (select returns DataFrame/Series like object)
    price_val = result.select('Price').to_series()[0]
    assert price_val == '$1,234.50'

    growth_val = result.select('Growth').to_series()[0]
    assert growth_val == '12.3%'

    volume_val = result.select('Volume').to_series()[0]
    assert volume_val == '1,000'

    desc_val = result.select('Description').to_series()[1]
    assert desc_val.endswith('...') and len(desc_val) == 13

    sale_date_val = result.select('Sale Date').to_series()[0]
    assert sale_date_val == 'January 15, 2024'


def test_format_for_display_invalid_input():
    with pytest.raises(TypeError):
        format_for_display([1, 2, 3], rules={})
