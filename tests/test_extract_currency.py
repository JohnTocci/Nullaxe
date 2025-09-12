import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._extract_currency import extract_currency


class TestExtractCurrencyPandas:
    def test_basic_currency_extraction(self):
        """Test basic currency extraction with various formats."""
        df = pd.DataFrame({
            'price': ['$123.45', '€1,234.56', '£999.99', '¥5000', '₹15,000.00'],
            'description': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
        })

        result = extract_currency(df, subset=['price'])

        assert 'price_currency' in result.columns
        assert result['price_currency'].iloc[0] == '$123.45'
        assert result['price_currency'].iloc[1] == '€1,234.56'
        assert result['price_currency'].iloc[2] == '£999.99'
        assert result['price_currency'].iloc[3] == '¥5000'
        assert result['price_currency'].iloc[4] == '₹15,000.00'

    def test_currency_without_symbols(self):
        """Test extraction of numeric values without currency symbols."""
        df = pd.DataFrame({
            'amount': ['123.45', '1,234.56', '999', '5000.00', '15,000'],
            'text': ['Some text', 'More text', 'Other', 'Data', 'Info']
        })

        result = extract_currency(df, subset=['amount'])

        assert 'amount_currency' in result.columns
        assert result['amount_currency'].iloc[0] == '123.45'
        assert result['amount_currency'].iloc[1] == '1,234.56'
        assert result['amount_currency'].iloc[2] == '999'
        assert result['amount_currency'].iloc[3] == '5000.00'
        assert result['amount_currency'].iloc[4] == '15,000'

    def test_mixed_content_extraction(self):
        """Test extraction from mixed content strings."""
        df = pd.DataFrame({
            'text': [
                'The price is $123.45 for this item',
                'Cost: €1,234.56 including tax',
                'Total £999.99 paid',
                'No currency here',
                'Multiple $100 and €50.25 amounts'
            ]
        })

        result = extract_currency(df, subset=['text'])

        assert 'text_currency' in result.columns
        assert result['text_currency'].iloc[0] == '$123.45'
        assert result['text_currency'].iloc[1] == '€1,234.56'
        assert result['text_currency'].iloc[2] == '£999.99'
        # For no currency, it should be NaN or empty
        assert pd.isna(result['text_currency'].iloc[3]) or result['text_currency'].iloc[3] == ''
        # For multiple currencies, it should extract the first one
        assert result['text_currency'].iloc[4] == '$100'

    def test_no_currency_values(self):
        """Test handling of strings with no currency values."""
        df = pd.DataFrame({
            'text': ['hello world', 'no numbers', 'just text', 'abc def'],
            'other': ['data', 'more', 'info', 'here']
        })

        result = extract_currency(df, subset=['text'])

        assert 'text_currency' in result.columns
        # All should be NaN or empty since no currency patterns
        for val in result['text_currency']:
            assert pd.isna(val) or val == ''

    def test_subset_parameter(self):
        """Test that only specified columns are processed."""
        df = pd.DataFrame({
            'price1': ['$123.45', '$234.56'],
            'price2': ['€345.67', '€456.78'],
            'description': ['Item 1', 'Item 2']
        })

        result = extract_currency(df, subset=['price1'])

        assert 'price1_currency' in result.columns
        assert 'price2_currency' not in result.columns
        assert 'description_currency' not in result.columns

    def test_non_string_columns_ignored(self):
        """Test that non-string columns are ignored."""
        df = pd.DataFrame({
            'numeric_col': [123.45, 234.56, 345.67],
            'string_col': ['$123.45', '$234.56', '$345.67']
        })

        result = extract_currency(df, subset=['numeric_col', 'string_col'])

        assert 'numeric_col_currency' not in result.columns
        assert 'string_col_currency' in result.columns

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        result = extract_currency(df, subset=[])

        assert result.empty

    def test_nonexistent_columns(self):
        """Test handling of nonexistent columns in subset."""
        df = pd.DataFrame({
            'existing_col': ['$123.45', '$234.56']
        })

        result = extract_currency(df, subset=['existing_col', 'nonexistent_col'])

        assert 'existing_col_currency' in result.columns
        assert 'nonexistent_col_currency' not in result.columns

    def test_null_values(self):
        """Test handling of null/NaN values."""
        df = pd.DataFrame({
            'price': ['$123.45', None, '$345.67', '', '$456.78']
        })

        result = extract_currency(df, subset=['price'])

        assert 'price_currency' in result.columns
        assert result['price_currency'].iloc[0] == '$123.45'
        assert pd.isna(result['price_currency'].iloc[1]) or result['price_currency'].iloc[1] == ''
        assert result['price_currency'].iloc[2] == '$345.67'
        assert pd.isna(result['price_currency'].iloc[4]) or result['price_currency'].iloc[4] == '$456.78'


class TestExtractCurrencyPolars:
    def test_basic_currency_extraction_polars(self):
        """Test basic currency extraction with Polars DataFrame."""
        df = pl.DataFrame({
            'price': ['$123.45', '€1,234.56', '£999.99', '¥5000', '₹15,000.00'],
            'description': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
        })

        result = extract_currency(df, subset=['price'])

        assert 'price_currency' in result.columns
        currency_col = result['price_currency'].to_list()
        assert currency_col[0] == '$123.45'
        assert currency_col[1] == '€1,234.56'
        assert currency_col[2] == '£999.99'
        assert currency_col[3] == '¥5000'
        assert currency_col[4] == '₹15,000.00'

    def test_mixed_content_polars(self):
        """Test extraction from mixed content strings with Polars."""
        df = pl.DataFrame({
            'text': [
                'The price is $123.45 for this item',
                'Cost: €1,234.56 including tax',
                'Total £999.99 paid',
                'No currency here',
                'Multiple $100 and €50.25 amounts'
            ]
        })

        result = extract_currency(df, subset=['text'])

        assert 'text_currency' in result.columns
        currency_col = result['text_currency'].to_list()
        assert currency_col[0] == '$123.45'
        assert currency_col[1] == '€1,234.56'
        assert currency_col[2] == '£999.99'
        # For no currency, it should be None
        assert currency_col[3] is None
        # For multiple currencies, it should extract the first one
        assert currency_col[4] == '$100'

    def test_subset_parameter_polars(self):
        """Test that only specified columns are processed with Polars."""
        df = pl.DataFrame({
            'price1': ['$123.45', '$234.56'],
            'price2': ['€345.67', '€456.78'],
            'description': ['Item 1', 'Item 2']
        })

        result = extract_currency(df, subset=['price1'])

        assert 'price1_currency' in result.columns
        assert 'price2_currency' not in result.columns
        assert 'description_currency' not in result.columns

    def test_non_string_columns_ignored_polars(self):
        """Test that non-string columns are ignored with Polars."""
        df = pl.DataFrame({
            'numeric_col': [123.45, 234.56, 345.67],
            'string_col': ['$123.45', '$234.56', '$345.67']
        })

        result = extract_currency(df, subset=['numeric_col', 'string_col'])

        assert 'numeric_col_currency' not in result.columns
        assert 'string_col_currency' in result.columns

    def test_null_values_polars(self):
        """Test handling of null values with Polars."""
        df = pl.DataFrame({
            'price': ['$123.45', None, '$345.67', '', '$456.78']
        })

        result = extract_currency(df, subset=['price'])

        assert 'price_currency' in result.columns
        currency_col = result['price_currency'].to_list()
        assert currency_col[0] == '$123.45'
        assert currency_col[1] is None
        assert currency_col[2] == '$345.67'
        # Empty string case
        assert currency_col[3] is None or currency_col[3] == ''
        assert currency_col[4] == '$456.78'


class TestExtractCurrencyCommon:
    def test_invalid_input_type(self):
        """Test that function raises TypeError for invalid input types."""
        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            extract_currency("not a dataframe", subset=['col'])

        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            extract_currency(['list', 'not', 'dataframe'], subset=['col'])

    def test_currency_regex_patterns(self):
        """Test various currency patterns are correctly matched."""
        test_cases = [
            ('$1', '$1'),
            ('$1.23', '$1.23'),
            ('$1,234', '$1,234'),
            ('$1,234.56', '$1,234.56'),
            ('€500', '€500'),
            ('£99.99', '£99.99'),
            ('¥1000', '¥1000'),
            ('₹2,500.00', '₹2,500.00'),
            ('123.45', '123.45'),  # No symbol
            ('1,000', '1,000'),    # No symbol with comma
            ('$ 100', '$100'),     # Space after symbol gets normalized
        ]

        for input_val, expected in test_cases:
            df = pd.DataFrame({'amount': [input_val]})
            result = extract_currency(df, subset=['amount'])

            assert result['amount_currency'].iloc[0] == expected, f"Failed for input: {input_val}"

    def test_edge_cases(self):
        """Test edge cases and corner scenarios."""
        df = pd.DataFrame({
            'edge_cases': [
                '$0.01',           # Minimum meaningful currency
                '$999,999.99',     # Large amount
                '$.50',            # Missing leading zero
                '$1,000,000',      # Million
                'Price: $50 only', # Text with currency
                '$-100',           # Negative (should not match)
                'abc$123def',      # Currency in middle
                '$',               # Symbol only
                '123',             # Plain number
                ''                 # Empty string
            ]
        })

        result = extract_currency(df, subset=['edge_cases'])

        assert 'edge_cases_currency' in result.columns
        # Verify some specific cases
        currency_col = result['edge_cases_currency']
        assert currency_col.iloc[0] == '$0.01'
        assert currency_col.iloc[1] == '$999,999.99'
        assert currency_col.iloc[4] == '$50'  # Should extract from text
        assert currency_col.iloc[8] == '123'  # Plain number should work
