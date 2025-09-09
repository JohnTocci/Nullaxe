import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add the src directory to the path to import sanex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanex.functions._extract_email import extract_email
from sanex.functions._extract_phone_numbers import extract_phone_numbers
from sanex.functions._extract_and_clean_numeric import extract_and_clean_numeric, clean_numeric


class TestDataExtractionFunctions:
    """Test data extraction and cleaning functions."""

    def test_extract_email_pandas(self):
        """Test email extraction with pandas DataFrame."""
        df = pd.DataFrame({
            'text_col': [
                'Contact john@email.com for info',
                'Invalid email format',
                'Multiple emails: alice@test.org and bob@company.co.uk',
                'No email here',
                'admin@domain.net'
            ],
            'numeric_col': [1, 2, 3, 4, 5]
        })

        result = extract_email(df)

        # Should create new column with extracted emails
        assert 'text_col_email' in result.columns
        assert result.loc[0, 'text_col_email'] == 'john@email.com'
        assert pd.isna(result.loc[1, 'text_col_email'])  # No valid email
        assert result.loc[2, 'text_col_email'] == 'alice@test.org'  # First email found
        assert result.loc[4, 'text_col_email'] == 'admin@domain.net'

    def test_extract_email_pandas_subset(self):
        """Test email extraction with column subset."""
        df = pd.DataFrame({
            'email_col': ['user@example.com', 'not an email'],
            'other_col': ['another@test.com', 'text'],
            'numeric_col': [1, 2]
        })

        result = extract_email(df, subset=['email_col'])

        assert 'email_col_email' in result.columns
        assert 'other_col_email' not in result.columns  # Not in subset
        assert result.loc[0, 'email_col_email'] == 'user@example.com'

    def test_extract_phone_numbers_pandas(self):
        """Test phone number extraction with pandas DataFrame."""
        df = pd.DataFrame({
            'contact_info': [
                '123-456-7890',
                '(555) 123-4567',
                '+1-800-555-0199',
                'Not a phone number',
                '555.123.4567'
            ]
        })

        result = extract_phone_numbers(df)

        assert 'contact_info_phone' in result.columns
        assert result.loc[0, 'contact_info_phone'] == '123-456-7890'
        assert result.loc[1, 'contact_info_phone'] == '(555) 123-4567'
        assert pd.isna(result.loc[3, 'contact_info_phone'])  # Invalid phone

    def test_extract_and_clean_numeric_pandas(self):
        """Test numeric extraction and cleaning with pandas DataFrame."""
        df = pd.DataFrame({
            'price_col': ['$50,000', '$75,000.50', '60000', 'Not a number', '45k'],
            'text_col': ['Text only', 'More text', 'Still text', 'No numbers', 'Text']
        })

        result = extract_and_clean_numeric(df)

        assert 'price_col_numeric' in result.columns
        assert result.loc[0, 'price_col_numeric'] == 50000.0
        assert result.loc[1, 'price_col_numeric'] == 75000.50
        assert result.loc[2, 'price_col_numeric'] == 60000.0
        assert pd.isna(result.loc[3, 'price_col_numeric'])  # No valid number

    def test_clean_numeric_pandas(self):
        """Test numeric cleaning with pandas DataFrame."""
        df = pd.DataFrame({
            'messy_numbers': ['$1,234.56', '€2,345.67', '¥3,456', '4567.89', 'invalid']
        })

        result = clean_numeric(df, columns=['messy_numbers'])

        assert result.loc[0, 'messy_numbers'] == 1234.56
        assert result.loc[1, 'messy_numbers'] == 2345.67
        assert result.loc[2, 'messy_numbers'] == 3456.0
        assert result.loc[3, 'messy_numbers'] == 4567.89
        assert pd.isna(result.loc[4, 'messy_numbers'])  # Invalid number

    def test_extract_email_polars(self):
        """Test email extraction with polars DataFrame."""
        df = pl.DataFrame({
            'text_col': [
                'Contact john@email.com for info',
                'Invalid email format',
                'admin@domain.net'
            ]
        })

        result = extract_email(df)

        assert 'text_col_email' in result.columns
        assert result[0, 'text_col_email'] == 'john@email.com'
        assert result[2, 'text_col_email'] == 'admin@domain.net'

    def test_extract_phone_numbers_polars(self):
        """Test phone number extraction with polars DataFrame."""
        df = pl.DataFrame({
            'contact_info': [
                '123-456-7890',
                'Not a phone number',
                '+1-800-555-0199'
            ]
        })

        result = extract_phone_numbers(df)

        assert 'contact_info_phone' in result.columns
        assert result[0, 'contact_info_phone'] == '123-456-7890'
        assert result[2, 'contact_info_phone'] == '+1-800-555-0199'

    def test_email_regex_validation(self):
        """Test email regex pattern validation."""
        df = pd.DataFrame({
            'emails': [
                'valid@email.com',
                'also.valid@test.org',
                'user+tag@example.co.uk',
                'invalid.email',
                '@invalid.com',
                'invalid@',
                'spaces in@email.com',
                'multiple@at@signs.com'
            ]
        })

        result = extract_email(df)

        # Valid emails should be extracted
        assert result.loc[0, 'emails_email'] == 'valid@email.com'
        assert result.loc[1, 'emails_email'] == 'also.valid@test.org'
        assert result.loc[2, 'emails_email'] == 'user+tag@example.co.uk'

        # Invalid emails should be NaN
        assert pd.isna(result.loc[3, 'emails_email'])
        assert pd.isna(result.loc[4, 'emails_email'])
        assert pd.isna(result.loc[5, 'emails_email'])

    def test_phone_number_formats(self):
        """Test various phone number format recognition."""
        df = pd.DataFrame({
            'phones': [
                '123-456-7890',      # Standard format
                '(555) 123-4567',    # Parentheses format
                '+1-800-555-0199',   # International format
                '555.123.4567',      # Dot format
                '5551234567',        # No separators
                '123-45-6789',       # Too short
                'not a phone',       # Invalid
                '123-456-78901'      # Too long
            ]
        })

        result = extract_phone_numbers(df)

        # Valid phone numbers should be extracted
        assert result.loc[0, 'phones_phone'] == '123-456-7890'
        assert result.loc[1, 'phones_phone'] == '(555) 123-4567'
        assert result.loc[2, 'phones_phone'] == '+1-800-555-0199'

        # Invalid formats should be NaN
        assert pd.isna(result.loc[6, 'phones_phone'])  # 'not a phone'

    def test_numeric_extraction_edge_cases(self):
        """Test numeric extraction with edge cases."""
        df = pd.DataFrame({
            'values': [
                '$0',               # Zero
                '-$1,234.56',      # Negative
                '$1,234,567.89',   # Large number
                '€0.01',           # Small decimal
                '100%',            # Percentage
                '$',               # Just symbol
                '',                # Empty string
                '1.2.3',           # Multiple decimals
                'abc123def'        # Mixed text/numbers
            ]
        })

        result = extract_and_clean_numeric(df)

        assert result.loc[0, 'values_numeric'] == 0.0
        assert result.loc[1, 'values_numeric'] == -1234.56
        assert result.loc[2, 'values_numeric'] == 1234567.89
        assert result.loc[3, 'values_numeric'] == 0.01
        assert pd.isna(result.loc[5, 'values_numeric'])  # Just '$'
        assert pd.isna(result.loc[6, 'values_numeric'])  # Empty string
