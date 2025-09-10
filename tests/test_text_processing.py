import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add the src directory to the path to import nullaxe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._replace_text import replace_text
from nullaxe.functions._remove_punctuation import remove_punctuation
from nullaxe.functions._extract_with_regex import extract_with_regex


class TestTextProcessingFunctions:
    """Test text processing and manipulation functions."""

    def test_replace_text_pandas(self):
        """Test text replacement with pandas DataFrame."""
        df = pd.DataFrame({
            'text_col': ['Hello World', 'Hello Universe', 'Goodbye World'],
            'numeric_col': [1, 2, 3]
        })

        result = replace_text(df, columns=['text_col'], old='Hello', new='Hi')

        assert result.loc[0, 'text_col'] == 'Hi World'
        assert result.loc[1, 'text_col'] == 'Hi Universe'
        assert result.loc[2, 'text_col'] == 'Goodbye World'  # Unchanged
        assert result['numeric_col'].tolist() == [1, 2, 3]  # Unchanged

    def test_replace_text_polars(self):
        """Test text replacement with polars DataFrame."""
        df = pl.DataFrame({
            'text_col': ['Hello World', 'Hello Universe', 'Goodbye World']
        })

        result = replace_text(df, columns=['text_col'], old='Hello', new='Hi')

        assert result[0, 'text_col'] == 'Hi World'
        assert result[1, 'text_col'] == 'Hi Universe'
        assert result[2, 'text_col'] == 'Goodbye World'

    def test_replace_text_multiple_columns(self):
        """Test text replacement across multiple columns."""
        df = pd.DataFrame({
            'col1': ['cat dog', 'dog cat', 'bird'],
            'col2': ['dog bird', 'cat', 'dog cat bird']
        })

        result = replace_text(df, columns=['col1', 'col2'], old='cat', new='kitten')

        assert result.loc[0, 'col1'] == 'kitten dog'
        assert result.loc[1, 'col1'] == 'dog kitten'
        assert result.loc[1, 'col2'] == 'kitten'
        assert result.loc[2, 'col2'] == 'dog kitten bird'

    def test_replace_text_regex_patterns(self):
        """Test text replacement with regex patterns."""
        df = pd.DataFrame({
            'text_col': ['phone: 123-456-7890', 'call 555-123-4567', 'no phone here']
        })

        result = replace_text(df, columns=['text_col'], old=r'\d{3}-\d{3}-\d{4}', new='[PHONE]', regex=True)

        assert result.loc[0, 'text_col'] == 'phone: [PHONE]'
        assert result.loc[1, 'text_col'] == 'call [PHONE]'
        assert result.loc[2, 'text_col'] == 'no phone here'  # Unchanged

    def test_remove_punctuation_pandas(self):
        """Test punctuation removal with pandas DataFrame."""
        df = pd.DataFrame({
            'text_col': ['Hello, World!', 'Test... string?', 'No punctuation here', ''],
            'numeric_col': [1, 2, 3, 4]
        })

        result = remove_punctuation(df, columns=['text_col'])

        assert result.loc[0, 'text_col'] == 'Hello World'
        assert result.loc[1, 'text_col'] == 'Test string'
        assert result.loc[2, 'text_col'] == 'No punctuation here'  # Unchanged
        assert result.loc[3, 'text_col'] == ''  # Empty string unchanged
        assert result['numeric_col'].tolist() == [1, 2, 3, 4]  # Unchanged

    def test_remove_punctuation_polars(self):
        """Test punctuation removal with polars DataFrame."""
        df = pl.DataFrame({
            'text_col': ['Hello, World!', 'Test... string?', 'No punctuation here']
        })

        result = remove_punctuation(df, columns=['text_col'])

        assert result[0, 'text_col'] == 'Hello World'
        assert result[1, 'text_col'] == 'Test string'
        assert result[2, 'text_col'] == 'No punctuation here'

    def test_extract_with_regex_pandas(self):
        """Test regex extraction with pandas DataFrame."""
        df = pd.DataFrame({
            'text_col': [
                'Product ID: ABC123',
                'Item: XYZ789',
                'No ID here',
                'Code: DEF456GHI'
            ]
        })

        result = extract_with_regex(df, columns=['text_col'], pattern=r'[A-Z]{3}\d{3}', new_column='extracted_id')

        assert 'extracted_id' in result.columns
        assert result.loc[0, 'extracted_id'] == 'ABC123'
        assert result.loc[1, 'extracted_id'] == 'XYZ789'
        assert pd.isna(result.loc[2, 'extracted_id'])  # No match
        assert result.loc[3, 'extracted_id'] == 'DEF456'  # First match

    def test_extract_with_regex_polars(self):
        """Test regex extraction with polars DataFrame."""
        df = pl.DataFrame({
            'text_col': [
                'Product ID: ABC123',
                'Item: XYZ789',
                'No ID here'
            ]
        })

        result = extract_with_regex(df, columns=['text_col'], pattern=r'[A-Z]{3}\d{3}', new_column='extracted_id')

        assert 'extracted_id' in result.columns
        assert result[0, 'extracted_id'] == 'ABC123'
        assert result[1, 'extracted_id'] == 'XYZ789'

    def test_complex_punctuation_removal(self):
        """Test removal of various punctuation marks."""
        df = pd.DataFrame({
            'text_col': [
                'Hello, World!',
                'Test... (with) [brackets] {and} <tags>',
                'Symbols: @#$%^&*()_+-=[]{}|;:,.<>?',
                "Quotes: 'single' and \"double\"",
                'Mixed: abc123!@#def456$%^'
            ]
        })

        result = remove_punctuation(df, columns=['text_col'])

        assert result.loc[0, 'text_col'] == 'Hello World'
        assert result.loc[1, 'text_col'] == 'Test with brackets and tags'
        assert result.loc[2, 'text_col'] == 'Symbols '
        assert result.loc[3, 'text_col'] == 'Quotes single and double'
        assert result.loc[4, 'text_col'] == 'Mixed abc123def456'

    def test_regex_extraction_edge_cases(self):
        """Test regex extraction with edge cases."""
        df = pd.DataFrame({
            'text_col': [
                '',  # Empty string
                'No matches here',
                'Multiple ABC123 and XYZ789 matches',
                'Partial ABC12 match',
                None  # Null value
            ]
        })

        result = extract_with_regex(df, columns=['text_col'], pattern=r'[A-Z]{3}\d{3}', new_column='codes')

        assert pd.isna(result.loc[0, 'codes'])  # Empty string
        assert pd.isna(result.loc[1, 'codes'])  # No matches
        assert result.loc[2, 'codes'] == 'ABC123'  # First match only
        assert pd.isna(result.loc[3, 'codes'])  # Partial match
        assert pd.isna(result.loc[4, 'codes'])  # Null input

    def test_text_replacement_case_sensitivity(self):
        """Test case sensitivity in text replacement."""
        df = pd.DataFrame({
            'text_col': ['Hello World', 'HELLO world', 'hello WORLD']
        })

        # Case sensitive replacement (default)
        result_sensitive = replace_text(df, columns=['text_col'], old='Hello', new='Hi')
        assert result_sensitive.loc[0, 'text_col'] == 'Hi World'
        assert result_sensitive.loc[1, 'text_col'] == 'HELLO world'  # Unchanged
        assert result_sensitive.loc[2, 'text_col'] == 'hello WORLD'  # Unchanged

        # Case insensitive replacement using regex
        result_insensitive = replace_text(df, columns=['text_col'], old=r'(?i)hello', new='Hi', regex=True)
        assert result_insensitive.loc[0, 'text_col'] == 'Hi World'
        assert result_insensitive.loc[1, 'text_col'] == 'Hi world'
        assert result_insensitive.loc[2, 'text_col'] == 'Hi WORLD'

    def test_text_functions_with_missing_values(self):
        """Test text functions handle missing values correctly."""
        df = pd.DataFrame({
            'text_col': ['Hello World', None, '', 'Test String', pd.NA]
        })

        # Test replace_text with missing values
        result_replace = replace_text(df, columns=['text_col'], old='Hello', new='Hi')
        assert result_replace.loc[0, 'text_col'] == 'Hi World'
        assert pd.isna(result_replace.loc[1, 'text_col'])
        assert result_replace.loc[2, 'text_col'] == ''
        assert result_replace.loc[3, 'text_col'] == 'Test String'

        # Test remove_punctuation with missing values
        df_punct = pd.DataFrame({
            'text_col': ['Hello, World!', None, '', 'Test!', pd.NA]
        })
        result_punct = remove_punctuation(df_punct, columns=['text_col'])
        assert result_punct.loc[0, 'text_col'] == 'Hello World'
        assert pd.isna(result_punct.loc[1, 'text_col'])
        assert result_punct.loc[2, 'text_col'] == ''
        assert result_punct.loc[3, 'text_col'] == 'Test'

    def test_invalid_regex_patterns(self):
        """Test handling of invalid regex patterns."""
        df = pd.DataFrame({
            'text_col': ['Test string', 'Another test']
        })

        # Invalid regex pattern should raise an error
        with pytest.raises(Exception):  # Could be re.error or similar
            extract_with_regex(df, columns=['text_col'], pattern='[invalid', new_column='result')
