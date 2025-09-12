import pytest
import pandas as pd
import polars as pl
import sys
import os

# Ensure src on path (mirrors style in other test modules)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._extract_urls import extract_urls  # noqa: E402


class TestExtractUrls:
    def test_extract_urls_pandas_basic(self):
        df = pd.DataFrame({
            'text': [
                'Visit https://example.com for more info',
                'No url here',
                'Multiple: http://a.com and https://b.org'
            ],
            'number': [1, 2, 3]
        })

        result = extract_urls(df)

        assert 'text_url' in result.columns
        # Should extract first URL only
        assert result.loc[0, 'text_url'] == 'https://example.com'
        assert pd.isna(result.loc[1, 'text_url'])
        assert result.loc[2, 'text_url'] == 'http://a.com'
        # Numeric column should not produce a corresponding _url column
        assert 'number_url' not in result.columns

    def test_extract_urls_pandas_subset(self):
        df = pd.DataFrame({
            'col1': ['See http://first.com', 'Nothing here'],
            'col2': ['Another https://second.org', 'Still none']
        })

        result = extract_urls(df, subset=['col1'])

        assert 'col1_url' in result.columns
        assert 'col2_url' not in result.columns  # subset respected
        assert result.loc[0, 'col1_url'] == 'http://first.com'
        assert pd.isna(result.loc[1, 'col1_url'])

    def test_extract_urls_polars_basic(self):
        df = pl.DataFrame({
            'text': [
                'Link https://site.io',
                'No link',
                'Two http://one.net then https://two.net'
            ],
            'vals': [10, 20, 30]
        })

        result = extract_urls(df)

        assert 'text_url' in result.columns
        assert result[0, 'text_url'] == 'https://site.io'
        assert result[1, 'text_url'] is None
        assert result[2, 'text_url'] == 'http://one.net'
        assert 'vals_url' not in result.columns

    def test_extract_urls_polars_subset(self):
        df = pl.DataFrame({
            'a': ['Go to https://aaa.com', 'No link'],
            'b': ['Visit http://bbb.org', 'Nothing']
        })

        result = extract_urls(df, subset=['b'])

        assert 'b_url' in result.columns
        assert 'a_url' not in result.columns
        assert result[0, 'b_url'] == 'http://bbb.org'
        assert result[1, 'b_url'] is None

    def test_extract_urls_invalid_input(self):
        with pytest.raises(TypeError):
            extract_urls(['not', 'a', 'dataframe'])  # type: ignore

