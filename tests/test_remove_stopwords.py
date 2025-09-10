import pandas as pd
import polars as pl
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanex.functions import remove_stopwords
from sanex.cleaner import Sanex


def test_remove_stopwords_pandas_basic():
    df = pd.DataFrame({'text': ['This is a simple test sentence', 'Another line with words', None, pd.NA]})
    result = remove_stopwords(df.copy(), subset=['text'], language='english')
    assert result.loc[0, 'text'] == 'simple test sentence'
    assert result.loc[1, 'text'] == 'Another line words'
    assert pd.isna(result.loc[2, 'text'])
    assert pd.isna(result.loc[3, 'text'])


def test_remove_stopwords_polars_basic():
    df = pl.DataFrame({'text': ['This is a simple test', 'Just words here']})
    result = remove_stopwords(df, subset=['text'], language='english')
    tokens = result[0, 'text'].split()
    assert 'This' not in tokens and 'is' not in tokens and 'a' not in tokens


def test_remove_stopwords_subset_ignores_missing():
    df = pd.DataFrame({'col1': ['This is text'], 'col2': ['Keep this intact']})
    result = remove_stopwords(df.copy(), subset=['missing', 'col1'], language='english')
    assert result.loc[0, 'col1'] == 'text'
    assert result.loc[0, 'col2'] == 'Keep this intact'


def test_remove_stopwords_multilingual():
    df = pd.DataFrame({'text': ['Ceci est une phrase de test', 'Une autre ligne']})
    try:
        result = remove_stopwords(df.copy(), subset=['text'], language='french')
        tokens0 = result.loc[0, 'text'].lower().split()
        tokens1 = result.loc[1, 'text'].lower().split()
        assert 'est' not in tokens0
        assert 'une' not in tokens0
        assert 'de' not in tokens0
        assert 'une' not in tokens1
    except LookupError:
        pytest.skip('French stopwords not available in environment')


def test_remove_stopwords_chain_with_sanex_pandas():
    df = pd.DataFrame({'text': ['This is a test of the chained interface']})
    sx = Sanex(df)
    sx.remove_stopwords(subset=['text'])
    tokens = sx.to_df().loc[0, 'text'].split()
    assert 'This' not in tokens and 'is' not in tokens and 'a' not in tokens


def test_remove_stopwords_chain_with_sanex_polars():
    df = pl.DataFrame({'text': ['This is a test line for polars']})
    sx = Sanex(df)
    sx.remove_stopwords(subset=['text'])
    tokens = sx.to_df()[0, 'text'].split()
    assert 'This' not in tokens and 'is' not in tokens and 'a' not in tokens


def test_remove_stopwords_non_string_columns_untouched():
    df = pd.DataFrame({'text': ['This is data'], 'num': [1]})
    result = remove_stopwords(df.copy(), subset=['text', 'num'], language='english')
    assert result.loc[0, 'text'] == 'data'
    assert result.loc[0, 'num'] == 1


def test_remove_stopwords_empty_tokens():
    df = pd.DataFrame({'text': ['the and if or but']})
    result = remove_stopwords(df.copy(), subset=['text'], language='english')
    assert result.loc[0, 'text'] == ''
