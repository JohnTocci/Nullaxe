import pytest
import pandas as pd
import polars as pl
import sys
import os

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanex.functions._remove_special_characters import (
    remove_special_characters,
    remove_emojis,
    remove_non_alphanumeric,
    remove_non_numeric,
    remove_non_ascii,
)


class TestSpecialCharacterFunctions:
    def test_remove_special_characters_pandas_default(self):
        df = pd.DataFrame({
            'text': ['Hello@World#2024!', 'No$pecial^Here', None, ''],
            'num': [1, 2, 3, 4]
        })
        result = remove_special_characters(df, columns=['text'])
        assert result.loc[0, 'text'] == 'HelloWorld2024'
        assert result.loc[1, 'text'] == 'NopecialHere'
        assert pd.isna(result.loc[2, 'text'])
        assert result.loc[3, 'text'] == ''
        assert result['num'].tolist() == [1, 2, 3, 4]

    def test_remove_special_characters_pandas_custom_set(self):
        df = pd.DataFrame({'text': ['A@B#C!D']})
        # Remove only @ and #, keep !
        result = remove_special_characters(df, columns=['text'], characters='@#')
        assert result.loc[0, 'text'] == 'ABC!D'

    def test_remove_special_characters_polars(self):
        df = pl.DataFrame({'text': ['Hello@World#2024!', 'No$pecial^Here']})
        result = remove_special_characters(df, columns=['text'])
        assert result[0, 'text'] == 'HelloWorld2024'
        assert result[1, 'text'] == 'NopecialHere'

    def test_remove_emojis_pandas(self):
        df = pd.DataFrame({'text': ['Hi 😊 there 🚀!', 'No emoji']})
        result = remove_emojis(df, columns=['text'])
        assert result.loc[0, 'text'] == 'Hi  there !'
        assert result.loc[1, 'text'] == 'No emoji'

    def test_remove_emojis_polars(self):
        df = pl.DataFrame({'text': ['Hi 😊 there 🚀!', 'No emoji']})
        result = remove_emojis(df, columns=['text'])
        assert result[0, 'text'] == 'Hi  there !'
        assert result[1, 'text'] == 'No emoji'

    def test_remove_non_alphanumeric_pandas(self):
        df = pd.DataFrame({'text': ['a-b_c@d! e#f$', None, '123 ABC!!!']})
        result = remove_non_alphanumeric(df, columns=['text'])
        assert result.loc[0, 'text'] == 'abcd ef'
        assert pd.isna(result.loc[1, 'text'])
        assert result.loc[2, 'text'] == '123 ABC'

    def test_remove_non_alphanumeric_polars(self):
        df = pl.DataFrame({'text': ['a-b_c@d! e#f$', '123 ABC!!!']})
        result = remove_non_alphanumeric(df, columns=['text'])
        assert result[0, 'text'] == 'abcd ef'
        assert result[1, 'text'] == '123 ABC'

    def test_remove_non_numeric_pandas(self):
        df = pd.DataFrame({'text': ['Price: $1,234.56', 'abc123xyz', None, '']})
        result = remove_non_numeric(df, columns=['text'])
        assert result.loc[0, 'text'] == '123456'
        assert result.loc[1, 'text'] == '123'
        assert pd.isna(result.loc[2, 'text'])
        assert result.loc[3, 'text'] == ''

    def test_remove_non_numeric_polars(self):
        df = pl.DataFrame({'text': ['Price: $1,234.56', 'abc123xyz']})
        result = remove_non_numeric(df, columns=['text'])
        assert result[0, 'text'] == '123456'
        assert result[1, 'text'] == '123'

    def test_remove_non_ascii_pandas(self):
        df = pd.DataFrame({'text': ['Café naïve – élève', 'ASCII only', None]})
        result = remove_non_ascii(df, columns=['text'])
        assert result.loc[0, 'text'] == 'Caf nave  lve'
        assert result.loc[1, 'text'] == 'ASCII only'
        assert pd.isna(result.loc[2, 'text'])

    def test_remove_non_ascii_polars(self):
        df = pl.DataFrame({'text': ['Café naïve – élève', 'ASCII only']})
        result = remove_non_ascii(df, columns=['text'])
        assert result[0, 'text'] == 'Caf nave  lve'
        assert result[1, 'text'] == 'ASCII only'

    def test_ignores_nonexistent_columns(self):
        df = pd.DataFrame({'text': ['@Hello#', 'World!'], 'other': ['keep', 'me']})
        # Passing a non-existent column should not raise and should keep data unchanged in existing columns
        result = remove_special_characters(df, columns=['missing_col'])
        assert result.equals(df)
