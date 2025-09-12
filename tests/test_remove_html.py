import pytest
import pandas as pd
import polars as pl
import sys
import os

# Ensure src directory is on path (consistent with other tests)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._remove_html import remove_html  # noqa: E402


class TestRemoveHTML:
    def test_remove_html_pandas_basic(self):
        df = pd.DataFrame({
            'col': [
                '<p>Hello</p>',
                'No tags',
                '<div><span>Nested</span> Text</div>',
                '<a href="https://example.com">Link</a>',
                '<img src="x" />Image'
            ]
        })
        result = remove_html(df.copy())
        assert result.loc[0, 'col'] == 'Hello'
        assert result.loc[1, 'col'] == 'No tags'
        # Nested tags collapse to combined inner text
        assert result.loc[2, 'col'] == 'Nested Text'
        # Anchor text retained
        assert result.loc[3, 'col'] == 'Link'
        # Self-closing tag removed, rest kept
        assert result.loc[4, 'col'] == 'Image'

    def test_remove_html_pandas_subset(self):
        df = pd.DataFrame({
            'html1': ['<b>Bold</b>', 'Plain'],
            'html2': ['<i>Italic</i>', '<u>Under</u>'],
            'num': [1, 2]
        })
        result = remove_html(df.copy(), subset=['html2'])
        # Only html2 cleaned
        assert result.loc[0, 'html1'] == '<b>Bold</b>'
        assert result.loc[0, 'html2'] == 'Italic'
        assert result.loc[1, 'html2'] == 'Under'

    def test_remove_html_polars_basic(self):
        df = pl.DataFrame({
            'text': ['<h1>Title</h1>', 'NoHTML', '<p>A <strong>B</strong></p>']
        })
        result = remove_html(df)
        assert result[0, 'text'] == 'Title'
        assert result[1, 'text'] == 'NoHTML'
        assert result[2, 'text'] == 'A B'

    def test_remove_html_polars_subset(self):
        df = pl.DataFrame({
            'a': ['<div>X</div>', 'Y'],
            'b': ['<span>Z</span>', 'Q']
        })
        result = remove_html(df, subset=['b'])
        assert result[0, 'a'] == '<div>X</div>'  # untouched
        assert result[0, 'b'] == 'Z'

    def test_remove_html_mixed_types(self):
        df = pd.DataFrame({
            'html': ['<p>123</p>', '<code>456</code>'],
            'num': [10, 20]
        })
        result = remove_html(df.copy())
        assert result.loc[0, 'html'] == '123'
        assert result.loc[1, 'html'] == '456'
        # Numeric column unchanged
        assert result.loc[0, 'num'] == 10

    def test_remove_html_invalid_input(self):
        with pytest.raises(TypeError):
            remove_html({'not': 'df'})  # type: ignore

