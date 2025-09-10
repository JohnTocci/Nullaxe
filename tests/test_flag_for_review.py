import pytest
import pandas as pd
import polars as pl
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._flag_for_review import flag_for_review


def test_flag_for_review_pandas_basic():
    df = pd.DataFrame({
        'col1': ['This is ok', 'Contains error code', None, 'FAIL case', 'another Error?'],
        'col2': [123, 456, 789, 101, 112]
    })
    # Use case-insensitive error pattern and explicit FAIL whole word
    patterns = [r'(?i)error', r'\bFAIL\b']

    result = flag_for_review(df.copy(), subset=['col1'], patterns=patterns)

    assert 'flagged_for_review' in result.columns
    # Expected flagged rows: indices 1 (error), 3 (FAIL), 4 (Error with different case)
    expected = [False, True, False, True, True]
    assert result['flagged_for_review'].tolist() == expected


def test_flag_for_review_pandas_missing_column():
    df = pd.DataFrame({
        'text': ['ok', 'error here', 'fine', 'FAIL', None],
        'num': [1, 2, 3, 4, 5]
    })
    patterns = [r'(?i)error', r'\bFAIL\b']

    # Include a missing column in subset; should be ignored silently
    result = flag_for_review(df.copy(), subset=['text', 'missing'], patterns=patterns)

    expected = [False, True, False, True, False]
    assert result['flagged_for_review'].tolist() == expected


def test_flag_for_review_polars_basic():
    df = pl.DataFrame({
        'text': ['All good', 'Possible FAIL', 'error found', None],
        'other': ['x', 'y', 'z', 'w']
    })
    patterns = [r'(?i)error', r'\bFAIL\b']

    result = flag_for_review(df, subset=['text'], patterns=patterns)

    assert 'flagged_for_review' in result.columns
    # Expected flagged rows: index 1 (FAIL), 2 (error); None row should be False
    flagged = result.select('flagged_for_review').to_series().to_list()
    assert flagged == [False, True, True, False]


def test_flag_for_review_empty_patterns_error():
    df = pd.DataFrame({'text': ['something']})
    with pytest.raises(ValueError):
        flag_for_review(df, subset=['text'], patterns=[])
