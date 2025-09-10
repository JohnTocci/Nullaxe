import pandas as pd
import polars as pl
import sys
import os

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions import remove_pii


def test_remove_pii_pandas_basic():
    df = pd.DataFrame({
        'text': [
            'Email: user@example.com, Phone: (555) 123-4567, SSN: 123-45-6789, URL: https://example.com',
            'No PII here',
            None,
            pd.NA
        ],
        'other': ['keep', 'these', 'values', 'intact']
    })

    result = remove_pii(df.copy(), subset=['text'])

    assert '[REDACTED_EMAIL]' in result.loc[0, 'text']
    assert '[REDACTED_PHONE]' in result.loc[0, 'text']
    assert '[REDACTED_SSN]' in result.loc[0, 'text']
    assert '[REDACTED_URL]' in result.loc[0, 'text']

    assert result.loc[1, 'text'] == 'No PII here'
    assert pd.isna(result.loc[2, 'text'])
    assert pd.isna(result.loc[3, 'text'])

    # Ensure other column unchanged
    assert result['other'].tolist() == ['keep', 'these', 'values', 'intact']


def test_remove_pii_pandas_multi_columns():
    df = pd.DataFrame({
        'col1': ['Contact: 555-123-4567 and john.doe@test.org'],
        'col2': ['Visit www.example.org and SSN 111-22-3333'],
        'col3': ['no pii']
    })

    result = remove_pii(df.copy(), subset=['col1', 'col2'])

    assert '[REDACTED_PHONE]' in result.loc[0, 'col1']
    assert '[REDACTED_EMAIL]' in result.loc[0, 'col1']
    assert '[REDACTED_URL]' in result.loc[0, 'col2']
    assert '[REDACTED_SSN]' in result.loc[0, 'col2']
    assert result.loc[0, 'col3'] == 'no pii'


def test_remove_pii_polars_basic():
    df = pl.DataFrame({
        'text': [
            'Email: user@example.com, Phone: +1 555 123 4567, SSN: 123.45.6789, URL: http://example.com',
            'No PII here'
        ]
    })

    result = remove_pii(df, subset=['text'])

    assert '[REDACTED_EMAIL]' in result[0, 'text']
    assert '[REDACTED_PHONE]' in result[0, 'text']
    assert '[REDACTED_SSN]' in result[0, 'text']
    assert '[REDACTED_URL]' in result[0, 'text']
    assert result[1, 'text'] == 'No PII here'


def test_remove_pii_subset_missing_columns_graceful():
    df = pd.DataFrame({'text': ['Contact me at 555-111-2222']})
    # Subset includes non-existent column; function should ignore it
    result = remove_pii(df.copy(), subset=['missing', 'text'])
    assert '[REDACTED_PHONE]' in result.loc[0, 'text']
