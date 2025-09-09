import pytest
import pandas as pd
import polars as pl
import numpy as np
from typing import Union

# Test data fixtures
@pytest.fixture
def sample_pandas_df():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'First Name': ['John', 'Jane', 'Bob', None, 'Alice'],
        'LastName': ['Doe', 'Smith', 'Johnson', 'Brown', None],
        'Age': [25, 30, None, 35, 28],
        'Email Address': ['john@email.com', 'invalid-email', 'bob@test.org', None, 'alice@company.co.uk'],
        'Phone Number': ['123-456-7890', '(555) 123-4567', 'not a phone', '+1-800-555-0199', None],
        'Salary': ['$50,000', '60000', '$75,000.50', None, '45k'],
        'Description': ['  Text with spaces  ', 'UPPER CASE TEXT', 'mixed Case Text', '', '   '],
        'Boolean Col': ['yes', 'no', 'true', 'false', 'Y'],
        'Duplicate Col': [1, 1, 1, 1, 1],  # Single value column
        'Outlier Col': [1, 2, 3, 4, 1000]  # Contains outlier
    })

@pytest.fixture
def sample_polars_df():
    """Create a sample polars DataFrame for testing."""
    return pl.DataFrame({
        'First Name': ['John', 'Jane', 'Bob', None, 'Alice'],
        'LastName': ['Doe', 'Smith', 'Johnson', 'Brown', None],
        'Age': [25, 30, None, 35, 28],
        'Email Address': ['john@email.com', 'invalid-email', 'bob@test.org', None, 'alice@company.co.uk'],
        'Phone Number': ['123-456-7890', '(555) 123-4567', 'not a phone', '+1-800-555-0199', None],
        'Salary': ['$50,000', '60000', '$75,000.50', None, '45k'],
        'Description': ['  Text with spaces  ', 'UPPER CASE TEXT', 'mixed Case Text', '', '   '],
        'Boolean Col': ['yes', 'no', 'true', 'false', 'Y'],
        'Duplicate Col': [1, 1, 1, 1, 1],  # Single value column
        'Outlier Col': [1, 2, 3, 4, 1000]  # Contains outlier
    })

@pytest.fixture
def messy_column_names_df():
    """DataFrame with messy column names for testing."""
    return pd.DataFrame({
        'First Name': [1, 2, 3],
        'LAST_NAME': [4, 5, 6],
        'email-address': [7, 8, 9],
        'Phone Number!': [10, 11, 12],
        'some__weird___column': [13, 14, 15],
        '123numeric': [16, 17, 18],
        'CamelCaseColumn': [19, 20, 21],
        'SCREAMING_SNAKE_CASE': [22, 23, 24]
    })

@pytest.fixture(params=['pandas', 'polars'])
def df_fixture(request, sample_pandas_df, sample_polars_df):
    """Parametrized fixture that returns both pandas and polars DataFrames."""
    if request.param == 'pandas':
        return sample_pandas_df
    else:
        return sample_polars_df
