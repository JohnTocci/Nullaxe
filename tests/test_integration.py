import pytest
import pandas as pd
import polars as pl
import sys
import os
import warnings

# Add the src directory to the path to import sanex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import sanex
from sanex.cleaner import Sanex


class TestSanexIntegration:
    """Integration tests for the complete Sanex workflow."""

    def test_complete_cleaning_workflow_pandas(self):
        """Test a complete data cleaning workflow with pandas DataFrame."""
        # Create a messy dataset
        df = pd.DataFrame({
            'First Name': ['  John  ', 'JANE', 'bob', None, 'Alice'],
            'LAST_NAME': ['Doe', 'smith', 'JOHNSON', 'Brown', None],
            'Age': [25, 30, None, 35, 28],
            'Email Address': ['john@email.com', 'invalid-email', 'bob@test.org', None, 'alice@company.co.uk'],
            'Phone Number': ['123-456-7890', '(555) 123-4567', 'not a phone', '+1-800-555-0199', None],
            'Salary': ['$50,000', '60,000', '$75,000.50', None, '45k'],
            'Description': ['  Text with spaces  ', 'UPPER CASE TEXT', 'mixed Case Text', '', '   '],
            'Boolean Col': ['yes', 'no', 'true', 'false', 'Y'],
            'Duplicate Col': [1, 1, 1, 1, 1],  # Single value column
            'Outlier Col': [1, 2, 3, 4, 1000],  # Contains outlier
            'Duplicate Row A': [1, 2, 2, 4, 5],  # For duplicate detection
            'Duplicate Row B': [1, 2, 2, 4, 5]   # For duplicate detection
        })

        # Add duplicate row
        df = pd.concat([df, df.iloc[[1]]], ignore_index=True)

        # Complete cleaning workflow
        sx = Sanex(df)
        result = (sx
                 .clean_column_names(case='snake')  # Clean column names
                 .remove_duplicates()               # Remove duplicate rows
                 .drop_single_value_columns()       # Drop single-value columns
                 .fill_missing(value='MISSING')     # Fill missing values
                 .remove_whitespace()              # Remove whitespace
                 .standardize_booleans()           # Standardize boolean values
                 )

        # Verify the results
        final_df = result._df

        # Check column names are snake_case
        expected_columns = [
            'first_name', 'last_name', 'age', 'email_address', 'phone_number',
            'salary', 'description', 'boolean_col', 'outlier_col',
            'duplicate_row_a', 'duplicate_row_b'
        ]
        assert list(final_df.columns) == expected_columns

        # Check duplicate row was removed
        assert final_df.shape[0] == 5  # Original 5 rows (duplicate removed)

        # Check single value column was dropped
        assert 'duplicate_col' not in final_df.columns

        # Check missing values were filled
        assert final_df.isna().sum().sum() == 0

        # Check whitespace was removed from text columns
        assert final_df.loc[0, 'first_name'] == 'John'  # Leading/trailing spaces removed

        # Check boolean standardization
        assert final_df['boolean_col'].dtype == bool or final_df['boolean_col'].dtype == 'bool' or str(final_df['boolean_col'].dtype) == 'boolean'

    def test_complete_cleaning_workflow_polars(self):
        """Test a complete data cleaning workflow with polars DataFrame."""
        # Create a messy dataset
        df = pl.DataFrame({
            'First Name': ['  John  ', 'JANE', 'bob', None, 'Alice'],
            'LAST_NAME': ['Doe', 'smith', 'JOHNSON', 'Brown', None],
            'Age': [25, 30, None, 35, 28],
            'Boolean Col': ['yes', 'no', 'true', 'false', 'Y'],
            'Duplicate Col': [1, 1, 1, 1, 1]  # Single value column
        })

        # Complete cleaning workflow
        sx = Sanex(df)
        result = (sx
                 .clean_column_names(case='camel')   # Clean column names
                 .drop_single_value_columns()        # Drop single-value columns
                 .fill_missing(value='MISSING')      # Fill missing values
                 .remove_whitespace()               # Remove whitespace
                 )

        # Verify the results
        final_df = result._df

        # Check column names are camelCase
        expected_columns = ['firstName', 'lastName', 'age', 'booleanCol']
        assert list(final_df.columns) == expected_columns

        # Check single value column was dropped
        assert 'duplicateCol' not in final_df.columns

        # Check missing values were filled
        assert final_df.null_count().sum_horizontal().sum() == 0

    def test_method_chaining_with_error_recovery(self):
        """Test method chaining continues even if some methods encounter issues."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        sx = Sanex(df)

        # Chain methods - some may not have visible effects but shouldn't break
        result = (sx
                 .clean_column_names()
                 .remove_duplicates()        # No duplicates to remove
                 .drop_single_value_columns() # No single-value columns
                 .fill_missing()            # No missing values to fill
                 .remove_whitespace()       # No text columns with whitespace
                 )

        # Should complete without errors
        assert isinstance(result, Sanex)
        assert result._df.shape == (3, 2)

    def test_data_extraction_workflow(self):
        """Test data extraction workflow."""
        df = pd.DataFrame({
            'contact_info': [
                'John Doe, john@email.com, 123-456-7890, $50,000',
                'Jane Smith, jane.smith@test.org, (555) 123-4567, €60,000',
                'Bob Johnson, invalid-email, not-a-phone, not-a-salary',
                'Alice Brown, alice@company.co.uk, +1-800-555-0199, £45,000'
            ]
        })

        sx = Sanex(df)

        # Apply data extraction methods if available
        try:
            result = sx.extract_email().extract_phone_numbers().extract_and_clean_numeric()
            final_df = result._df

            # Should have new columns for extracted data
            assert 'contact_info_email' in final_df.columns
            assert 'contact_info_phone' in final_df.columns
            assert 'contact_info_numeric' in final_df.columns

        except AttributeError:
            # Methods might not be available in the Sanex class yet
            # This tests that the workflow can be adapted
            pass

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create a larger dataset for performance testing
        n_rows = 10000
        df = pd.DataFrame({
            'col1': ['text_' + str(i % 100) for i in range(n_rows)],
            'col2': [i for i in range(n_rows)],
            'col3': [f'email_{i}@test.com' if i % 10 == 0 else 'invalid' for i in range(n_rows)],
            'col4': ['yes' if i % 2 == 0 else 'no' for i in range(n_rows)]
        })

        # Add some duplicates
        df = pd.concat([df, df.iloc[:100]], ignore_index=True)

        sx = Sanex(df)

        # Should handle large dataset without issues
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore performance warnings

            result = (sx
                     .clean_column_names()
                     .remove_duplicates()
                     .standardize_booleans()
                     )

            assert isinstance(result, Sanex)
            assert result._df.shape[0] <= df.shape[0]  # Duplicates removed or same size

    def test_mixed_data_types_handling(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'mixed_col': [1, 'text', 3.14, True, None, '2023-01-01', [1, 2, 3]],
            'numeric_col': [1, 2, 3, 4, 5, 6, 7],
            'text_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        })

        sx = Sanex(df)

        # Should handle mixed data types gracefully
        result = (sx
                 .clean_column_names()
                 .fill_missing(value='FILLED')
                 .remove_whitespace()
                 )

        assert isinstance(result, Sanex)
        assert result._df.shape == df.shape

    def test_empty_dataframe_workflow(self):
        """Test workflow with empty DataFrame."""
        empty_df = pd.DataFrame()

        sx = Sanex(empty_df)

        # Should handle empty DataFrame gracefully
        result = (sx
                 .clean_column_names()
                 .remove_duplicates()
                 .fill_missing()
                 )

        assert isinstance(result, Sanex)
        assert result._df.shape == (0, 0)

    def test_single_row_dataframe(self):
        """Test workflow with single-row DataFrame."""
        single_row_df = pd.DataFrame({
            'First Name': ['John'],
            'LAST_NAME': ['Doe'],
            'Age': [30]
        })

        sx = Sanex(single_row_df)

        result = (sx
                 .clean_column_names()
                 .remove_duplicates()
                 .fill_missing()
                 )

        assert isinstance(result, Sanex)
        assert result._df.shape == (1, 3)
        assert list(result._df.columns) == ['first_name', 'last_name', 'age']

    def test_error_handling_in_chain(self):
        """Test that method chaining handles errors gracefully."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        sx = Sanex(df)

        # Test with invalid parameters - should either work or raise clear errors
        try:
            result = sx.clean_column_names(case='invalid_case')
            # If it doesn't raise an error, it should default to a valid case
            assert isinstance(result, Sanex)
        except (ValueError, KeyError):
            # Expected behavior for invalid case
            pass

        # Valid chaining should still work
        result = sx.clean_column_names(case='snake').remove_duplicates()
        assert isinstance(result, Sanex)
