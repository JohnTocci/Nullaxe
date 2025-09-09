import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add the src directory to the path to import sanex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanex.functions._clean_column_names import (
    snakecase, camelcase, pascalcase, kebabcase,
    titlecase, lowercase, screaming_snakecase, clean_column_names,
    _convert_to_snake_case, _convert_to_camel_case, _convert_to_pascal_case,
    _convert_to_kebab_case
)


class TestColumnNameCleaningFunctions:
    """Test column name cleaning functions."""

    def test_convert_to_snake_case(self):
        """Test snake case conversion function."""
        assert _convert_to_snake_case("FirstName") == "first_name"
        assert _convert_to_snake_case("LAST_NAME") == "last_name"
        assert _convert_to_snake_case("email-address") == "email_address"
        assert _convert_to_snake_case("Phone Number!") == "phone_number"
        assert _convert_to_snake_case("some__weird___column") == "some_weird_column"
        assert _convert_to_snake_case("123numeric") == "123numeric"
        assert _convert_to_snake_case("CamelCaseColumn") == "camel_case_column"
        assert _convert_to_snake_case("") == ""

    def test_convert_to_camel_case(self):
        """Test camel case conversion function."""
        assert _convert_to_camel_case("first_name") == "firstName"
        assert _convert_to_camel_case("LAST_NAME") == "lastName"
        assert _convert_to_camel_case("email-address") == "emailAddress"
        assert _convert_to_camel_case("Phone Number") == "phoneNumber"
        assert _convert_to_camel_case("some__weird___column") == "someWeirdColumn"
        assert _convert_to_camel_case("") == ""

    def test_convert_to_pascal_case(self):
        """Test Pascal case conversion function."""
        assert _convert_to_pascal_case("first_name") == "FirstName"
        assert _convert_to_pascal_case("LAST_NAME") == "LastName"
        assert _convert_to_pascal_case("email-address") == "EmailAddress"
        assert _convert_to_pascal_case("Phone Number") == "PhoneNumber"
        assert _convert_to_pascal_case("") == ""

    def test_convert_to_kebab_case(self):
        """Test kebab case conversion function."""
        assert _convert_to_kebab_case("FirstName") == "first-name"
        assert _convert_to_kebab_case("LAST_NAME") == "last-name"
        assert _convert_to_kebab_case("Phone Number") == "phone-number"
        assert _convert_to_kebab_case("CamelCaseColumn") == "camel-case-column"

    def test_snakecase_pandas(self, messy_column_names_df):
        """Test snakecase function with pandas DataFrame."""
        result = snakecase(messy_column_names_df)

        expected_columns = [
            'first_name', 'last_name', 'email_address', 'phone_number',
            'some_weird_column', 'numeric', 'camel_case_column', 'screaming_snake_case'
        ]

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_columns

    def test_camelcase_pandas(self, messy_column_names_df):
        """Test camelcase function with pandas DataFrame."""
        result = camelcase(messy_column_names_df)

        expected_columns = [
            'firstName', 'lastName', 'emailAddress', 'phoneNumber',
            'someWeirdColumn', 'numeric', 'camelCaseColumn', 'screamingSnakeCase'
        ]

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_columns

    def test_pascalcase_pandas(self, messy_column_names_df):
        """Test pascalcase function with pandas DataFrame."""
        result = pascalcase(messy_column_names_df)

        expected_columns = [
            'FirstName', 'LastName', 'EmailAddress', 'PhoneNumber',
            'SomeWeirdColumn', 'Numeric', 'CamelCaseColumn', 'ScreamingSnakeCase'
        ]

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_columns

    def test_kebabcase_pandas(self, messy_column_names_df):
        """Test kebabcase function with pandas DataFrame."""
        result = kebabcase(messy_column_names_df)

        expected_columns = [
            'first-name', 'last-name', 'email-address', 'phone-number',
            'some-weird-column', 'numeric', 'camel-case-column', 'screaming-snake-case'
        ]

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_columns

    def test_titlecase_pandas(self, messy_column_names_df):
        """Test titlecase function with pandas DataFrame."""
        result = titlecase(messy_column_names_df)

        expected_columns = [
            'First Name', 'Last Name', 'Email Address', 'Phone Number',
            'Some Weird Column', 'Numeric', 'Camel Case Column', 'Screaming Snake Case'
        ]

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_columns

    def test_lowercase_pandas(self, messy_column_names_df):
        """Test lowercase function with pandas DataFrame."""
        result = lowercase(messy_column_names_df)

        assert isinstance(result, pd.DataFrame)
        assert all(col.islower() or not col.isalpha() for col in result.columns)

    def test_screaming_snakecase_pandas(self, messy_column_names_df):
        """Test screaming_snakecase function with pandas DataFrame."""
        result = screaming_snakecase(messy_column_names_df)

        expected_columns = [
            'FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS', 'PHONE_NUMBER',
            'SOME_WEIRD_COLUMN', 'NUMERIC', 'CAMEL_CASE_COLUMN', 'SCREAMING_SNAKE_CASE'
        ]

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_columns

    def test_clean_column_names_pandas(self, messy_column_names_df):
        """Test clean_column_names function with different cases."""
        # Test default (snake case)
        result_snake = clean_column_names(messy_column_names_df)
        assert 'first_name' in result_snake.columns

        # Test camel case
        result_camel = clean_column_names(messy_column_names_df, case='camel')
        assert 'firstName' in result_camel.columns

        # Test pascal case
        result_pascal = clean_column_names(messy_column_names_df, case='pascal')
        assert 'FirstName' in result_pascal.columns

        # Test kebab case
        result_kebab = clean_column_names(messy_column_names_df, case='kebab')
        assert 'first-name' in result_kebab.columns

        # Test invalid case
        with pytest.raises(ValueError, match="Unsupported case format"):
            clean_column_names(messy_column_names_df, case='invalid')

    def test_polars_support(self):
        """Test that functions work with polars DataFrames."""
        polars_df = pl.DataFrame({
            'First Name': [1, 2, 3],
            'LAST_NAME': [4, 5, 6],
            'CamelCaseColumn': [7, 8, 9]
        })

        result = snakecase(polars_df)
        assert isinstance(result, pl.DataFrame)
        assert 'first_name' in result.columns
        assert 'last_name' in result.columns
        assert 'camel_case_column' in result.columns
