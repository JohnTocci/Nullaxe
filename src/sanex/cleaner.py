from .functions import(
snakecase, camelcase, pascalcase,
kebabcase, titlecase, lowercase,
screaming_snakecase, clean_column_names,
remove_duplicates)
import pandas as pd
import polars as pl
from typing import Union

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

class Sanex:
    def __init__(self, df):
        if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError("Input must be a pandas or polars DataFrame.")
        self._df = df


    def clean_column_names(self, case: str = 'snake'):
                """
                Cleans the column names of the DataFrame.

                Args:
                    case (str): The desired case format for the column names.
                                Defaults to 'snake'. Supported formats include:
                                'snake', 'camel', 'pascal', 'kebab', 'title',
                                'lower', and 'screaming_snake'.

                Returns:
                    Sanex: The instance of the class to allow method chaining.

                This is a chainable method.
                """
                self._df = clean_column_names(self._df, case=case)
                return self

    def remove_duplicates(self):
        """
        Removes duplicate rows and columns from the DataFrame.

        Returns:
            Sanex: The instance of the class to allow method chaining.

        This is a chainable method.
        """
        self._df = remove_duplicates(self._df)
        return self

    def snakecase(self):
        """
        Converts all column names in the DataFrame to snake_case.

        This is a chainable method.
        """
        self._df = snakecase(self._df)
        return self

    def camelcase(self):
        """
        Converts all column names in the DataFrame to camelCase.

        This is a chainable method.
        """
        self._df = camelcase(self._df)
        return self

    def pascalcase(self):
        """
        Converts all column names in the DataFrame to PascalCase.

        This is a chainable method.
        """
        self._df = pascalcase(self._df)
        return self

    def kebabcase(self):
        """
        Converts all column names in the DataFrame to kebab-case.

        This is a chainable method.
        """
        self._df = kebabcase(self._df)
        return self

    def titlecase(self):
        """
        Converts all column names in the DataFrame to Title Case.

        This is a chainable method.
        """
        self._df = titlecase(self._df)
        return self

    def lowercase(self):
        """
        Converts all column names in the DataFrame to lowercase.

        This is a chainable method.
        """
        self._df = lowercase(self._df)
        return self

    def screaming_snakecase(self):
        """
        Converts all column names in the DataFrame to SCREAMING_SNAKE_CASE.

        This is a chainable method.
        """
        self._df = screaming_snakecase(self._df)
        return self

    def to_df(self) -> DataFrameType:
        """
        Returns the final, cleaned DataFrame.
        """
        return self._df

