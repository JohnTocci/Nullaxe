from .cleaner import Sanex
from .functions import *
import sys

__version__ = "0.2.1"
__author__ = "John Tocci"
__email__ = "john@johntocci.com"
__license__ = "MIT"
__description__ = "A data cleaning library for pandas and polars DataFrames."

class _SanexModule:
    def __call__(self, df):
        return Sanex(df)

    def __getattr__(self, name):
        # This allows accessing functions like sx.snakecase
        # by forwarding the attribute access to the module.
        return globals()[name]

sys.modules[__name__] = _SanexModule()

