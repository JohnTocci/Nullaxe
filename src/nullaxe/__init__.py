from .cleaner import Nullaxe
from .functions import *  # noqa
import sys

__version__ = "0.5.0"
__author__ = "John Tocci"
__email__ = "john@johntocci.com"
__license__ = "MIT"
__description__ = "A data cleaning library for pandas and polars DataFrames."

class _NullaxeModule:
    def __call__(self, df):
        return Nullaxe(df)

    def __getattr__(self, name):
        # Allow accessing functions like nlx.snakecase
        return globals()[name]

sys.modules[__name__] = _NullaxeModule()
