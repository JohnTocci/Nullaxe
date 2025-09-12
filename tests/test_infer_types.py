import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add src to path (consistent with other tests)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._infer_types import infer_types  # noqa: E402


class TestInferTypesPandas:
    def test_numeric_and_int_float_inference(self):
        df = pd.DataFrame({
            'ints': ['1', '2', '3', None],
            'floats': ['1.1', '2.5', '3.0', 'bad'],
        })
        out = infer_types(df.copy())
        assert str(out['ints'].dtype) == 'Int64'
        assert str(out['floats'].dtype) == 'Float64'
        # Non-numeric token coerced to NaN
        assert pd.isna(out.loc[3, 'floats'])

    def test_datetime_inference(self):
        df = pd.DataFrame({
            'dates': ['2024-01-01', '2024-02-02', 'not a date', None],
            'mixed': ['10', '20', 'x', 'y']
        })
        out = infer_types(df.copy())
        assert pd.api.types.is_datetime64_any_dtype(out['dates'])
        # 'mixed' mostly numeric? 2/4 = 0.5 < default 0.6 -> stays object
        assert out['mixed'].dtype == object

    def test_boolean_inference(self):
        df = pd.DataFrame({
            'bools': ['True', 'false', 'YES', 'no', '1', '0', None]
        })
        out = infer_types(df.copy())
        assert str(out['bools'].dtype) == 'boolean'
        assert out['bools'].sum() == 3  # True, YES, 1

    def test_category_inference(self):
        # Create many rows with few unique values so unique/rows <= 0.05
        vals = ['A'] * 90 + ['B'] * 10
        df = pd.DataFrame({'cat_like': vals})
        out = infer_types(df.copy())
        assert str(out['cat_like'].dtype) == 'category'

    def test_subset_and_inplace_false(self):
        df = pd.DataFrame({
            'num': ['1', '2', '3'],
            'text': ['a', 'b', 'c']
        })
        original = df.copy()
        out = infer_types(df, subset=['num'], inplace=False)
        # num converted, text untouched
        assert str(out['num'].dtype) == 'Int64'
        assert out['text'].dtype == object
        # inplace False means original unchanged
        assert df.equals(original)

    def test_type_error_invalid_input(self):
        with pytest.raises(TypeError):
            infer_types(['not', 'a', 'df'])  # type: ignore

    def test_threshold_prevents_conversion(self):
        df = pd.DataFrame({'maybe_num': ['1', 'x', '2', 'y']})  # 2/4 numeric = 0.5
        out = infer_types(df.copy(), numeric_threshold=0.75)
        assert out['maybe_num'].dtype == object  # not converted


class TestInferTypesPolars:
    def test_polars_numeric_and_datetime(self):
        df = pl.DataFrame({
            'ints': ['1', '2', '3', None],
            'floats': ['1.2', '3.4', 'x', None],
            'dates': ['2024-01-01', '2024-02-02', 'bad', None]
        })
        out = infer_types(df)
        # Ints should become Int64
        assert out['ints'].dtype == pl.Int64
        # floats: attempt int fails then float succeeds
        assert out['floats'].dtype == pl.Float64
        # dates should be datetime
        assert out['dates'].dtype == pl.Datetime

    def test_polars_boolean_and_category(self):
        bool_tokens = ['true', 'False', 'YES', 'no', '1', '0']
        bools = (bool_tokens * 8) + bool_tokens[:2]  # 6*8=48 +2 = 50
        assert len(bools) == 50
        cat_vals = ['A'] * 48 + ['B'] * 2  # 2 uniques / 50 rows = 0.04
        df = pl.DataFrame({
            'bools': bools,
            'cat_like': cat_vals,
        })
        out = infer_types(df)
        assert out['bools'].dtype == pl.Boolean
        assert out['cat_like'].dtype == pl.Categorical

    def test_polars_subset(self):
        df = pl.DataFrame({
            'num': ['1', '2', '3'],
            'text': ['x', 'y', 'z']
        })
        out = infer_types(df, subset=['num'])
        assert out['num'].dtype in (pl.Int64, pl.Int32)
        assert out['text'].dtype == pl.Utf8

    def test_polars_threshold_block(self):
        df = pl.DataFrame({'maybe': ['1', 'x', '2', 'y']})  # 2/4 numeric
        out = infer_types(df, numeric_threshold=0.75)
        # stays Utf8
        assert out['maybe'].dtype == pl.Utf8
