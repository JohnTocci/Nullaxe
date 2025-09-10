import pandas as pd
import polars as pl
from typing import Union, Mapping

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def enforce_data_types(
    df: DataFrameType,
    dtype_map: Mapping[str, object],
    *,
    coerce: bool = False,           # coerce bad values to NA/null instead of raising
    missing_ok: bool = True         # skip keys not present
) -> DataFrameType:
    if isinstance(df, pd.DataFrame):
        # Filter to existing columns if desired
        mapping = {k: v for k, v in dtype_map.items() if k in df.columns or not missing_ok}
        # For pandas, this returns a new DataFrame (chainable)
        # Tip: use nullable dtypes like 'Int64' when NaNs are possible
        return df.astype(mapping, errors="ignore" if coerce else "raise")

    elif isinstance(df, pl.DataFrame):
        exprs = []
        for k, v in dtype_map.items():
            if missing_ok and k not in df.columns:
                continue
            # Normalize a few common shorthand types to Polars
            if isinstance(v, str):
                # map some pandas-like strings to Polars
                v = {
                    "float64": pl.Float64,
                    "float32": pl.Float32,
                    "int64": pl.Int64,
                    "int32": pl.Int32,
                    "string": pl.Utf8,
                    "utf8": pl.Utf8,
                    "bool": pl.Boolean,
                    "datetime64[ns]": pl.Datetime,  # naive datetime
                }.get(v, v)  # fall through if already an 'i64'/'f64' style
            exprs.append(pl.col(k).cast(v, strict=not coerce).alias(k))
        # Returns a new DataFrame (chainable)
        return df.with_columns(exprs)

    else:
        raise TypeError("Input must be a pandas or Polars DataFrame.")
