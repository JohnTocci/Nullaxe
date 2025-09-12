import pandas as pd
import polars as pl
from typing import Union, List, Optional

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def _all_int_like(series: pd.Series) -> bool:
    """Check if all non-null values in a pandas Series are integer-like."""
    if series.empty:
        return False
    return (series.astype(float) % 1 == 0).all()

def infer_types(
    df: DataFrameType,
    subset: Optional[List[str]] = None,
    numeric_threshold: float = 0.6,
    datetime_threshold: float = 0.6,
    category_unique_ratio: float = 0.05,
    inplace: bool = True,
) -> DataFrameType:
    """
    Infer and cast column types for pandas or polars DataFrames.

    Order tried per column: datetime -> numeric -> boolean -> category.

    Parameters:
    ----------
    df : DataFrameType
        Input pandas or polars DataFrame.
    subset : List[str], optional
        Columns to process (default all columns).
    numeric_threshold : float
        Fraction of non-null values that must parse as numeric to cast.
    datetime_threshold : float
        Fraction of non-null values that must parse as datetime to cast.
    category_unique_ratio : float
        If (n_unique / non_null) <= this ratio, cast to category.
    inplace : bool
        For pandas: if False operate on a copy.

    Returns:
    -------
    DataFrameType
        DataFrame with inferred types.
    """
    # Pandas branch
    if isinstance(df, pd.DataFrame):
        if not inplace:
            df = df.copy()
        cols = list(df.columns) if subset is None else [c for c in subset if c in df.columns]
        
        for col in cols:
            s = df[col]
            non_null = s.dropna()
            if non_null.empty:
                continue
                
            # 1) DATETIME
            parsed_dt = pd.to_datetime(non_null, errors="coerce")
            if (parsed_dt.notna().sum() / len(non_null)) >= datetime_threshold:
                df[col] = pd.to_datetime(s, errors="coerce")
                continue
                
            # 2) NUMERIC
            parsed_num = pd.to_numeric(non_null, errors="coerce")
            if (parsed_num.notna().sum() / len(non_null)) >= numeric_threshold:
                full_num = pd.to_numeric(s, errors="coerce")
                non_null_full = full_num.dropna()
                if _all_int_like(non_null_full):
                    df[col] = full_num.astype("Int64")
                else:
                    df[col] = full_num.astype("Float64")
                continue
                
            # 3) BOOLEAN
            lowered = non_null.astype(str).str.strip().str.lower()
            bool_map = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
            mapped = lowered.map(bool_map)
            if (mapped.notna().sum() / len(non_null)) >= 0.95:
                df[col] = s.astype(str).str.strip().str.lower().map(bool_map).astype("boolean")
                continue
                
            # 4) CATEGORY
            if (non_null.nunique() / len(non_null)) <= category_unique_ratio:
                df[col] = s.astype("category")
                
        return df

    # Polars branch
    if isinstance(df, pl.DataFrame):
        cols = df.columns if subset is None else [c for c in subset if c in df.columns]
        total = df.height
        
        for col in cols:
            series = df[col]
            non_null = total - series.null_count()
            if non_null == 0:
                continue
                
            # 1) DATETIME (ISO format first, then generic)
            dt_candidate = None
            if series.dtype == pl.Utf8:
                # Try ISO date format first
                try:
                    iso = series.str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
                    iso_successes = total - iso.null_count()
                    if iso_successes / non_null >= datetime_threshold:
                        dt_candidate = iso
                except Exception:
                    pass
                    
                # If ISO didn't work, try generic parsing
                if dt_candidate is None:
                    try:
                        generic = series.str.strptime(pl.Datetime, strict=False)
                        gen_successes = total - generic.null_count()
                        if gen_successes / non_null >= datetime_threshold:
                            dt_candidate = generic
                    except Exception:
                        pass
            else:
                # For non-string types, try direct casting
                try:
                    cast_dt = series.cast(pl.Datetime, strict=False)
                    cast_successes = total - cast_dt.null_count()
                    if cast_successes / non_null >= datetime_threshold:
                        dt_candidate = cast_dt
                except Exception:
                    pass
                    
            if dt_candidate is not None:
                df = df.with_columns(dt_candidate.alias(col))
                continue
                
            # 2) NUMERIC INT (skip if decimals present in sample)
            has_decimal = False
            if series.dtype == pl.Utf8:
                try:
                    sample = [v for v in series.drop_nulls().head(50).to_list() if isinstance(v, str)]
                    has_decimal = any(("." in v) or ("e" in v.lower()) for v in sample)
                except Exception:
                    has_decimal = False
                    
            if not has_decimal:
                try:
                    int_cast = series.cast(pl.Int64, strict=False)
                    if (total - int_cast.null_count()) / non_null >= numeric_threshold:
                        df = df.with_columns(int_cast.alias(col))
                        continue
                except Exception:
                    pass
                    
            # 2b) NUMERIC FLOAT
            try:
                float_cast = series.cast(pl.Float64, strict=False)
                if (total - float_cast.null_count()) / non_null >= numeric_threshold:
                    df = df.with_columns(float_cast.alias(col))
                    continue
            except Exception:
                pass
                
            # 3) BOOLEAN
            if series.dtype == pl.Utf8:
                lower_expr = pl.col(col).str.to_lowercase()
                bool_expr = (
                    pl.when(lower_expr.is_in(["true", "1", "yes"]))
                    .then(True)
                    .when(lower_expr.is_in(["false", "0", "no"]))
                    .then(False)
                    .otherwise(None)
                )
                temp = df.select(bool_expr.alias("__tmp_bool__"))["__tmp_bool__"]
                if (total - temp.null_count()) / non_null >= 0.95:
                    df = df.with_columns(bool_expr.alias(col).cast(pl.Boolean))
                    continue
                    
            # 4) CATEGORY
            try:
                uniques = df.select(pl.col(col).n_unique()).to_series()[0]
                if uniques / max(1, non_null) <= category_unique_ratio:
                    df = df.with_columns(pl.col(col).cast(pl.Categorical).alias(col))
                    continue
            except Exception:
                pass
                
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")
