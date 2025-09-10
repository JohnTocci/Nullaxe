import pandas as pd
import polars as pl
from typing import Union, Dict, Optional

from ._clean_column_names import titlecase

DataFrameType = Union[pd.DataFrame, pl.DataFrame]


def format_for_display(
    df: DataFrameType,
    rules: Dict[str, Dict],
    column_case: Optional[str] = 'title'
) -> DataFrameType:
    """Apply presentation formatting rules to DataFrame columns.

    Supported rule specs per column (key = column name):
      - {'type': 'currency', 'symbol': '$', 'decimals': 2}
      - {'type': 'percentage', 'decimals': 1}
      - {'type': 'thousands'}
      - {'type': 'truncate', 'length': 50}
      - {'type': 'datetime', 'format': '%B %d, %Y'}

    This is intended as a final step before display; numeric columns will be
    converted to strings where formatting is applied.
    """
    if isinstance(df, pd.DataFrame):
        df_formatted = df.copy()

        for col, rule in rules.items():
            if col not in df_formatted.columns:
                continue
            rule_type = rule.get('type')

            if rule_type == 'currency':
                symbol = rule.get('symbol', '$')
                decimals = rule.get('decimals', 2)
                df_formatted[col] = df_formatted[col].map(
                    lambda x, s=symbol, d=decimals: f"{s}{x:,.{d}f}" if pd.notna(x) else x
                )
            elif rule_type == 'percentage':
                decimals = rule.get('decimals', 1)
                df_formatted[col] = df_formatted[col].map(
                    lambda x, d=decimals: f"{x * 100:.{d}f}%" if pd.notna(x) else x
                )
            elif rule_type == 'thousands':
                def _fmt_thousands(x):
                    if pd.isna(x):
                        return x
                    try:
                        if isinstance(x, float) and x.is_integer():
                            return f"{int(x):,}"
                        return f"{x:,}"
                    except Exception:
                        return x
                df_formatted[col] = df_formatted[col].map(_fmt_thousands)
            elif rule_type == 'truncate':
                length = rule.get('length', 50)
                df_formatted[col] = df_formatted[col].astype(str).apply(
                    lambda x, L=length: (x[:L] + '...') if len(x) > L else x
                )
            elif rule_type == 'datetime':
                date_format = rule.get('format', '%Y-%m-%d %H:%M:%S')
                df_formatted[col] = pd.to_datetime(df_formatted[col], errors='coerce').dt.strftime(date_format)

        if column_case == 'title':
            df_formatted = titlecase(df_formatted)
        return df_formatted

    if isinstance(df, pl.DataFrame):
        expressions = []
        original_columns = set(df.columns)

        for col, rule in rules.items():
            if col not in original_columns:
                continue
            rule_type = rule.get('type')

            if rule_type == 'currency':
                symbol = rule.get('symbol', '$')
                decimals = rule.get('decimals', 2)
                expressions.append(
                    pl.col(col).map_elements(
                        lambda x, s=symbol, d=decimals: None if x is None else f"{s}{x:,.{d}f}",
                        return_dtype=pl.Utf8
                    ).alias(col)
                )
            elif rule_type == 'percentage':
                decimals = rule.get('decimals', 1)
                expressions.append(
                    pl.col(col).map_elements(
                        lambda x, d=decimals: None if x is None else f"{x*100:.{d}f}%",
                        return_dtype=pl.Utf8
                    ).alias(col)
                )
            elif rule_type == 'thousands':
                def _pl_fmt_thousands(x):
                    if x is None:
                        return None
                    try:
                        if isinstance(x, float) and x.is_integer():
                            return f"{int(x):,}"
                        return f"{x:,}"
                    except Exception:
                        return x
                expressions.append(
                    pl.col(col).map_elements(_pl_fmt_thousands, return_dtype=pl.Utf8).alias(col)
                )
            elif rule_type == 'truncate':
                length = rule.get('length', 50)
                expressions.append(
                    pl.when(pl.col(col).cast(pl.Utf8).str.len_chars() > length)
                    .then(pl.col(col).cast(pl.Utf8).str.slice(0, length) + '...')
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            elif rule_type == 'datetime':
                date_format = rule.get('format', '%Y-%m-%d %H:%M:%S')
                def _dt_fmt(x, fmt=date_format):
                    if x is None:
                        return None
                    try:
                        ts = pd.to_datetime(x, errors='coerce')
                        if pd.isna(ts):
                            return None
                        return ts.strftime(fmt)
                    except Exception:
                        return None
                expressions.append(
                    pl.col(col).map_elements(_dt_fmt, return_dtype=pl.Utf8).alias(col)
                )

        df_formatted = df.with_columns(expressions) if expressions else df
        if column_case == 'title':
            df_formatted = titlecase(df_formatted)
        return df_formatted

    raise TypeError("Input must be a pandas or polars DataFrame.")
