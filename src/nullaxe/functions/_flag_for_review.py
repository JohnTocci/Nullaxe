import pandas as pd
import polars as pl
import re
from typing import Union, List

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

def _extract_leading_flags(pattern: str):
    """Extract leading global inline flags like (?imxs) and return (sanitized_pattern, re_flags, flag_str).
    Only leading construct of the exact form (?letters) where letters subset of imsx is considered.
    """
    m = re.match(r'^\(\?([imsx]+)\)', pattern)
    if not m:
        return pattern, 0, ''
    flags_part = m.group(1)
    flag_value = 0
    for ch in flags_part:
        if ch == 'i':
            flag_value |= re.IGNORECASE
        elif ch == 'm':
            flag_value |= re.MULTILINE
        elif ch == 's':
            flag_value |= re.DOTALL
        elif ch == 'x':
            flag_value |= re.VERBOSE
    # Remove the leading flag group
    sanitized = pattern[m.end():]
    return sanitized, flag_value, flags_part

def flag_for_review(df: DataFrameType, subset: List[str], patterns: List[str]) -> DataFrameType:
    """
    Flags rows in the DataFrame for review if specified patterns are found in the given columns.
    Adds a new boolean column 'flagged_for_review' indicating whether any pattern was matched.

    Parameters:
    df (DataFrameType): Input pandas or polars DataFrame.
    subset (List[str]): Columns to check for patterns (silently ignores missing columns).
    patterns (List[str]): List of regex patterns to search for (each may include leading inline flags like (?i)).

    Returns:
    DataFrameType: DataFrame with an additional 'flagged_for_review' column.
    """
    if not patterns:
        raise ValueError("At least one pattern must be provided.")

    # Preprocess patterns to safely support leading inline flags
    processed_patterns = []  # list of tuples (sanitized_pattern, re_flags, original_inline_flags)
    for pat in patterns:
        sanitized, flags_value, flags_str = _extract_leading_flags(pat)
        processed_patterns.append((sanitized, flags_value, flags_str))

    if isinstance(df, pd.DataFrame):
        df['flagged_for_review'] = False
        for col in subset:
            if col in df.columns and df[col].dtype in ['object', 'string']:
                series = df[col].astype('string')  # ensure string accessor compatibility
                # Accumulate matches across all patterns for this column
                col_matches = pd.Series([False] * len(df), index=df.index)
                for sanitized, flags_value, _flagstr in processed_patterns:
                    try:
                        # Use pandas vectorized contains with per-pattern flags
                        matches = series.str.contains(sanitized, regex=True, flags=flags_value, na=False)
                    except re.error:
                        # If pattern invalid, skip it (could log); choose not to flag
                        continue
                    if matches.any():
                        col_matches = col_matches | matches
                if col_matches.any():
                    df.loc[col_matches, 'flagged_for_review'] = True
        return df

    elif isinstance(df, pl.DataFrame):
        flag_expr = pl.lit(False)
        for col in subset:
            if col in df.columns and df[col].dtype == pl.String:
                col_expr = pl.lit(False)
                for sanitized, flags_value, flags_str in processed_patterns:
                    pattern_for_polars = sanitized
                    # For polars we only explicitly handle IGNORECASE; other flags less common here
                    if flags_value & re.IGNORECASE:
                        # Prepend (?i) at start (allowed at start of entire regex)
                        pattern_for_polars = f"(?i){pattern_for_polars}"
                    try:
                        expr = pl.col(col).str.contains(pattern_for_polars)
                    except Exception:
                        continue
                    # Treat null as False before OR chaining
                    col_expr = col_expr | expr.fill_null(False)
                flag_expr = flag_expr | col_expr
        df = df.with_columns(flag_expr.alias('flagged_for_review'))
        return df

    raise TypeError("Input must be a pandas or polars DataFrame.")