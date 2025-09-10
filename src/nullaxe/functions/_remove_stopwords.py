import pandas as pd
import polars as pl
from typing import Union, List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

# Lazy ensure NLTK resources are available
_DEF_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
]

def _ensure_nltk_resources():
    for path, pkg in _DEF_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:  # pragma: no cover (network dependent)
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # Ignore download failure; fallback logic will handle

def _tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        _ensure_nltk_resources()
        try:
            return word_tokenize(text)
        except Exception:
            # Final fallback: simple whitespace split
            return text.split()

def _remove_stopwords_from_text(text: str, stop_words: set) -> str:
    if pd.isna(text):
        return text
    words = _tokenize(str(text))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_stopwords(df: DataFrameType, subset: List[str], language: str = 'english') -> DataFrameType:
    """Removes stopwords from specified string columns.

    Parameters:
        df: Input pandas or polars DataFrame.
        subset: Columns to process (silently ignores missing columns).
        language: Stopword language (default 'english').

    Returns:
        DataFrame with stopwords removed in target columns.
    """
    _ensure_nltk_resources()
    stop_words = set(stopwords.words(language))

    if isinstance(df, pd.DataFrame):
        for col in subset:
            if col in df.columns and df[col].dtype in ['object', 'string']:
                df[col] = df[col].apply(lambda x: _remove_stopwords_from_text(x, stop_words))
        return df

    elif isinstance(df, pl.DataFrame):
        df_copy = df.clone()
        for col in subset:
            if col in df_copy.columns and df_copy[col].dtype == pl.String:
                processed = [
                    _remove_stopwords_from_text(val, stop_words) for val in df_copy[col].to_list()
                ]
                df_copy = df_copy.with_columns(pl.Series(name=col, values=processed))
        return df_copy

    raise TypeError("Input must be a pandas or polars DataFrame.")