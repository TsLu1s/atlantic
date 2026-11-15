"""
Atlantic - Column Type Utilities.

Single source of truth for deciding whether a column is numeric, categorical,
boolean or datetime. Every other module (mixins, pipeline, encoders, imputers,
feature selectors) delegates here so the classification never drifts between
call sites.

Detection is done per dtype with ``pandas.api.types`` predicates rather than
``DataFrame.select_dtypes`` with hard-coded dtype names, because name lists
silently miss perfectly valid columns:

* unsigned integers (``uint8`` ...) and nullable ``Int64`` / ``Float64``;
* the pandas >= 3 default string dtype (``str``) and the ``string`` dtype;
* datetimes that are not nanosecond resolution (pandas >= 3 defaults to
  microseconds) and timezone-aware datetimes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

# Legacy dtype-name lists. Kept for backwards compatibility with code that
# imports them; detection itself no longer relies on these (see module
# docstring) and uses the ``is_*_column`` predicates below instead.
NUMERIC_TYPES = ["int", "int8", "int16", "int32", "int64", "float", "float16", "float32", "float64"]
CATEGORICAL_TYPES = ["object", "category"]
DATETIME_TYPES = ["datetime64[ns]", "datetime64"]


# -----------------------------------------------------------------------------
# dtype-level predicates
# -----------------------------------------------------------------------------


def _is_categorical_dtype(dtype: Any) -> bool:
    return isinstance(dtype, pd.CategoricalDtype)


def _is_boolean(dtype: Any) -> bool:
    # A categorical whose categories are booleans reports as a bool dtype in
    # pandas; it is a categorical for our purposes.
    if _is_categorical_dtype(dtype):
        return False
    return bool(is_bool_dtype(dtype))


def _is_numeric(dtype: Any) -> bool:
    if _is_categorical_dtype(dtype):
        return False
    return (
        bool(is_numeric_dtype(dtype))
        and not bool(is_bool_dtype(dtype))
        and not bool(is_complex_dtype(dtype))
    )


def _is_datetime(dtype: Any) -> bool:
    return bool(is_datetime64_any_dtype(dtype))


def _is_categorical(dtype: Any) -> bool:
    if _is_categorical_dtype(dtype):
        return True
    return bool(is_object_dtype(dtype)) or bool(is_string_dtype(dtype))


# -----------------------------------------------------------------------------
# Series-level predicates
# -----------------------------------------------------------------------------


def is_numeric_column(series: pd.Series) -> bool:
    """True for real-valued numeric columns.

    Includes every integer width, unsigned integers, floats and the nullable
    ``Int64`` / ``Float64`` extension dtypes. Excludes booleans, complex
    numbers, timedeltas, datetimes and categoricals.
    """
    return _is_numeric(series.dtype)


def is_boolean_column(series: pd.Series) -> bool:
    """True for ``bool`` and nullable ``boolean`` columns."""
    return _is_boolean(series.dtype)


def is_categorical_column(series: pd.Series) -> bool:
    """True for ``object``, ``category`` and any string dtype column."""
    return _is_categorical(series.dtype)


def is_datetime_column(series: pd.Series) -> bool:
    """True for datetime columns of any resolution, timezone-aware or naive."""
    return _is_datetime(series.dtype)


def is_numeric_target(series: pd.Series) -> bool:
    """
    Single source of truth for deciding whether a target column is numeric
    (regression) or not (classification).

    Nullable extension dtypes (``Int64``/``Float64``), narrow widths
    (``int8``/``int16``/``float16``) and unsigned integers are all numeric.
    Booleans (``bool`` and nullable ``boolean``) are treated as non-numeric,
    i.e. as a classification target.

    Returns:
        True if the target should be treated as a regression target.
    """
    return _is_numeric(series.dtype)


# -----------------------------------------------------------------------------
# DataFrame-level helpers
# -----------------------------------------------------------------------------


def _columns_where(X: pd.DataFrame, predicate, exclude: Optional[List[str]]) -> List[str]:
    excluded = set(exclude or [])
    return [col for col, dtype in X.dtypes.items() if col not in excluded and predicate(dtype)]


def get_numeric_columns(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of numeric columns.

    Returns:
        List of numeric column names.
    """
    return _columns_where(X, _is_numeric, exclude)


def get_categorical_columns(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of categorical columns (object, category and string dtypes).

    Returns:
        List of categorical column names.
    """
    return _columns_where(X, _is_categorical, exclude)


def get_boolean_columns(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of boolean columns (``bool`` and nullable ``boolean``).

    Returns:
        List of boolean column names.
    """
    return _columns_where(X, _is_boolean, exclude)


def get_datetime_columns(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of datetime columns (any resolution, timezone-aware or naive).

    Returns:
        List of datetime column names.
    """
    return _columns_where(X, _is_datetime, exclude)


def get_columns_by_null_percentage(X: pd.DataFrame, threshold: float = 0.5) -> List[str]:
    """
    Get columns with null percentage above threshold.

    Returns:
        List of column names with high null percentage.
    """
    if len(X) == 0:
        return []
    null_percentages = X.isnull().sum() / len(X)
    return null_percentages[null_percentages > threshold].index.tolist()


def get_constant_columns(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get columns with only one unique value.

    Returns:
        List of constant column names.
    """
    exclude = exclude or []
    return [col for col in X.columns if col not in exclude and X[col].nunique() == 1]


def get_unique_columns(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get columns where all values are unique (potential IDs).

    Returns:
        List of all-unique column names.
    """
    exclude = exclude or []
    return [col for col in X.columns if col not in exclude and X[col].nunique() == len(X)]


def get_high_cardinality_columns(
    X: pd.DataFrame, threshold: int = 50, exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get categorical columns with high cardinality.

    Returns:
        List of high-cardinality column names.
    """
    exclude = exclude or []
    cat_cols = get_categorical_columns(X, exclude)

    return [col for col in cat_cols if X[col].nunique() >= threshold]


def separate_columns_by_type(X: pd.DataFrame, target: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Separate columns by type.

    Returns:
        Dictionary with 'numeric', 'categorical', 'datetime' keys.
    """
    exclude = [target] if target else []

    return {
        "numeric": get_numeric_columns(X, exclude),
        "categorical": get_categorical_columns(X, exclude),
        "datetime": get_datetime_columns(X),
    }
