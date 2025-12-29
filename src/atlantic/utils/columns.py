from __future__ import annotations

from typing import List, Optional

import pandas as pd


NUMERIC_TYPES = ['int', 'int8', 'int16', 'int32', 'int64', 
                 'float', 'float16', 'float32', 'float64']
CATEGORICAL_TYPES = ['object', 'category']
DATETIME_TYPES = ['datetime64[ns]', 'datetime64']


def get_numeric_columns(
    X: pd.DataFrame,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of numeric columns.

    Returns:
        List of numeric column names.
    """
    exclude = exclude or []
    return [
        col for col in X.select_dtypes(include=NUMERIC_TYPES).columns
        if col not in exclude
    ]


def get_categorical_columns(
    X: pd.DataFrame,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of categorical columns.

    Returns:
        List of categorical column names.
    """
    exclude = exclude or []
    return [
        col for col in X.select_dtypes(include=CATEGORICAL_TYPES).columns
        if col not in exclude
    ]


def get_datetime_columns(X: pd.DataFrame) -> List[str]:
    """
    Get list of datetime columns.

    Returns:
        List of datetime column names.
    """
    return X.select_dtypes(include=['datetime64[ns]']).columns.tolist()


def get_columns_by_null_percentage(
    X: pd.DataFrame,
    threshold: float = 0.5
) -> List[str]:
    """
    Get columns with null percentage above threshold.
 
    Returns:
        List of column names with high null percentage.
    """
    null_percentages = X.isnull().sum() / len(X)
    return null_percentages[null_percentages > threshold].index.tolist()


def get_constant_columns(
    X: pd.DataFrame,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get columns with only one unique value.

    Returns:
        List of constant column names.
    """
    exclude = exclude or []
    return [
        col for col in X.columns
        if col not in exclude and X[col].nunique() == 1
    ]


def get_unique_columns(
    X: pd.DataFrame,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get columns where all values are unique (potential IDs).

    Returns:
        List of all-unique column names.
    """
    exclude = exclude or []
    return [
        col for col in X.columns
        if col not in exclude and X[col].nunique() == len(X)
    ]


def get_high_cardinality_columns(
    X: pd.DataFrame,
    threshold: int = 50,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get categorical columns with high cardinality.

    Returns:
        List of high-cardinality column names.
    """
    exclude = exclude or []
    cat_cols = get_categorical_columns(X, exclude)
    
    return [
        col for col in cat_cols
        if X[col].nunique() >= threshold
    ]


def separate_columns_by_type(
    X: pd.DataFrame,
    target: Optional[str] = None
) -> dict:
    """
    Separate columns by type.

    Returns:
        Dictionary with 'numeric', 'categorical', 'datetime' keys.
    """
    exclude = [target] if target else []
    
    return {
        'numeric': get_numeric_columns(X, exclude),
        'categorical': get_categorical_columns(X, exclude),
        'datetime': get_datetime_columns(X)
    }