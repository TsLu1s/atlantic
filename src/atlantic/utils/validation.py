from __future__ import annotations

from typing import Any, List, Optional, Union

import pandas as np
import pandas as pd

from atlantic.core.exceptions import ValidationError


def validate_dataframe(X: Any, name: str = "Input", require_non_empty: bool = True) -> None:
    """
    Validate that input is a valid DataFrame.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame")

    if require_non_empty and X.empty:
        raise ValidationError(f"{name} DataFrame is empty")


def validate_column_exists(X: pd.DataFrame, column: str, name: str = "Column") -> None:
    """
    Validate that column exists in DataFrame.

    Raises:
        ValidationError: If column not found.
    """
    if column not in X.columns:
        raise ValidationError(
            f"{name} '{column}' not found in DataFrame. " f"Available columns: {list(X.columns)}"
        )


def validate_columns_exist(X: pd.DataFrame, columns: List[str], name: str = "Columns") -> None:
    """
    Validate that all columns exist in DataFrame.

    Raises:
        ValidationError: If any column not found.
    """
    missing = set(columns) - set(X.columns)
    if missing:
        raise ValidationError(f"{name} not found in DataFrame: {missing}")


def validate_numeric_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "Value",
) -> None:
    """
    Validate that numeric value is within range.

    Raises:
        ValidationError: If value out of range.
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")


def validate_in_choices(value: Any, choices: List[Any], name: str = "Value") -> None:
    """
    Validate that value is in allowed choices.

    Raises:
        ValidationError: If value not in choices.
    """
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got '{value}'")


def validate_no_nulls(
    X: pd.DataFrame, columns: Optional[List[str]] = None, name: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame has no null values.

    Raises:
        ValidationError: If nulls found.
    """
    if columns is None:
        columns = X.columns.tolist()

    null_counts = X[columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        raise ValidationError(
            f"{name} contains null values in columns: " f"{dict(cols_with_nulls)}"
        )


def validate_numeric_columns(X: pd.DataFrame, columns: List[str], name: str = "Columns") -> None:
    """
    Validate that columns are numeric.

    Raises:
        ValidationError: If non-numeric columns found.
    """
    numeric_types = ["int", "int32", "int64", "float", "float32", "float64"]

    non_numeric = []
    for col in columns:
        if col in X.columns and X[col].dtype.name not in numeric_types:
            non_numeric.append(col)

    if non_numeric:
        raise ValidationError(f"{name} must be numeric. Non-numeric columns: {non_numeric}")
