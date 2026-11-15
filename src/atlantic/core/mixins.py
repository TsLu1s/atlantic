"""
Atlantic - Reusable Mixins.

Thin, stateless helpers shared by the pipeline, the evaluators and the
feature selectors. Column-type detection and datetime engineering delegate to
:mod:`atlantic.utils.columns` / :mod:`atlantic.utils.datetime`, which are the
single implementations of that logic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from atlantic.core.exceptions import ValidationError
from atlantic.utils.columns import CATEGORICAL_TYPES as _CATEGORICAL_TYPES
from atlantic.utils.columns import DATETIME_TYPES as _DATETIME_TYPES
from atlantic.utils.columns import NUMERIC_TYPES as _NUMERIC_TYPES
from atlantic.utils.columns import (
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    is_numeric_target,
)
from atlantic.utils.datetime import DATE_COMPONENTS as _DATE_COMPONENTS
from atlantic.utils.datetime import engineer_datetime_features

SPLIT_RATIO_MIN: float = 0.5
SPLIT_RATIO_MAX: float = 0.98


class DataValidationMixin:
    """Mixin for DataFrame validation."""

    target: Optional[str]  # Expected to be defined in the implementing class

    def _validate_dataframe(self, X: pd.DataFrame, require_target: bool = True) -> None:
        """
        Validate input DataFrame.

        Args:
            X: Input DataFrame to validate.
            require_target: Whether to require target column presence.

        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        if X.empty:
            raise ValidationError("Input DataFrame is empty")

        target = getattr(self, "target", None)
        if require_target and target and target not in X.columns:
            raise ValidationError(
                f"Target column '{target}' not found in DataFrame. "
                f"Available columns: {list(X.columns)}"
            )

    def _validate_split_ratio(self, ratio: float) -> None:
        """
        Validate split ratio value.

        Raises:
            ValidationError: If ratio is out of valid range.
        """
        if not SPLIT_RATIO_MIN <= ratio <= SPLIT_RATIO_MAX:
            raise ValidationError(
                f"split_ratio must be between {SPLIT_RATIO_MIN} and {SPLIT_RATIO_MAX}, got {ratio}"
            )

    def _validate_columns_exist(self, X: pd.DataFrame, columns: List[str]) -> None:
        """
        Validate that specified columns exist in DataFrame.

        Raises:
            ValidationError: If any column is missing.
        """
        missing = set(columns) - set(X.columns)
        if missing:
            raise ValidationError(f"Columns not found in DataFrame: {missing}")


class ColumnTypeMixin:
    """Mixin for column type detection.

    Detection delegates to :mod:`atlantic.utils.columns`. The ``*_TYPES``
    class attributes are legacy dtype-name lists kept for backwards
    compatibility; they are no longer what drives detection.
    """

    NUMERIC_TYPES: List[str] = list(_NUMERIC_TYPES)
    CATEGORICAL_TYPES: List[str] = list(_CATEGORICAL_TYPES)
    DATETIME_TYPES: List[str] = list(_DATETIME_TYPES)

    target: Optional[str]  # Expected to be defined in the implementing class

    def _target_exclusion(self) -> List[str]:
        target = getattr(self, "target", None)
        return [target] if target else []

    def _get_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of numeric columns, excluding target.

        Args:
            X: Input DataFrame.

        Returns:
            List of numeric column names.
        """
        return get_numeric_columns(X, exclude=self._target_exclusion())

    def _get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of categorical columns, excluding target.

        Returns:
            List of categorical column names.
        """
        return get_categorical_columns(X, exclude=self._target_exclusion())

    def _get_datetime_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of datetime columns (any resolution, tz-aware or naive).

        Returns:
            List of datetime column names.
        """
        return get_datetime_columns(X)

    def _get_all_feature_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get all feature columns (excluding target).

        Returns:
            List of all feature column names.
        """
        target = getattr(self, "target", None)
        return [col for col in X.columns if col != target]


class DateEngineeringMixin:
    """Mixin for datetime feature engineering."""

    DATE_COMPONENTS: List[str] = list(_DATE_COMPONENTS)

    def _engineer_datetime_features(self, X: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.

        Returns:
            DataFrame with engineered datetime features.
        """
        return engineer_datetime_features(X, drop_original=drop)


class SchemaValidationMixin:
    """Mixin for input/output schema validation."""

    _input_schema: Optional[dict] = None

    def _store_input_schema(self, X: pd.DataFrame) -> None:
        """
        Store input schema during fit.

        Args:
            X: Input DataFrame to capture schema from.
        """
        self._input_schema = {
            "columns": list(X.columns),
            "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
            "n_features": len(X.columns),
        }

    def _validate_input_schema(self, X: pd.DataFrame, strict: bool = False) -> None:
        """
        Validate input schema against stored schema.

        Raises:
            ValidationError: If schema validation fails.
        """
        if self._input_schema is None:
            raise ValidationError("No input schema stored. Call fit first.")

        expected_cols = set(self._input_schema["columns"])
        actual_cols = set(X.columns)

        if strict:
            if expected_cols != actual_cols:
                missing = expected_cols - actual_cols
                extra = actual_cols - expected_cols
                raise ValidationError(f"Column mismatch. Missing: {missing}, Extra: {extra}")
        else:
            missing = expected_cols - actual_cols
            if missing:
                raise ValidationError(f"Missing required columns: {missing}")


class TargetTypeMixin:
    """Mixin for target variable type detection."""

    NUMERIC_TYPES: List[str] = list(_NUMERIC_TYPES)

    target: Optional[str]
    pred_type: Optional[str] = None
    n_classes: Optional[int] = None
    eval_metric: Optional[str] = None

    def _detect_target_type(self, X: pd.DataFrame) -> Tuple[str, str]:
        """
        Detect target variable type and set appropriate metric.

        Delegates to the single ``is_numeric_target`` helper (see
        ``utils.columns``) so nullable extension dtypes (``Int64``,
        ``Float64``), narrow widths (``int8``/``int16``/``float16``) and
        unsigned integers are all classified consistently across the package.

        Returns:
            Tuple of (prediction_type, evaluation_metric).
        """
        if self.target not in X.columns:
            raise ValidationError(f"Target column '{self.target}' not found")

        if is_numeric_target(X[self.target]):
            self.pred_type = "Reg"
            self.eval_metric = "Mean Absolute Error"
            self.n_classes = None
        else:
            self.pred_type = "Class"
            self.n_classes = int(X[self.target].nunique())
            self.eval_metric = "F1" if self.n_classes > 2 else "Precision"

        return self.pred_type, self.eval_metric
