"""
Atlantic - Numerical Scalers.

Thin DataFrame-aware adapters around the scikit-learn scalers. They fit on
the columns they are given and return a DataFrame holding exactly those
columns, with the caller's index preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from atlantic.core.exceptions import ValidationError
from atlantic.preprocessing.base import BaseScaler

if TYPE_CHECKING:
    from typing import Self


class _SklearnScalerAdapter(BaseScaler):
    """Shared fit / transform / inverse_transform logic for sklearn scalers."""

    _scaler_factory: Callable[[], Any]

    def __init__(self):
        super().__init__()
        self._scaler = type(self)._scaler_factory()
        self._columns: Optional[List[str]] = None

    def _require_columns(self, X: pd.DataFrame) -> None:
        missing = [col for col in self._columns if col not in X.columns]
        if missing:
            raise ValidationError(
                f"{type(self).__name__} was fitted on columns that are missing "
                f"from the input: {missing}"
            )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit the scaler to numerical columns.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the fitted numerical columns.

        Returns:
            DataFrame with the scaled columns (index preserved).
        """
        self._check_is_fitted()
        self._require_columns(X)

        X_scaled = self._scaler.transform(X[self._columns])
        return pd.DataFrame(X_scaled, columns=self._columns, index=X.index)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled values back to original scale.

        Returns:
            DataFrame with original scale values (index preserved).
        """
        self._check_is_fitted()
        self._require_columns(X)

        X_original = self._scaler.inverse_transform(X[self._columns])
        return pd.DataFrame(X_original, columns=self._columns, index=X.index)


class AutoMinMaxScaler(_SklearnScalerAdapter):
    """
    Automatic Min-Max Scaler for numerical columns.

    Scales features to a fixed range [0, 1] using sklearn MinMaxScaler.
    """

    _scaler_factory = MinMaxScaler


class AutoStandardScaler(_SklearnScalerAdapter):
    """
    Automatic Standard Scaler for numerical columns.

    Scales features to have zero mean and unit variance using sklearn StandardScaler.
    """

    _scaler_factory = StandardScaler


class AutoRobustScaler(_SklearnScalerAdapter):
    """
    Automatic Robust Scaler for numerical columns.

    Scales features using statistics robust to outliers (median and IQR).
    """

    _scaler_factory = RobustScaler
