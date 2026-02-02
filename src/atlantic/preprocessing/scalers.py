from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from atlantic.core.exceptions import NotFittedError
from atlantic.preprocessing.base import BaseScaler


class AutoMinMaxScaler(BaseScaler):
    """
    Automatic Min-Max Scaler for numerical columns.

    Scales features to a fixed range [0, 1] using sklearn MinMaxScaler.
    """

    def __init__(self):
        """Initialize AutoMinMaxScaler."""
        super().__init__()
        self._scaler = MinMaxScaler()
        self._columns = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit MinMaxScaler to numerical columns.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical columns to [0, 1] range.

        Returns:
            DataFrame with scaled columns.
        """
        self._check_is_fitted()

        X_scaled = self._scaler.transform(X[self._columns])
        return pd.DataFrame(X_scaled, columns=self._columns, index=X.index)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled values back to original scale.

        Returns:
            DataFrame with original scale values.
        """
        self._check_is_fitted()

        X_original = self._scaler.inverse_transform(X[self._columns])
        return pd.DataFrame(X_original, columns=self._columns, index=X.index)


class AutoStandardScaler(BaseScaler):
    """
    Automatic Standard Scaler for numerical columns.

    Scales features to have zero mean and unit variance using sklearn StandardScaler.
    """

    def __init__(self):
        """Initialize AutoStandardScaler."""
        super().__init__()
        self._scaler = StandardScaler()
        self._columns = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit StandardScaler to numerical columns.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical columns to zero mean and unit variance.

        Returns:
            DataFrame with scaled columns.
        """
        self._check_is_fitted()

        X_scaled = self._scaler.transform(X[self._columns])
        return pd.DataFrame(X_scaled, columns=self._columns, index=X.index)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled values back to original scale.

        Returns:
            DataFrame with original scale values.
        """
        self._check_is_fitted()

        X_original = self._scaler.inverse_transform(X[self._columns])
        return pd.DataFrame(X_original, columns=self._columns, index=X.index)


class AutoRobustScaler(BaseScaler):
    """
    Automatic Robust Scaler for numerical columns.

    Scales features using statistics robust to outliers (median and IQR).
    """

    def __init__(self):
        """Initialize AutoRobustScaler."""
        super().__init__()
        self._scaler = RobustScaler()
        self._columns = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit RobustScaler to numerical columns.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical columns using robust scaling.

        Returns:
            DataFrame with scaled columns.
        """
        self._check_is_fitted()

        X_scaled = self._scaler.transform(X[self._columns])
        return pd.DataFrame(X_scaled, columns=self._columns, index=X.index)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled values back to original scale.

        Returns:
            DataFrame with original scale values.
        """
        self._check_is_fitted()

        X_original = self._scaler.inverse_transform(X[self._columns])
        return pd.DataFrame(X_original, columns=self._columns, index=X.index)
