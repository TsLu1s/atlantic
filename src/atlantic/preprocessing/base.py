from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from atlantic.core.base import BasePreprocessor
from atlantic.core.exceptions import NotFittedError

if TYPE_CHECKING:
    from typing import Self


class BaseEncoder(BasePreprocessor):
    """Abstract base class for categorical encoders."""

    def __init__(self):
        super().__init__()
        self._encoders: dict = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit encoder to categorical columns.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using fitted encoder.

        Returns:
            DataFrame with encoded columns.
        """
        ...

    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform encoded columns back to original values.

        Returns:
            DataFrame with original categorical values.
        """
        ...


class BaseScaler(BasePreprocessor):
    """Abstract base class for numerical scalers."""

    def __init__(self):
        super().__init__()
        self._scaler = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit scaler to numerical columns.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical columns using fitted scaler.

        Returns:
            DataFrame with scaled columns.
        """
        ...

    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled columns back to original scale.

        Returns:
            DataFrame with original scale values.
        """
        ...


class BaseImputer(BasePreprocessor):
    """Abstract base class for null value imputers."""

    def __init__(self, target: Optional[str] = None):
        super().__init__()
        self._imputer = None
        self.target = target
        self._numeric_columns: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit imputer to data.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values.

        Returns:
            DataFrame with imputed values.
        """
        ...

    def impute(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        """
        Fit on train and transform both train and test.

        Returns:
            Tuple of (imputed_train, imputed_test).
        """
        self.fit(train)
        return self.transform(train.copy()), self.transform(test.copy())


class BaseFeatureSelector(BasePreprocessor):
    """Abstract base class for feature selection methods."""

    def __init__(self, target: str):
        super().__init__()
        self.target = target
        self._selected_features: List[str] = []
        self._feature_importance: Optional[pd.DataFrame] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit feature selector to data.

        Returns:
            Self for method chaining.
        """
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select fitted features from DataFrame.

        Returns:
            DataFrame with selected features only.
        """
        self._check_is_fitted()

        # Include target if present
        cols = self._selected_features.copy()
        if self.target in X.columns and self.target not in cols:
            cols.append(self.target)

        return X[cols]

    @property
    def selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self._selected_features

    @property
    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance DataFrame."""
        return self._feature_importance

    @property
    def n_selected(self) -> int:
        """Number of selected features."""
        return len(self._selected_features)
