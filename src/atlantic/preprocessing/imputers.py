from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Self

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from atlantic.core.exceptions import NotFittedError
from atlantic.preprocessing.base import BaseImputer


class AutoSimpleImputer(BaseImputer):
    """
    Automatic Simple Imputer for null values.

    Uses sklearn SimpleImputer with configurable strategy.
    """

    def __init__(self, strategy: str = "mean", target: Optional[str] = None):
        """
        Initialize AutoSimpleImputer.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
            target: Target column name to exclude from imputation.
        """
        super().__init__(target)
        self.strategy = strategy
        self._imputer = None
        self._numeric_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit SimpleImputer to numerical columns.

        Returns:
            Self for method chaining.
        """
        X_fit = X.drop(columns=[self.target]) if self.target and self.target in X.columns else X

        self._numeric_columns = X_fit.select_dtypes(include=[np.number]).columns.tolist()

        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X_fit[self._numeric_columns])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        self._check_is_fitted()

        X_transformed = X.copy()

        # Only use numeric columns that exist in X
        available_cols = [c for c in self._numeric_columns if c in X.columns]

        if available_cols:
            imputed_numeric = self._imputer.transform(X[available_cols])
            X_transformed[available_cols] = imputed_numeric

        return X_transformed


class AutoKNNImputer(BaseImputer):
    """
    Automatic KNN Imputer for null values.

    Uses sklearn KNNImputer with configurable neighbors.
    """

    def __init__(
        self, n_neighbors: int = 5, weights: str = "uniform", target: Optional[str] = None
    ):
        """
        Initialize AutoKNNImputer.

        Args:
            n_neighbors: Number of neighbors to use.
            weights: Weight function ('uniform' or 'distance').
            target: Target column name to exclude from imputation.
        """
        super().__init__(target)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer = None
        self._numeric_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit KNNImputer to numerical columns.

        Returns:
            Self for method chaining.
        """
        X_fit = X.drop(columns=[self.target]) if self.target and self.target in X.columns else X

        self._numeric_columns = X_fit.select_dtypes(include=[np.number]).columns.tolist()

        self._imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        self._imputer.fit(X_fit[self._numeric_columns])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values using KNN."""
        self._check_is_fitted()

        X_transformed = X.copy()

        # Only use numeric columns that exist in X
        available_cols = [c for c in self._numeric_columns if c in X.columns]

        if available_cols:
            imputed_numeric = self._imputer.transform(X[available_cols])
            X_transformed[available_cols] = imputed_numeric

        return X_transformed


class AutoIterativeImputer(BaseImputer):
    """
    Automatic Iterative Imputer for null values.

    Uses sklearn IterativeImputer (multivariate imputation).
    """

    def __init__(
        self,
        max_iter: int = 10,
        random_state: Optional[int] = None,
        initial_strategy: str = "mean",
        imputation_order: str = "ascending",
        target: Optional[str] = None,
    ):
        """
        Initialize AutoIterativeImputer.

        Args:
            max_iter: Maximum number of imputation rounds.
            random_state: Random seed for reproducibility.
            initial_strategy: Initial imputation strategy.
            imputation_order: Order of feature imputation.
            target: Target column name to exclude from imputation.
        """
        super().__init__(target)
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self._imputer = None
        self._numeric_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit IterativeImputer to numerical columns.

        Returns:
            Self for method chaining.
        """
        X_fit = X.drop(columns=[self.target]) if self.target and self.target in X.columns else X

        self._numeric_columns = X_fit.select_dtypes(include=[np.number]).columns.tolist()

        self._imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy=self.initial_strategy,
            imputation_order=self.imputation_order,
        )
        self._imputer.fit(X_fit[self._numeric_columns])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values iteratively.

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()

        X_transformed = X.copy()

        # Only use numeric columns that exist in X
        available_cols = [c for c in self._numeric_columns if c in X.columns]

        if available_cols:
            imputed_numeric = self._imputer.transform(X[available_cols])
            X_transformed[available_cols] = imputed_numeric

        return X_transformed
