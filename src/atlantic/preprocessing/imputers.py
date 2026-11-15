"""
Atlantic - Null Value Imputers.

DataFrame-aware adapters around the scikit-learn imputers. They impute the
numeric columns of a frame (never the target) and leave every other column
untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from atlantic.core.exceptions import ValidationError
from atlantic.preprocessing.base import BaseImputer
from atlantic.utils.columns import get_numeric_columns

if TYPE_CHECKING:
    from typing import Self


class _SklearnImputerAdapter(BaseImputer):
    """Shared fit / transform logic for the sklearn-backed imputers.

    * Only numeric columns are imputed; the target column is excluded.
    * A frame with no numeric columns is a valid no-op.
    * Columns that are entirely missing during fit cannot be modelled; they
      are kept and filled with ``EMPTY_FILL_VALUE`` instead of being silently
      dropped by scikit-learn (which would change the output shape).
    * Transforming a frame that lacks a fitted column raises a clear
      :class:`ValidationError`.
    """

    EMPTY_FILL_VALUE: float = 0.0

    def __init__(self, target: Optional[str] = None):
        super().__init__(target)
        self._imputer = None
        self._numeric_columns: List[str] = []
        self._fit_columns: List[str] = []
        self._empty_columns: List[str] = []

    def _create_imputer(self) -> Any:
        """Build the underlying (unfitted) sklearn imputer."""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit the imputer to the numerical columns of ``X``.

        Returns:
            Self for method chaining.
        """
        X_fit = X.drop(columns=[self.target]) if self.target and self.target in X.columns else X

        self._numeric_columns = get_numeric_columns(X_fit)
        self._empty_columns = [col for col in self._numeric_columns if X_fit[col].isna().all()]
        self._fit_columns = [
            col for col in self._numeric_columns if col not in set(self._empty_columns)
        ]
        self._imputer = None

        if self._fit_columns:
            self._imputer = self._create_imputer()
            self._imputer.fit(X_fit[self._fit_columns])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values in the fitted numeric columns.

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()

        X_transformed = X.copy()

        # ``_fit_columns`` / ``_empty_columns`` are absent on imputers pickled
        # by older versions, which fitted every numeric column.
        fit_columns = getattr(self, "_fit_columns", self._numeric_columns)
        empty_columns = getattr(self, "_empty_columns", [])

        missing = [col for col in self._numeric_columns if col not in X.columns]
        if missing:
            raise ValidationError(
                f"{type(self).__name__} was fitted on columns that are missing "
                f"from the input: {missing}"
            )

        if fit_columns and self._imputer is not None:
            X_transformed[fit_columns] = self._imputer.transform(X[fit_columns])

        for col in empty_columns:
            X_transformed[col] = X[col].astype("float64").fillna(self.EMPTY_FILL_VALUE)

        return X_transformed


class AutoSimpleImputer(_SklearnImputerAdapter):
    """
    Automatic Simple Imputer for null values.

    Uses sklearn SimpleImputer with configurable strategy.
    """

    def __init__(
        self,
        strategy: str = "mean",
        target: Optional[str] = None,
        fill_value: Optional[Any] = None,
    ):
        """
        Initialize AutoSimpleImputer.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
            target: Target column name to exclude from imputation.
            fill_value: Value used when ``strategy="constant"``.
        """
        super().__init__(target)
        self.strategy = strategy
        self.fill_value = fill_value

    def _create_imputer(self) -> SimpleImputer:
        return SimpleImputer(strategy=self.strategy, fill_value=getattr(self, "fill_value", None))


class AutoKNNImputer(_SklearnImputerAdapter):
    """
    Automatic KNN Imputer for null values.

    Uses sklearn KNNImputer with configurable neighbors.

    Note:
        KNN imputation computes pairwise distances and scales quadratically
        with the number of rows; expect it to dominate runtime on large data.
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

    def _create_imputer(self) -> KNNImputer:
        return KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)


class AutoIterativeImputer(_SklearnImputerAdapter):
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

    def _create_imputer(self) -> IterativeImputer:
        return IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy=self.initial_strategy,
            imputation_order=self.imputation_order,
        )
