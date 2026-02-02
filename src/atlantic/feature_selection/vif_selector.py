from __future__ import annotations

from typing import Optional, Self  # , TYPE_CHECKING

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from atlantic.core.exceptions import ValidationError
from atlantic.preprocessing.base import BaseFeatureSelector


class VIFFeatureSelector(BaseFeatureSelector):
    """
    Variance Inflation Factor based feature selection.

    Removes features with high multicollinearity iteratively.
    """

    VIF_MIN_THRESHOLD: float = 3.0
    VIF_MAX_THRESHOLD: float = 30.0

    def __init__(self, target: str, vif_threshold: float = 10.0):
        """
        Initialize VIF feature selector.

        Args:
            target: Target column name.
            vif_threshold: Maximum allowed VIF value.
        """
        super().__init__(target)

        if not self.VIF_MIN_THRESHOLD <= vif_threshold <= self.VIF_MAX_THRESHOLD:
            raise ValidationError(
                f"VIF threshold must be between {self.VIF_MIN_THRESHOLD} "
                f"and {self.VIF_MAX_THRESHOLD}"
            )

        self._vif_threshold = vif_threshold
        self._vif_df: Optional[pd.DataFrame] = None

    def _calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VIF for all features.

        Returns:
            DataFrame with VIF values.
        """
        vif = pd.DataFrame()
        vif["variables"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif.sort_values("VIF", ascending=False)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit VIF selector by iteratively removing high-VIF features.

        Returns:
            Self for method chaining.
        """
        # Get only numerical columns excluding target
        cols = [
            col
            for col in X.columns
            if col != self.target and X[col].dtype in ["int64", "float64", "int32", "float32"]
        ]

        if not cols:
            self._selected_features = list(X.columns)
            self._is_fitted = True
            return self

        X_numeric = X[cols].copy()

        # Check for non-numeric or null values
        if X_numeric.isnull().values.any():
            raise ValidationError("Null values not supported in VIF calculation")

        # Initial VIF calculation
        self._vif_df = self._calculate_vif(X_numeric)

        # Iteratively remove high-VIF features
        while self._vif_df["VIF"].max() >= self._vif_threshold:
            # Remove feature with highest VIF
            max_vif_var = self._vif_df.loc[
                self._vif_df["VIF"] == self._vif_df["VIF"].max(), "variables"
            ].iloc[0]

            cols = [c for c in cols if c != max_vif_var]

            if len(cols) <= 1:
                break

            X_numeric = X[cols]
            self._vif_df = self._calculate_vif(X_numeric)

        self._selected_features = cols
        self._is_fitted = True

        return self

    @property
    def vif_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get VIF DataFrame from last calculation.
        """
        return self._vif_df

    @property
    def n_removed(self) -> int:
        """
        Number of features removed.
        """
        if not self._is_fitted:
            return 0
        return len(self._vif_df) - len(self._selected_features) if self._vif_df is not None else 0
