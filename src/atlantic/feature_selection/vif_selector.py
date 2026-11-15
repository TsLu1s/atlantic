"""
Atlantic - VIF Feature Selection.

Iterative removal of multicollinear numeric features using the Variance
Inflation Factor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from atlantic.core.exceptions import ValidationError
from atlantic.preprocessing.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing import Self

# Squared eigenvector loadings below this are numerical noise.
_LOADING_TOLERANCE: float = 1e-18


def compute_vif(values: np.ndarray) -> np.ndarray:
    """
    Variance Inflation Factor of every column of a 2-D float array.

    ``VIF_i = 1 / (1 - R_i^2)``, where ``R_i^2`` comes from regressing column
    ``i`` on all the other columns *with an intercept*. That quantity equals
    the i-th diagonal element of the inverse correlation matrix, so all VIFs
    are obtained from one eigen-decomposition instead of one least-squares
    fit per column. The result matches
    ``statsmodels.stats.outliers_influence.variance_inflation_factor`` (with
    an explicit constant) to floating-point precision while being orders of
    magnitude faster on wide data.

    Degenerate inputs are handled explicitly:

    * a constant column has ``VIF = 1`` (it is explained by the intercept and
      does not inflate anything);
    * columns taking part in an exact linear dependency get ``VIF = inf``.

    Args:
        values: Array of shape (n_rows, n_columns) without missing values.

    Returns:
        Array of shape (n_columns,) with the VIF of each column.
    """
    values = np.asarray(values, dtype="float64")
    n_columns = values.shape[1]
    vif = np.ones(n_columns, dtype="float64")

    if values.shape[0] < 2 or n_columns == 0:
        return vif

    # Relative tolerance so a column that is constant up to float noise is
    # still treated as constant.
    scale = np.maximum(np.abs(values).max(axis=0), 1.0)
    varying = np.flatnonzero(values.std(axis=0) > 1e-12 * scale)
    if len(varying) < 2:
        return vif

    correlation = np.corrcoef(values[:, varying], rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)

    # diag(R^-1)_i = sum_j V_ij^2 / lambda_j. Eigenvalues at (numerical) zero
    # mark exact linear dependencies: every column loading on them is infinite.
    tolerance = len(varying) * np.finfo("float64").eps * max(float(eigenvalues.max()), 1.0)
    loadings = eigenvectors**2
    loadings[loadings < _LOADING_TOLERANCE] = 0.0
    singular = eigenvalues <= tolerance

    regular = loadings[:, ~singular] @ (1.0 / eigenvalues[~singular])
    regular[loadings[:, singular].sum(axis=1) > 0] = np.inf

    vif[varying] = regular
    return vif


class VIFFeatureSelector(BaseFeatureSelector):
    """
    Variance Inflation Factor based feature selection.

    Removes features with high multicollinearity iteratively: while the worst
    VIF is at or above ``vif_threshold`` the corresponding feature is dropped
    and the VIFs are recomputed.

    Only numeric (and boolean) features take part in the VIF computation.
    Non-numeric features are passed through untouched: they stay in
    ``selected_features`` and in the ``transform`` output. The target is never
    a selected feature.
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
        self._n_input_features: int = 0
        self._removed_features: List[str] = []

    def _calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VIF for all features of a numeric, null-free frame.

        The VIFs correspond to auxiliary regressions that include an
        intercept (see :func:`compute_vif`).

        Returns:
            DataFrame with VIF values per feature, sorted descending.
        """
        vif = pd.DataFrame(
            {"variables": list(X.columns), "VIF": compute_vif(X.to_numpy(dtype="float64"))}
        )
        return vif.sort_values("VIF", ascending=False, kind="stable")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit VIF selector by iteratively removing high-VIF features.

        Returns:
            Self for method chaining.
        """
        features = [col for col in X.columns if col != self.target]

        # Numeric columns of every width, nullable numerics and booleans all
        # take part; anything else is passed through.
        cols = [col for col in features if is_numeric_dtype(X[col])]

        self._n_input_features = len(cols)
        self._removed_features = []
        self._vif_df = None

        if cols:
            X_numeric = X[cols].astype("float64")

            if X_numeric.isnull().values.any():
                raise ValidationError("Null values not supported in VIF calculation")
            if not np.isfinite(X_numeric.to_numpy()).all():
                raise ValidationError("Infinite values not supported in VIF calculation")

            self._vif_df = self._calculate_vif(X_numeric)

            # Iteratively remove high-VIF features. A VIF of NaN/inf marks a
            # degenerate column (exact collinearity); such columns are always
            # removed FIRST - otherwise ``max()`` silently skips NaN and the
            # loop can terminate while collinearity remains.
            while True:
                vif = self._vif_df["VIF"]
                sanitized = vif.replace([np.inf, -np.inf], np.nan)
                worst = sanitized.max(skipna=True)
                degenerate_mask = vif.isna() | np.isinf(vif)

                if degenerate_mask.any():
                    remove_var = self._vif_df.loc[degenerate_mask, "variables"].iloc[0]
                elif pd.notna(worst) and worst >= self._vif_threshold:
                    remove_var = self._vif_df.loc[sanitized == worst, "variables"].iloc[0]
                else:
                    break

                cols = [c for c in cols if c != remove_var]
                self._removed_features.append(remove_var)

                if not cols:
                    self._vif_df = pd.DataFrame({"variables": [], "VIF": []})
                    break

                self._vif_df = self._calculate_vif(X_numeric[cols])

        removed = set(self._removed_features)
        self._selected_features = [col for col in features if col not in removed]
        self._is_fitted = True

        return self

    @property
    def vif_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get VIF DataFrame from last calculation (None when there were no
        numeric features).
        """
        return self._vif_df

    @property
    def removed_features(self) -> List[str]:
        """Features removed for multicollinearity, in removal order."""
        return list(getattr(self, "_removed_features", []))

    @property
    def n_removed(self) -> int:
        """
        Number of numeric features removed by VIF filtering.
        """
        if not self._is_fitted:
            return 0
        return len(getattr(self, "_removed_features", []))
