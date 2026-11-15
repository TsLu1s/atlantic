"""
Atlantic - Categorical Encoders.

Label, one-hot and inverse-frequency encoders with a shared fit/transform
contract. All encoders accept ``object``, ``category`` and string dtype
columns and tolerate categories unseen during fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from atlantic.preprocessing.base import BaseEncoder

if TYPE_CHECKING:
    from typing import Self

# Literal used to represent missing values as an explicit label class.
_MISSING_LABEL = "nan"


def _as_object(series: pd.Series) -> pd.Series:
    """View any categorical-like column (``category``, ``str``, ``string``,
    ``object``) as plain Python objects so value lookups behave identically
    regardless of the input dtype."""
    return series.astype(object)


def _as_labels(series: pd.Series) -> pd.Series:
    """Stringify a column for label encoding.

    Missing values become the literal label ``"nan"``. Doing this explicitly
    keeps the encoding identical across pandas versions: pandas < 3 turns
    ``NaN`` into ``"nan"`` on ``astype(str)`` while pandas >= 3 keeps it
    missing.
    """
    values = _as_object(series)
    values = values.where(series.notna(), _MISSING_LABEL)
    return values.astype(str)


class AutoLabelEncoder(BaseEncoder):
    """
    Automatic Label Encoder for categorical columns.

    Applies sklearn LabelEncoder to multiple columns with support
    for unknown values during transform.

    Every column gets one extra reserved class for categories never seen
    during fit. It is called ``"Unknown"`` unless that literal is itself a
    training category, in which case a non-colliding token is used so the real
    category keeps its own code. Missing values are encoded as their own
    ``"nan"`` class.
    """

    UNKNOWN_TOKEN: str = "Unknown"

    def __init__(self):
        """Initialize AutoLabelEncoder."""
        super().__init__()
        self._label_encoders: dict = {}
        self._columns = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit LabelEncoder to each categorical column.

        Args:
            X: DataFrame with categorical columns.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._label_encoders = {}

        for col in self._columns:
            le = LabelEncoder()
            le.fit(_as_labels(X[col]))

            # Reserve the last class for unseen values
            token = self.UNKNOWN_TOKEN
            existing = set(le.classes_.tolist())
            while token in existing:
                token = f"_{token}_"
            le.classes_ = np.append(le.classes_, token)
            self._label_encoders[col] = le

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns to encoded values.

        Args:
            X: DataFrame with categorical columns.

        Returns:
            DataFrame with encoded columns.
        """
        self._check_is_fitted()

        X_encoded = X.copy()
        for col in self._columns:
            if col in X_encoded.columns:
                le = self._label_encoders[col]
                # Vectorized encoding; unseen categories map to the unknown code
                mapping = {cls: code for code, cls in enumerate(le.classes_)}
                unknown_code = len(le.classes_) - 1
                X_encoded[col] = _as_labels(X[col]).map(mapping).fillna(unknown_code).astype(int)

        return X_encoded

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform encoded values back to original categories.

        Args:
            X: DataFrame with encoded columns.

        Returns:
            DataFrame with original categorical values.
        """
        self._check_is_fitted()

        X_decoded = X.copy()
        for col in self._columns:
            if col in X_decoded.columns:
                le = self._label_encoders[col]
                X_decoded[col] = le.inverse_transform(X[col].astype(int))

        return X_decoded


class AutoOneHotEncoder(BaseEncoder):
    """
    Automatic One-Hot Encoder for categorical columns.

    Applies sklearn OneHotEncoder to multiple columns with support
    for unknown values during transform.
    """

    def __init__(self):
        """Initialize AutoOneHotEncoder."""
        super().__init__()
        self._one_hot_encoders: dict = {}
        self._columns = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit OneHotEncoder to each categorical column.

        Args:
            X: DataFrame with categorical columns.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._one_hot_encoders = {}

        for col in self._columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            # np.asarray: arrow-backed .values does not support .reshape
            col_2d = np.asarray(_as_object(X[col])).reshape(-1, 1)
            encoder.fit(col_2d)
            self._one_hot_encoders[col] = encoder

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns to one-hot encoded values.

        Args:
            X: DataFrame with categorical columns.

        Returns:
            DataFrame with one-hot encoded columns.
        """
        self._check_is_fitted()

        X_encoded = X.copy()

        for col in self._columns:
            if col in X_encoded.columns:
                encoder = self._one_hot_encoders[col]
                col_2d = np.asarray(_as_object(X_encoded[col])).reshape(-1, 1)
                encoded_col = encoder.transform(col_2d)

                encoded_columns = encoder.get_feature_names_out([col])
                X_encoded[encoded_columns] = encoded_col
                X_encoded = X_encoded.drop(columns=[col])

        return X_encoded

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform one-hot encoded values back to original categories.

        Args:
            X: DataFrame with one-hot encoded columns.

        Returns:
            DataFrame with original categorical values.
        """
        self._check_is_fitted()

        X_decoded = pd.DataFrame(index=X.index)

        for col in self._columns:
            encoder = self._one_hot_encoders[col]
            encoded_columns = encoder.get_feature_names_out([col])
            encoded_data = X[encoded_columns].values
            decoded_col = encoder.inverse_transform(encoded_data)
            X_decoded[col] = decoded_col.flatten()

        return X_decoded


class AutoIFrequencyEncoder(BaseEncoder):
    """
    Automatic Inverse Frequency Encoder for categorical columns.

    Encodes categories based on inverse document frequency principle,
    giving rare categories higher values: ``log1p(n_rows / frequency)``.

    The output is always ``float64`` - including for ``category`` dtype
    inputs. Missing values stay missing (so a downstream imputer can handle
    them) and categories unseen during fit are encoded as the most frequent
    training category, keeping outputs inside the training range.
    """

    def __init__(self):
        """Initialize AutoIFrequencyEncoder."""
        super().__init__()
        self._freq_fit: dict = {}
        self._columns = None
        self._X_size: int = 0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit frequency encoder by computing category frequencies.

        Args:
            X: DataFrame with categorical columns.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._columns = X.columns.tolist()
        self._X_size = len(X)
        self._freq_fit = {}

        for col in self._columns:
            # Count on the object view to skip unobserved categories
            self._freq_fit[col] = _as_object(X[col]).value_counts().to_dict()

        self._is_fitted = True
        return self

    def _unknown_frequency(self, frequency_map: Dict) -> float:
        """Frequency assigned to categories unseen during fit.

        Unseen categories are treated as MAXIMALLY frequent (not minimally
        frequent). Inverse-frequency encoding maps high frequency to a *small*
        output value, so an unknown category maps to the smallest
        in-distribution value rather than an extrapolated maximum that exceeds
        anything seen in training. A column with no observed category at all
        (entirely missing during fit) falls back to the row count.
        """
        if frequency_map:
            return float(max(frequency_map.values()))
        return float(max(self._X_size, 1))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using inverse frequency encoding.

        Args:
            X: DataFrame with categorical columns.

        Returns:
            DataFrame with frequency encoded columns.
        """
        self._check_is_fitted()

        X_encoded = X.copy()

        for col in self._columns:
            if col in X_encoded.columns:
                frequency_map = self._freq_fit[col]
                values = _as_object(X[col])

                frequency = pd.to_numeric(values.map(frequency_map), errors="coerce").astype(
                    "float64"
                )
                unseen = frequency.isna() & values.notna()
                if unseen.any():
                    frequency = frequency.mask(unseen, self._unknown_frequency(frequency_map))

                X_encoded[col] = np.log1p(self._X_size / frequency)

        return X_encoded

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform frequency encoded values back to original categories.

        Note:
            Inverse-frequency encoding is not injective: categories that share
            the same training frequency share one encoded value, so only one
            of them can be recovered.

        Args:
            X: DataFrame with frequency encoded columns.

        Returns:
            DataFrame with original categorical values.
        """
        self._check_is_fitted()

        X_decoded = pd.DataFrame(index=X.index)

        for col in self._columns:
            if col in X.columns:
                frequency_map = self._freq_fit[col]
                inverse_frequency_map = {
                    np.log1p(self._X_size / v): k for k, v in frequency_map.items()
                }
                encoded_values = sorted(inverse_frequency_map.keys(), reverse=True)

                def decode(x, lookup=inverse_frequency_map, candidates=encoded_values):
                    if pd.isna(x) or not candidates:
                        return np.nan if pd.isna(x) else "unknown"
                    closest = min(candidates, key=lambda value: abs(value - x))
                    if abs(closest - x) > 1e-6:
                        return "unknown"
                    return lookup[closest]

                X_decoded[col] = X[col].map(decode)

        return X_decoded
