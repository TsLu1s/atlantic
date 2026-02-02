from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from atlantic.core.exceptions import NotFittedError
from atlantic.preprocessing.base import BaseEncoder

if TYPE_CHECKING:
    from typing import Self


class AutoLabelEncoder(BaseEncoder):
    """
    Automatic Label Encoder for categorical columns.

    Applies sklearn LabelEncoder to multiple columns with support
    for unknown values during transform.
    """

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

        for col in self._columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            # Add unknown label for unseen values
            le.classes_ = np.append(le.classes_, "Unknown")
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
                X_encoded[col] = (
                    X[col]
                    .astype(str)
                    .apply(
                        lambda x: (
                            le.transform([x])[0]
                            if x in le.classes_
                            else le.transform(["Unknown"])[0]
                        )
                    )
                )

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
        self._decoded = None

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

        for col in X.columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(X[col].values.reshape(-1, 1))
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
        self._decoded = X.copy()

        for col in self._columns:
            if col in X_encoded.columns:
                encoder = self._one_hot_encoders[col]
                encoded_col = encoder.transform(X_encoded[col].values.reshape(-1, 1))

                # Handle unknown categories
                if len(encoder.get_feature_names_out([col])) > 1:
                    unknown_cols = [
                        c
                        for c in encoder.get_feature_names_out([col])
                        if c.startswith(f"{col}_Unknown")
                    ]
                    for unknown_col in unknown_cols:
                        X_encoded[unknown_col] = 0

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

        X_decoded = pd.DataFrame()

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
    giving rare categories higher values.
    """

    def __init__(self):
        """Initialize AutoIFrequencyEncoder."""
        super().__init__()
        self._freq_fit: dict = {}
        self._columns = None
        self._X_size: int = 0
        self._ratio: float = 1.0
        self._decoded = None

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

        for col in self._columns:
            frequency = X[col].value_counts().to_dict()
            self._freq_fit[col] = frequency

        self._is_fitted = True
        return self

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
        self._decoded = X.copy()

        for col in self._columns:
            if col in X_encoded.columns:
                frequency_map = self._freq_fit[col]

                def transform_value(x):
                    if pd.isna(x):
                        return x
                    if x in frequency_map:
                        return np.log1p(int(self._X_size * self._ratio) / frequency_map[x])
                    else:
                        # Handle unseen categories with minimum frequency
                        return np.log1p(
                            int(self._X_size * self._ratio) / min(frequency_map.values())
                        )

                X_encoded[col] = X[col].map(transform_value)

        return X_encoded

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform frequency encoded values back to original categories.

        Args:
            X: DataFrame with frequency encoded columns.

        Returns:
            DataFrame with original categorical values.
        """
        self._check_is_fitted()

        X_decoded = pd.DataFrame()

        for col in self._columns:
            if col in X.columns:
                frequency_map = self._freq_fit[col]
                inverse_frequency_map = {
                    np.log1p(int(self._X_size * self._ratio) / v): k
                    for k, v in frequency_map.items()
                }
                sorted_freq_keys = sorted(inverse_frequency_map.keys(), reverse=True)

                def inverse_transform_value(x):
                    if pd.isna(x):
                        return x
                    closest_log_val = min(sorted_freq_keys, key=lambda log_val: abs(log_val - x))
                    if abs(closest_log_val - x) > 1e-6:
                        return "unknown"
                    return inverse_frequency_map[closest_log_val]

                X_decoded[col] = X[col].map(inverse_transform_value)

        return X_decoded
