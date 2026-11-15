"""Tests for atlantic.preprocessing module: encoders, scalers, imputers, registry."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.exceptions import NotFittedError, RegistryError
from atlantic.preprocessing.encoders import (
    AutoIFrequencyEncoder,
    AutoLabelEncoder,
    AutoOneHotEncoder,
)
from atlantic.preprocessing.imputers import (
    AutoIterativeImputer,
    AutoKNNImputer,
    AutoSimpleImputer,
)
from atlantic.preprocessing.registry import (
    ComponentRegistry,
    EncoderRegistry,
    ImputerRegistry,
    ScalerRegistry,
)
from atlantic.preprocessing.scalers import (
    AutoMinMaxScaler,
    AutoRobustScaler,
    AutoStandardScaler,
)


# ============================================================================
# Label Encoder
# ============================================================================


class TestAutoLabelEncoder:
    def test_fit_transform(self, categorical_df):
        enc = AutoLabelEncoder()
        enc.fit(categorical_df)
        result = enc.transform(categorical_df)
        assert result.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all()
        assert result.shape == categorical_df.shape

    def test_is_fitted_flag(self, categorical_df):
        enc = AutoLabelEncoder()
        assert enc.is_fitted is False
        enc.fit(categorical_df)
        assert enc.is_fitted is True

    def test_transform_before_fit_raises(self, categorical_df):
        enc = AutoLabelEncoder()
        with pytest.raises(NotFittedError):
            enc.transform(categorical_df)

    def test_unknown_values_handled(self, categorical_df):
        enc = AutoLabelEncoder()
        enc.fit(categorical_df)
        new_df = categorical_df.copy()
        new_df.iloc[0, 0] = "NEVER_SEEN"
        result = enc.transform(new_df)
        assert result.shape == new_df.shape

    def test_inverse_transform(self, categorical_df):
        enc = AutoLabelEncoder()
        enc.fit(categorical_df)
        encoded = enc.transform(categorical_df)
        decoded = enc.inverse_transform(encoded)
        pd.testing.assert_frame_equal(decoded, categorical_df)

    def test_fit_transform_method(self, categorical_df):
        enc = AutoLabelEncoder()
        result = enc.fit_transform(categorical_df)
        assert result.shape == categorical_df.shape
        assert enc.is_fitted


# ============================================================================
# OneHot Encoder
# ============================================================================


class TestAutoOneHotEncoder:
    def test_fit_transform(self, categorical_df):
        enc = AutoOneHotEncoder()
        enc.fit(categorical_df)
        result = enc.transform(categorical_df)
        assert result.shape[0] == categorical_df.shape[0]
        # More columns after one-hot
        assert result.shape[1] > categorical_df.shape[1]

    def test_transform_before_fit_raises(self, categorical_df):
        enc = AutoOneHotEncoder()
        with pytest.raises(NotFittedError):
            enc.transform(categorical_df)

    def test_inverse_transform(self, categorical_df):
        enc = AutoOneHotEncoder()
        enc.fit(categorical_df)
        encoded = enc.transform(categorical_df)
        decoded = enc.inverse_transform(encoded)
        assert decoded.shape == categorical_df.shape
        for col in categorical_df.columns:
            assert set(decoded[col].dropna()) <= set(categorical_df[col].dropna())

    def test_unknown_category_handled(self, categorical_df):
        enc = AutoOneHotEncoder()
        enc.fit(categorical_df)
        new_df = categorical_df.copy()
        new_df.iloc[0, 0] = "UNSEEN_VALUE"
        result = enc.transform(new_df)
        assert result.shape[0] == new_df.shape[0]

    def test_unknown_category_produces_all_zero_row(self, categorical_df):
        """Regression test: with ``handle_unknown='ignore'`` an unseen
        category must produce an all-zeros encoding for that column,
        not an extra ``<col>_Unknown`` sentinel column.

        The previous implementation injected ``f"{col}_Unknown"`` columns
        which only existed if "Unknown" was a literal training category;
        the branch was dead code.
        """
        enc = AutoOneHotEncoder()
        enc.fit(categorical_df[["col1"]])
        unseen = pd.DataFrame({"col1": ["NEVER_SEEN_BEFORE"]})
        result = enc.transform(unseen)
        # All encoded col1_* columns should be 0 for the unseen row
        col1_cols = [c for c in result.columns if c.startswith("col1_")]
        assert len(col1_cols) > 0
        assert (result.iloc[0][col1_cols] == 0).all()
        # And no spurious col1_Unknown column should appear
        assert "col1_Unknown" not in result.columns or (
            result["col1_Unknown"].iloc[0] == 0
        )


# ============================================================================
# IFrequency Encoder
# ============================================================================


class TestAutoIFrequencyEncoder:
    def test_fit_transform(self, categorical_df):
        enc = AutoIFrequencyEncoder()
        enc.fit(categorical_df)
        result = enc.transform(categorical_df)
        assert result.shape == categorical_df.shape
        # All values should be numeric (float from log)
        for col in result.columns:
            assert np.issubdtype(result[col].dtype, np.floating)

    def test_transform_before_fit_raises(self, categorical_df):
        enc = AutoIFrequencyEncoder()
        with pytest.raises(NotFittedError):
            enc.transform(categorical_df)

    def test_rare_categories_higher_value(self):
        """Rare categories should get higher IDF-like scores."""
        df = pd.DataFrame({"col": ["A"] * 90 + ["B"] * 10})
        enc = AutoIFrequencyEncoder()
        enc.fit(df)
        result = enc.transform(df)
        val_a = result.loc[0, "col"]
        val_b = result.loc[90, "col"]
        assert val_b > val_a  # rare category has higher score

    def test_unseen_category_handled(self, categorical_df):
        enc = AutoIFrequencyEncoder()
        enc.fit(categorical_df)
        new_df = categorical_df.copy()
        new_df.iloc[0, 0] = "NEVER_SEEN"
        result = enc.transform(new_df)
        assert result.shape == new_df.shape
        assert not np.isnan(result.iloc[0, 0])

    def test_unseen_category_bounded_by_training_range(self):
        """Regression test: unseen categories must NOT extrapolate above
        the largest training-time encoded value.

        The previous implementation used ``min(frequency_map.values())``
        for unknowns, which after the inverse-frequency formula produced
        the *largest* encoded value — exceeding anything seen in
        training. The fix uses ``max(frequency)`` so unknowns map to the
        smallest in-distribution output.
        """
        # "A" is frequent (smallest value), "B" is rare (largest value)
        df = pd.DataFrame({"col": ["A"] * 90 + ["B"] * 10})
        enc = AutoIFrequencyEncoder()
        enc.fit(df)
        train_encoded = enc.transform(df)
        max_train_val = train_encoded["col"].max()

        unseen = pd.DataFrame({"col": ["UNKNOWN"]})
        unseen_encoded = enc.transform(unseen)
        unseen_val = unseen_encoded.iloc[0, 0]

        assert unseen_val <= max_train_val + 1e-9
        # Unknown maps to the most-frequent category ("A")
        expected = train_encoded.loc[0, "col"]  # "A" encoding
        assert unseen_val == pytest.approx(expected)

    def test_nan_preserved(self):
        df = pd.DataFrame({"col": ["A", "B", None, "A"]})
        enc = AutoIFrequencyEncoder()
        enc.fit(df)
        result = enc.transform(df)
        assert pd.isna(result.iloc[2, 0])

    def test_inverse_transform(self, categorical_df):
        enc = AutoIFrequencyEncoder()
        enc.fit(categorical_df)
        encoded = enc.transform(categorical_df)
        decoded = enc.inverse_transform(encoded)
        assert decoded.shape == categorical_df.shape


# ============================================================================
# Scalers
# ============================================================================


class TestAutoStandardScaler:
    def test_fit_transform(self, numeric_df):
        sc = AutoStandardScaler()
        sc.fit(numeric_df)
        result = sc.transform(numeric_df)
        assert result.shape == numeric_df.shape
        # Mean should be near 0
        assert abs(result.mean().mean()) < 0.1

    def test_inverse_transform(self, numeric_df):
        sc = AutoStandardScaler()
        sc.fit(numeric_df)
        scaled = sc.transform(numeric_df)
        original = sc.inverse_transform(scaled)
        pd.testing.assert_frame_equal(original, numeric_df, atol=1e-10)

    def test_transform_before_fit_raises(self, numeric_df):
        sc = AutoStandardScaler()
        with pytest.raises(NotFittedError):
            sc.transform(numeric_df)

    def test_index_preserved(self, numeric_df):
        df = numeric_df.copy()
        df.index = range(10, 10 + len(df))
        sc = AutoStandardScaler()
        sc.fit(df)
        result = sc.transform(df)
        assert list(result.index) == list(df.index)


class TestAutoMinMaxScaler:
    def test_fit_transform(self, numeric_df):
        sc = AutoMinMaxScaler()
        sc.fit(numeric_df)
        result = sc.transform(numeric_df)
        assert result.shape == numeric_df.shape
        assert result.min().min() >= -1e-10
        assert result.max().max() <= 1.0 + 1e-10

    def test_inverse_transform(self, numeric_df):
        sc = AutoMinMaxScaler()
        sc.fit(numeric_df)
        scaled = sc.transform(numeric_df)
        original = sc.inverse_transform(scaled)
        pd.testing.assert_frame_equal(original, numeric_df, atol=1e-10)

    def test_transform_before_fit_raises(self, numeric_df):
        sc = AutoMinMaxScaler()
        with pytest.raises(NotFittedError):
            sc.transform(numeric_df)


class TestAutoRobustScaler:
    def test_fit_transform(self, numeric_df):
        sc = AutoRobustScaler()
        sc.fit(numeric_df)
        result = sc.transform(numeric_df)
        assert result.shape == numeric_df.shape

    def test_inverse_transform(self, numeric_df):
        sc = AutoRobustScaler()
        sc.fit(numeric_df)
        scaled = sc.transform(numeric_df)
        original = sc.inverse_transform(scaled)
        pd.testing.assert_frame_equal(original, numeric_df, atol=1e-10)


# ============================================================================
# Imputers
# ============================================================================


class TestAutoSimpleImputer:
    def test_fit_transform_no_nulls(self, numeric_df):
        imp = AutoSimpleImputer(strategy="mean")
        imp.fit(numeric_df)
        result = imp.transform(numeric_df)
        assert result.isnull().sum().sum() == 0

    def test_imputes_nulls(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        imp = AutoSimpleImputer(strategy="mean")
        imp.fit(df)
        result = imp.transform(df)
        assert result.isnull().sum().sum() == 0
        assert result.loc[1, "a"] == pytest.approx(2.0)

    def test_excludes_target(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "target": [10.0, 20.0, 30.0]})
        imp = AutoSimpleImputer(strategy="mean", target="target")
        imp.fit(df)
        result = imp.transform(df)
        assert result.isnull().sum().sum() == 0

    def test_transform_before_fit_raises(self, numeric_df):
        imp = AutoSimpleImputer()
        with pytest.raises(NotFittedError):
            imp.transform(numeric_df)

    def test_median_strategy(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0]})
        imp = AutoSimpleImputer(strategy="median")
        imp.fit(df)
        result = imp.transform(df)
        assert result.loc[1, "a"] == pytest.approx(3.0)


class TestAutoKNNImputer:
    def test_imputes_nulls(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0, 5.0],
                "b": [2.0, 3.0, np.nan, 5.0, 6.0],
            }
        )
        imp = AutoKNNImputer(n_neighbors=2)
        imp.fit(df)
        result = imp.transform(df)
        assert result.isnull().sum().sum() == 0

    def test_transform_before_fit_raises(self, numeric_df):
        imp = AutoKNNImputer()
        with pytest.raises(NotFittedError):
            imp.transform(numeric_df)


class TestAutoIterativeImputer:
    def test_imputes_nulls(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0, 5.0],
                "b": [2.0, 3.0, np.nan, 5.0, 6.0],
            }
        )
        imp = AutoIterativeImputer(max_iter=5, random_state=42)
        imp.fit(df)
        result = imp.transform(df)
        assert result.isnull().sum().sum() == 0

    def test_transform_before_fit_raises(self, numeric_df):
        imp = AutoIterativeImputer()
        with pytest.raises(NotFittedError):
            imp.transform(numeric_df)


class TestBaseImputer:
    def test_impute_method(self):
        """Test the convenience impute(train, test) method."""
        train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        test = pd.DataFrame({"a": [np.nan, 2.0], "b": [7.0, np.nan]})
        imp = AutoSimpleImputer(strategy="mean")
        train_imp, test_imp = imp.impute(train, test)
        assert train_imp.isnull().sum().sum() == 0
        assert test_imp.isnull().sum().sum() == 0


# ============================================================================
# Registry
# ============================================================================


class TestComponentRegistry:
    def test_encoder_registry_list(self):
        available = EncoderRegistry.list_available()
        assert "label" in available
        assert "ifrequency" in available
        assert "onehot" in available

    def test_scaler_registry_list(self):
        available = ScalerRegistry.list_available()
        assert "standard" in available
        assert "minmax" in available
        assert "robust" in available

    def test_imputer_registry_list(self):
        available = ImputerRegistry.list_available()
        assert "simple" in available
        assert "knn" in available
        assert "iterative" in available

    def test_get_encoder(self):
        enc = EncoderRegistry.get("label")
        assert isinstance(enc, AutoLabelEncoder)

    def test_get_scaler(self):
        sc = ScalerRegistry.get("standard")
        assert isinstance(sc, AutoStandardScaler)

    def test_get_unknown_raises(self):
        with pytest.raises(RegistryError):
            EncoderRegistry.get("nonexistent")

    def test_is_registered(self):
        assert EncoderRegistry.is_registered("label")
        assert not EncoderRegistry.is_registered("bogus")

    def test_get_class(self):
        cls = EncoderRegistry.get_class("label")
        assert cls is AutoLabelEncoder

    def test_case_insensitive(self):
        enc = EncoderRegistry.get("LABEL")
        assert isinstance(enc, AutoLabelEncoder)
