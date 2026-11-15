"""Tests for atlantic.feature_selection module: VIF selector and registry."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.exceptions import RegistryError, ValidationError
from atlantic.feature_selection.registry import FeatureSelectorRegistry
from atlantic.feature_selection.vif_selector import VIFFeatureSelector


class TestVIFFeatureSelector:
    @pytest.fixture
    def collinear_df(self):
        """DataFrame with known collinearity."""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = x1 * 0.95 + np.random.randn(n) * 0.05  # highly correlated with x1
        x3 = np.random.randn(n)  # independent
        return pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "target": np.random.randn(n),
            }
        )

    @pytest.fixture
    def independent_df(self):
        """DataFrame with no collinearity."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame(
            {
                "a": np.random.randn(n),
                "b": np.random.randn(n),
                "c": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )

    def test_removes_collinear_features(self, collinear_df):
        selector = VIFFeatureSelector(target="target", vif_threshold=5.0)
        selector.fit(collinear_df)
        selected = selector.selected_features
        # Should have removed one of the collinear pair
        assert len(selected) < 3
        assert "x3" in selected  # independent feature should survive

    def test_keeps_independent_features(self, independent_df):
        selector = VIFFeatureSelector(target="target", vif_threshold=10.0)
        selector.fit(independent_df)
        assert len(selector.selected_features) == 3

    def test_transform(self, collinear_df):
        selector = VIFFeatureSelector(target="target", vif_threshold=5.0)
        selector.fit(collinear_df)
        result = selector.transform(collinear_df)
        assert "target" in result.columns
        assert result.shape[1] <= collinear_df.shape[1]

    def test_is_fitted(self, independent_df):
        selector = VIFFeatureSelector(target="target")
        assert selector.is_fitted is False
        selector.fit(independent_df)
        assert selector.is_fitted is True

    def test_vif_dataframe_property(self, independent_df):
        selector = VIFFeatureSelector(target="target")
        selector.fit(independent_df)
        vif_df = selector.vif_dataframe
        assert vif_df is not None
        assert "VIF" in vif_df.columns
        assert "variables" in vif_df.columns

    def test_n_removed(self, collinear_df):
        selector = VIFFeatureSelector(target="target", vif_threshold=5.0)
        selector.fit(collinear_df)
        assert selector.n_removed >= 0

    def test_invalid_threshold_low(self):
        with pytest.raises(ValidationError):
            VIFFeatureSelector(target="t", vif_threshold=1.0)

    def test_invalid_threshold_high(self):
        with pytest.raises(ValidationError):
            VIFFeatureSelector(target="t", vif_threshold=50.0)

    def test_null_values_raise(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "target": [4.0, 5.0, 6.0],
            }
        )
        selector = VIFFeatureSelector(target="target")
        with pytest.raises(ValidationError, match="Null"):
            selector.fit(df)

    def test_no_numeric_columns(self):
        df = pd.DataFrame(
            {
                "cat": ["A", "B", "C"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        selector = VIFFeatureSelector(target="target")
        selector.fit(df)
        # Should include all columns since no numeric to filter
        assert selector.is_fitted

    def test_n_selected(self, independent_df):
        selector = VIFFeatureSelector(target="target")
        selector.fit(independent_df)
        assert selector.n_selected == len(selector.selected_features)

    def test_vif_uses_explicit_constant(self, independent_df):
        """Regression test: ``variance_inflation_factor`` requires an
        explicit intercept column; without ``add_constant`` the VIFs are
        run against an intercept-less regression and are mathematically
        incorrect (almost always inflated, often dramatically).

        For three independent (uncorrelated) standard-normal columns
        with the correct VIF formulation, the values should sit close to
        1.0. If ``add_constant`` is missing, this test will see VIFs
        well above 1 (typically several).
        """
        selector = VIFFeatureSelector(target="target", vif_threshold=10.0)
        selector.fit(independent_df)
        vif_df = selector.vif_dataframe
        assert vif_df is not None
        # All independent features should have VIF very close to 1.
        assert (vif_df["VIF"] < 2.0).all(), (
            f"Independent features should have VIF~1, got {vif_df['VIF'].tolist()}"
        )

    def test_vif_detects_strong_collinearity(self):
        """With a near-perfect linear dependency, VIF must be large
        (>>10) so that the offending feature is removed."""
        np.random.seed(7)
        n = 300
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-3  # near-perfect copy of x1
        x3 = np.random.randn(n)
        df = pd.DataFrame(
            {"x1": x1, "x2": x2, "x3": x3, "target": np.random.randn(n)}
        )
        selector = VIFFeatureSelector(target="target", vif_threshold=10.0)
        selector.fit(df)
        assert "x3" in selector.selected_features
        # Exactly one of the collinear pair must be dropped.
        assert sum(c in selector.selected_features for c in ("x1", "x2")) == 1


class TestFeatureSelectorRegistry:
    def test_list_available(self):
        available = FeatureSelectorRegistry.list_available()
        assert "h2o" in available
        assert "vif" in available

    def test_is_registered(self):
        assert FeatureSelectorRegistry.is_registered("vif")
        assert FeatureSelectorRegistry.is_registered("h2o")
        assert not FeatureSelectorRegistry.is_registered("bogus")

    def test_get_class_vif(self):
        cls = FeatureSelectorRegistry.get_class("vif")
        assert cls is VIFFeatureSelector

    def test_get_unknown_raises(self):
        with pytest.raises(RegistryError):
            FeatureSelectorRegistry.get("nonexistent", target="t")
