"""Tests for atlantic.pipeline module: Atlantic, AtlanticBuilder, AtlanticPipeline."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.exceptions import NotFittedError, ValidationError
from atlantic.core.schemas import BuilderConfig
from atlantic.pipeline.atlantic import Atlantic
from atlantic.pipeline.builder import AtlanticBuilder, AtlanticPipeline


# ============================================================================
# AtlanticBuilder
# ============================================================================


class TestAtlanticBuilder:
    def test_default_build(self):
        pipeline = AtlanticBuilder().build()
        assert isinstance(pipeline, AtlanticPipeline)
        assert pipeline.is_fitted is False

    def test_with_date_engineering(self):
        builder = AtlanticBuilder().with_date_engineering(enabled=False, drop=False)
        cfg = builder.get_config()
        assert cfg.date_engineering is False
        assert cfg.date_drop is False

    def test_with_null_removal(self):
        builder = AtlanticBuilder().with_null_removal(threshold=0.8)
        cfg = builder.get_config()
        assert cfg.null_removal_threshold == 0.8

    def test_with_feature_selection(self):
        builder = AtlanticBuilder().with_feature_selection(
            method="h2o", relevance=0.95, h2o_models=10
        )
        cfg = builder.get_config()
        assert cfg.feature_selection.method == "h2o"
        assert cfg.feature_selection.h2o_relevance == 0.95
        assert cfg.feature_selection.h2o_max_models == 10

    def test_with_feature_selection_none(self):
        builder = AtlanticBuilder().with_feature_selection(method="none")
        cfg = builder.get_config()
        assert cfg.feature_selection.method == "none"

    def test_with_encoding(self):
        builder = AtlanticBuilder().with_encoding(
            scaler="robust", encoder="label", auto_select=True
        )
        cfg = builder.get_config()
        assert cfg.encoding.scaler == "robust"
        assert cfg.encoding.encoder == "label"
        assert cfg.encoding.auto_select is True

    def test_with_imputation(self):
        builder = AtlanticBuilder().with_imputation(method="knn", auto_select=True)
        cfg = builder.get_config()
        assert cfg.imputation.method == "knn"
        assert cfg.imputation.auto_select is True

    def test_with_vif_filtering(self):
        builder = AtlanticBuilder().with_vif_filtering(threshold=8.0)
        cfg = builder.get_config()
        assert cfg.feature_selection.vif_threshold == 8.0

    def test_with_optimization(self):
        builder = AtlanticBuilder().with_optimization(
            optimization_level="thorough", random_state=123
        )
        cfg = builder.get_config()
        assert cfg.optimizer.optimization_level == "thorough"
        assert cfg.optimizer.random_state == 123

    def test_method_chaining(self):
        pipeline = (
            AtlanticBuilder()
            .with_date_engineering(enabled=True)
            .with_null_removal(threshold=0.9)
            .with_encoding(scaler="minmax")
            .with_imputation(method="iterative")
            .with_optimization(optimization_level="fast")
            .build()
        )
        assert isinstance(pipeline, AtlanticPipeline)

    def test_get_config_returns_builder_config(self):
        cfg = AtlanticBuilder().get_config()
        assert isinstance(cfg, BuilderConfig)


# ============================================================================
# AtlanticPipeline
# ============================================================================


class TestAtlanticPipeline:
    def test_not_fitted_transform_raises(self):
        pipeline = AtlanticBuilder().build()
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not fitted"):
            pipeline.transform(df)

    def test_config_property(self):
        pipeline = AtlanticBuilder().with_optimization(optimization_level="fast").build()
        assert pipeline.config.optimizer.optimization_level == "fast"


# ============================================================================
# Atlantic Main Class
# ============================================================================


class TestAtlantic:
    def test_init_with_data(self, sample_regression_df):
        atl = Atlantic(X=sample_regression_df, target="target")
        assert atl.X is not None
        assert atl.target == "target"
        assert atl.pred_type == "Reg"
        assert atl._is_fitted is False

    def test_init_without_data(self):
        atl = Atlantic()
        assert atl.X is None
        assert atl.target is None
        assert atl._is_fitted is False

    def test_init_classification(self, sample_classification_df):
        atl = Atlantic(X=sample_classification_df, target="target")
        assert atl.pred_type == "Class"
        assert atl.n_classes == 2

    def test_data_processing_before_fit_raises(self, sample_regression_df):
        atl = Atlantic(X=sample_regression_df, target="target")
        with pytest.raises(NotFittedError):
            atl.data_processing(sample_regression_df)

    def test_builder_classmethod(self):
        builder = Atlantic.builder()
        assert isinstance(builder, AtlanticBuilder)

    def test_fit_processing_invalid_split_ratio(self, sample_regression_df):
        atl = Atlantic(X=sample_regression_df, target="target")
        with pytest.raises(ValidationError):
            atl.fit_processing(split_ratio=0.1)

    def test_fit_processing_regression(self):
        """Full integration test for regression pipeline (no H2O)."""
        from atlantic.data.generator import DatasetGenerator

        data, target = DatasetGenerator.generate_regression(
            n_samples=300, n_features=5, n_categorical=2, null_percentage=0.05, random_state=42
        )
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data, train_size=0.8, random_state=42)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        atl = Atlantic(X=train, target=target)
        atl.fit_processing(
            split_ratio=0.75,
            relevance=1.0,  # skip H2O
            vif_ratio=10.0,
            optimization_level="fast",
        )

        assert atl._is_fitted is True
        assert atl.enc_method is not None
        assert len(atl.cols) > 0

        # Transform train
        train_processed = atl.data_processing(train)
        assert isinstance(train_processed, pd.DataFrame)
        assert len(train_processed) == len(train)

        # Transform test
        test_processed = atl.data_processing(test)
        assert isinstance(test_processed, pd.DataFrame)
        assert len(test_processed) == len(test)

    def test_fit_processing_classification(self):
        """Full integration test for classification pipeline (no H2O)."""
        from atlantic.data.generator import DatasetGenerator

        data, target = DatasetGenerator.generate_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_classes=2,
            n_categorical=2,
            null_percentage=0.05,
            random_state=42,
        )
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data, train_size=0.8, random_state=42)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        atl = Atlantic(X=train, target=target)
        atl.fit_processing(
            split_ratio=0.75,
            relevance=1.0,
            vif_ratio=10.0,
            optimization_level="fast",
        )

        assert atl._is_fitted is True
        train_processed = atl.data_processing(train)
        assert isinstance(train_processed, pd.DataFrame)
        test_processed = atl.data_processing(test)
        assert isinstance(test_processed, pd.DataFrame)

    def test_save_and_load(self, tmp_path):
        """Test pipeline serialization roundtrip."""
        from atlantic.data.generator import DatasetGenerator

        data, target = DatasetGenerator.generate_regression(
            n_samples=200, n_features=5, n_categorical=2, null_percentage=0.05, random_state=42
        )
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data, train_size=0.8, random_state=42)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        atl = Atlantic(X=train, target=target)
        atl.fit_processing(
            split_ratio=0.75,
            relevance=1.0,
            optimization_level="fast",
        )

        path = str(tmp_path / "pipeline.pkl")
        atl.save(path)
        loaded = Atlantic.load(path)

        assert loaded._is_fitted is True
        assert loaded.enc_method == atl.enc_method

        result_orig = atl.data_processing(test)
        result_loaded = loaded.data_processing(test)
        pd.testing.assert_frame_equal(result_orig, result_loaded)

    def test_fitted_components_property(self, sample_regression_df):
        atl = Atlantic(X=sample_regression_df, target="target")
        fc = atl.fitted_components
        assert fc is not None
        assert fc.selected_columns == []

    def test_fit_no_nulls(self):
        """Test pipeline on data with zero nulls."""
        from atlantic.data.generator import DatasetGenerator

        data, target = DatasetGenerator.generate_regression(
            n_samples=200, n_features=5, n_categorical=2, null_percentage=0.0, random_state=42
        )
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data, train_size=0.8, random_state=42)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        atl = Atlantic(X=train, target=target)
        atl.fit_processing(
            split_ratio=0.75,
            relevance=1.0,
            optimization_level="fast",
        )
        assert atl._is_fitted
        assert atl.imp_method == "Undefined"
        result = atl.data_processing(test)
        assert result.isnull().sum().sum() == 0

    def test_random_state_is_plumbed_to_split(self):
        """Regression test: ``_split_data`` previously hard-coded
        ``random_state=42`` and ignored the user's seed. Different seeds
        must now produce different train/test splits."""
        from atlantic.core.base import BasePipeline

        # Build two Atlantic instances with different seeds and inspect
        # the train split that ``_split_data`` produces.
        np.random.seed(0)
        n = 200
        df = pd.DataFrame(
            {"a": np.random.randn(n), "target": np.random.randn(n)}
        )

        atl_a = Atlantic(X=df.copy(), target="target", random_state=1)
        atl_b = Atlantic(X=df.copy(), target="target", random_state=99)

        train_a, _ = BasePipeline._split_data(atl_a, df.copy(), 0.75)
        train_b, _ = BasePipeline._split_data(atl_b, df.copy(), 0.75)

        # Same shape, different content (different seed → different rows).
        assert train_a.shape == train_b.shape
        assert not train_a["a"].equals(train_b["a"])

    def test_random_state_reproducible(self):
        """Same seed must reproduce the same split exactly."""
        from atlantic.core.base import BasePipeline

        np.random.seed(0)
        df = pd.DataFrame(
            {"a": np.random.randn(150), "target": np.random.randn(150)}
        )
        atl_1 = Atlantic(X=df.copy(), target="target", random_state=7)
        atl_2 = Atlantic(X=df.copy(), target="target", random_state=7)
        t1, _ = BasePipeline._split_data(atl_1, df.copy(), 0.75)
        t2, _ = BasePipeline._split_data(atl_2, df.copy(), 0.75)
        pd.testing.assert_frame_equal(t1, t2)

    def test_final_processors_fit_only_on_train_rows(self):
        """Regression test for the data-leakage fix.

        Strategy: spy on ``_fit_final_processors`` and assert it is called
        with a DataFrame whose row count equals the size of the training
        split — never the full pre-split frame. The previous
        implementation passed the full frame, leaking test rows into the
        inference-time scaler/encoder/imputer state.
        """
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "cat_x": np.random.choice(["A", "B", "C"], n),
                "target": np.random.randn(n),
            }
        )
        split_ratio = 0.75
        # ``_split_data`` drops null targets and there are none, so the
        # split sizes are deterministic with our seed:
        expected_train_size = int(round(n * split_ratio))

        atl = Atlantic(X=df.copy(), target="target", random_state=42)

        captured: list[pd.DataFrame] = []
        original = atl._fit_final_processors

        def spy(X):
            captured.append(X)
            return original(X)

        atl._fit_final_processors = spy
        atl.fit_processing(
            split_ratio=split_ratio, relevance=1.0, optimization_level="fast"
        )

        assert len(captured) == 1, "_fit_final_processors should be called once"
        passed_df = captured[0]
        # Must be the train split (not the full frame).
        assert passed_df.shape[0] != n, (
            "Leakage: _fit_final_processors received the full DataFrame "
            f"(n={n}) instead of the train split"
        )
        # Allow for ±1 row variance from rounding/stratification.
        assert abs(passed_df.shape[0] - expected_train_size) <= 1, (
            f"Expected ~{expected_train_size} train rows, got {passed_df.shape[0]}"
        )
        # And it must be PRE-encoded (raw categorical values still present),
        # not the post-encoding numeric form — otherwise the final encoder
        # cannot be fit. ``cat_x`` should be a non-numeric dtype regardless
        # of the pandas version (object / category / str / StringDtype).
        cat_dtype = passed_df["cat_x"].dtype
        assert not pd.api.types.is_numeric_dtype(cat_dtype), (
            f"Final processors must be fit on pre-encoded data — got {cat_dtype}"
        )
