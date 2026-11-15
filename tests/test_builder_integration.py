"""End-to-end tests for the AtlanticBuilder configuration plumbing and for the
critical-bug regressions fixed in 2.0.31.

All tests use ``with_feature_selection(method="none")`` so they never require an
H2O cluster / Java.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlantic.core.exceptions import SchemaValidationError, ValidationError
from atlantic.pipeline import Atlantic, AtlanticBuilder
from atlantic.preprocessing.imputers import AutoKNNImputer, AutoSimpleImputer
from atlantic.preprocessing.scalers import AutoRobustScaler, AutoStandardScaler


@pytest.fixture
def reg_df():
    np.random.seed(0)
    n = 160
    df = pd.DataFrame(
        {
            "num_a": np.random.randn(n),
            "num_b": np.random.rand(n) * 100,
            "cat_x": np.random.choice(list("ABCD"), n),
            "target": np.random.randn(n) * 10 + 50,
        }
    )
    df.loc[np.random.choice(n, 12, replace=False), "num_a"] = np.nan
    return df


class TestBuilderConfigHonored:
    def test_manual_encoding_and_imputation_are_applied(self, reg_df):
        pipe = (
            AtlanticBuilder()
            .with_feature_selection(method="none")
            .with_encoding(scaler="robust", encoder="label", auto_select=False)
            .with_imputation(method="knn", auto_select=False, n_neighbors=4)
            .with_optimization(optimization_level="fast")
            .build()
        )
        out = pipe.fit_transform(reg_df.copy(), target="target")
        atl = pipe._atlantic

        assert isinstance(atl.scaler, AutoRobustScaler)
        assert atl.imp_method == "KNN"
        assert isinstance(atl.imputer, AutoKNNImputer)
        assert atl.imputer.n_neighbors == 4
        assert out.shape[0] == reg_df.shape[0]

    def test_null_removal_threshold_is_honored(self, reg_df):
        # Column that is 90% null should be dropped at threshold 0.85.
        df = reg_df.copy()
        df["mostly_null"] = np.nan
        df.loc[: int(len(df) * 0.1), "mostly_null"] = 1.0

        pipe = (
            AtlanticBuilder()
            .with_feature_selection(method="none")
            .with_null_removal(threshold=0.85)
            .with_optimization(optimization_level="fast")
            .build()
        )
        pipe.fit_transform(df, target="target")
        assert "mostly_null" not in pipe._atlantic.cols

    def test_auto_builder_still_runs(self, reg_df):
        pipe = (
            AtlanticBuilder()
            .with_feature_selection(method="none")
            .with_optimization(optimization_level="fast")
            .build()
        )
        out = pipe.fit_transform(reg_df.copy(), target="target")
        atl = pipe._atlantic
        assert atl.enc_method.startswith("Encoding Version")
        assert isinstance(atl.scaler, (AutoStandardScaler, AutoRobustScaler)) or atl.scaler is not None
        assert out.shape[0] == reg_df.shape[0]

    def test_onehot_manual_is_rejected(self, reg_df):
        with pytest.raises(ValidationError):
            (
                AtlanticBuilder()
                .with_feature_selection(method="none")
                .with_encoding(encoder="onehot", auto_select=False)
                .build()
            ).fit_transform(reg_df.copy(), target="target")


class TestCriticalRegressions:
    def test_fit_does_not_mutate_caller_dataframe(self):
        """H2: classification target must not be cast in the caller's frame."""
        np.random.seed(1)
        n = 120
        df = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "cat_x": np.random.choice(list("AB"), n),
                "target": np.random.choice([0, 1], n),  # numeric labels
            }
        )
        original_dtype = df["target"].dtype
        atl = Atlantic(X=df, target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert df["target"].dtype == original_dtype, "caller's target dtype was mutated"

    def test_missing_target_rows_do_not_create_phantom_class(self):
        """C3: NaN target rows must be dropped, not become a 'nan' class."""
        np.random.seed(2)
        n = 150
        df = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "cat_x": np.random.choice(list("AB"), n),
                "target": np.random.choice(["yes", "no"], n).astype(object),
            }
        )
        df.loc[np.random.choice(n, 15, replace=False), "target"] = np.nan
        atl = Atlantic(X=df, target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert atl.n_classes == 2, "a phantom 'nan' class leaked into n_classes"

    def test_continuous_float_features_are_not_dropped(self):
        """C2: all-unique float columns must survive column pruning."""
        np.random.seed(3)
        n = 100
        df = pd.DataFrame(
            {
                "cont_a": np.random.randn(n),  # essentially all-unique floats
                "cont_b": np.random.rand(n),
                "target": np.random.randn(n),
            }
        )
        atl = Atlantic(X=df, target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert "cont_a" in atl.cols and "cont_b" in atl.cols

    def test_inference_nulls_are_imputed_even_when_train_had_none(self):
        """H8: a fallback imputer must handle NaNs seen only at inference."""
        np.random.seed(4)
        n = 120
        df = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "num_b": np.random.rand(n) * 5,
                "target": np.random.randn(n),
            }
        )  # no nulls in training
        atl = Atlantic(X=df, target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert atl.imputer is not None

        future = df.drop(columns=["target"]).head(10).copy()
        future.iloc[0, future.columns.get_loc("num_a")] = np.nan
        out = atl.data_processing(future)
        assert out.isnull().sum().sum() == 0, "inference NaN was not imputed"

    def test_saved_pipeline_excludes_training_data(self, tmp_path):
        """Saved pipelines must not embed the raw training frame."""
        np.random.seed(6)
        n = 120
        df = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "cat_x": np.random.choice(list("AB"), n),
                "target": np.random.randn(n),
            }
        )
        atl = Atlantic(X=df, target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert atl.X is not None  # still available on the live object

        path = tmp_path / "atl.pkl"
        atl.save(str(path))
        loaded = Atlantic.load(str(path))
        assert loaded.X is None, "training data must not be serialized"
        # still fully usable for inference
        out = loaded.data_processing(df.drop(columns=["target"]).head(5))
        assert len(out) == 5

    def test_missing_column_raises_schema_error(self):
        """H5: a clear SchemaValidationError instead of a bare KeyError."""
        np.random.seed(5)
        n = 100
        df = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "num_b": np.random.rand(n),
                "target": np.random.randn(n),
            }
        )
        atl = Atlantic(X=df, target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        bad = df.drop(columns=["target", "num_b"]).head(5)
        with pytest.raises(SchemaValidationError):
            atl.data_processing(bad)
