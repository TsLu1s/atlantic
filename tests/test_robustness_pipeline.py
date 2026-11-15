"""Regression tests for pipeline-level robustness fixes.

Covers input validation, datetime handling (any resolution, timezone-aware,
builder flags honoured at transform time), dtype support, seed/parallelism
plumbing, refit state, transparency of removed columns, persistence and the
builder end-to-end.

Pipeline fits run with a shrunken model search (see ``quick_eval``) so each
one takes about a second; the tests assert on behaviour, not model quality.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import atlantic.pipeline.atlantic as atlantic_module
from atlantic.core.base import BasePipeline
from atlantic.core.exceptions import (
    AtlanticError,
    H2OConnectionError,
    NotFittedError,
    SchemaValidationError,
    StateSerializationError,
    ValidationError,
)
from atlantic.optimization import dimensionality as dim
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.pipeline import Atlantic, AtlanticBuilder, AtlanticPipeline
from atlantic.pipeline.pattern import Pattern
from atlantic.preprocessing.imputers import AutoKNNImputer
from atlantic.preprocessing.scalers import AutoMinMaxScaler
from atlantic.state.pipeline_state import StateManager
from tests._h2o_stub import importance_frame, make_h2o_stub


@pytest.fixture
def quick_eval(monkeypatch):
    """Shrink the internal model search so a full pipeline fit takes ~1 second."""
    small = dim.ModelHyperparameterRanges(
        rf_n_estimators=(5, 8),
        rf_max_depth=(2, 4),
        rf_min_samples_split=(2, 4),
        xgb_n_estimators=(5, 8),
        xgb_max_depth=(2, 3),
        xgb_learning_rate=(0.05, 0.1),
    )
    monkeypatch.setattr(
        dim.ModelHyperparameterRanges,
        "for_complexity",
        classmethod(lambda cls, complexity: small),
    )
    for tier in dim.DimensionalityTier:
        monkeypatch.setitem(
            dim.DimensionalityStrategy.TRIAL_MATRIX[tier], dim.OptimizationLevel.FAST, 2
        )


def make_frame(n: int = 120, seed: int = 0, classification: bool = False) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "num_a": rng.randn(n),
            "num_b": rng.rand(n) * 100,
            "cat_x": rng.choice(list("ABCD"), n),
        }
    )
    if classification:
        df["target"] = rng.choice(["yes", "no"], n)
    else:
        df["target"] = df["num_a"] * 3 + rng.randn(n)
    return df


def with_dates(df: pd.DataFrame, tz=None) -> pd.DataFrame:
    """Add a datetime column with repeated values (so it is not ID-like)."""
    df = df.copy()
    when = pd.Series(pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(len(df)) % 40, "D"))
    df["when"] = when.dt.tz_localize(tz) if tz else when
    return df


def fit(df: pd.DataFrame, random_state: int = 42, **kwargs) -> Atlantic:
    atl = Atlantic(X=df, target="target", random_state=random_state, verbosity=0)
    atl.fit_processing(relevance=1.0, optimization_level="fast", **kwargs)
    return atl


# ============================================================================
# Input validation
# ============================================================================


class TestInputValidation:
    def test_missing_target_name_raises_clear_error(self):
        atl = Atlantic(X=make_frame())
        with pytest.raises(ValidationError, match="target column name"):
            atl.fit_processing(relevance=1.0)

    def test_empty_constructor_then_fit(self):
        with pytest.raises(ValidationError, match="DataFrame"):
            Atlantic().fit_processing(relevance=1.0)

    def test_data_attached_after_construction(self, quick_eval):
        atl = Atlantic(verbosity=0)
        atl.X, atl.target = make_frame(), "target"
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert atl.is_fitted and atl.pred_type == "Reg"
        assert atl.state.target == "target"

    def test_duplicate_column_names_rejected(self):
        df = make_frame()
        df = pd.concat([df, df[["num_a"]]], axis=1)
        with pytest.raises(ValidationError, match="duplicate column names"):
            Atlantic(X=df, target="target")

    @pytest.mark.parametrize("bad", [[1, 2, 3], "frame", {"a": [1]}])
    def test_non_dataframe_rejected(self, bad):
        with pytest.raises(ValidationError, match="pandas DataFrame"):
            Atlantic(X=bad, target="target")

    def test_empty_dataframe_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            Atlantic(X=pd.DataFrame(), target="target")

    @pytest.mark.parametrize(
        "kwargs, fragment",
        [
            ({"vif_ratio": 2.0}, "vif_threshold"),
            ({"optimization_level": "turbo"}, "optimization_level"),
            ({"relevance": 0.2}, "relevance"),
            ({"h2o_fs_models": 0}, "h2o_fs_models"),
        ],
    )
    def test_invalid_configuration_is_an_atlantic_error(self, kwargs, fragment):
        atl = Atlantic(X=make_frame(), target="target")
        params = {"relevance": 1.0, **kwargs}
        with pytest.raises(ValidationError, match=fragment) as excinfo:
            atl.fit_processing(**params)
        assert isinstance(excinfo.value, AtlanticError)
        assert not atl.is_fitted

    def test_h2o_selection_without_h2o_fails_fast_with_hint(self, monkeypatch):
        monkeypatch.setattr(atlantic_module, "is_h2o_available", lambda: False)
        atl = Atlantic(X=make_frame(), target="target")
        monkeypatch.setattr(
            atl, "_prepare_data", lambda *a, **k: pytest.fail("work started before the check")
        )
        with pytest.raises(H2OConnectionError, match=r"atlantic\[h2o\]") as excinfo:
            atl.fit_processing()  # default relevance=0.99 -> H2O requested
        assert "relevance=1.0" in str(excinfo.value)

    def test_single_sample_class_rejected(self):
        df = make_frame(classification=True)
        df.loc[0, "target"] = "rare"
        with pytest.raises(ValidationError, match="single sample") as excinfo:
            fit(df)
        assert "rare" in str(excinfo.value)

    def test_no_usable_features_rejected(self):
        df = pd.DataFrame(
            {
                "const": [1] * 50,
                "row_id": [f"id_{i}" for i in range(50)],
                "target": np.arange(50.0),
            }
        )
        with pytest.raises(ValidationError, match="No usable feature columns") as excinfo:
            fit(df)
        assert "const" in str(excinfo.value) and "row_id" in str(excinfo.value)

    def test_integer_coded_labels_warn_about_regression(self):
        df = make_frame(classification=True)
        df["target"] = (df["target"] == "yes").astype(int)
        with pytest.warns(UserWarning, match="REGRESSION"):
            atl = Atlantic(X=df, target="target")
        assert atl.pred_type == "Reg"

    @pytest.mark.parametrize("classification", [False, True])
    def test_ordinary_targets_do_not_warn(self, classification):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Atlantic(X=make_frame(classification=classification), target="target")

    def test_inert_chained_setters_warn(self):
        atl = Atlantic(X=make_frame(), target="target")
        with pytest.warns(FutureWarning, match="does not affect fit_processing"):
            assert atl.with_encoding(scaler="robust") is atl
        with pytest.warns(FutureWarning):
            atl.with_date_engineering().with_null_removal().with_feature_selection()
        with pytest.warns(FutureWarning):
            atl.with_imputation(n_neighbors=3).with_vif_filtering().with_optimization()
        assert atl._builder_config["scaler"] == "robust"

    def test_builder_rejects_out_of_range_values(self):
        with pytest.raises(ValidationError, match="vif_threshold"):
            AtlanticBuilder().with_vif_filtering(threshold=50.0).build()

    def test_unfitted_pipeline_errors(self, tmp_path):
        pipeline = AtlanticBuilder().build()
        with pytest.raises(NotFittedError):
            pipeline.transform(make_frame())
        with pytest.raises(NotFittedError):
            pipeline.save(tmp_path / "p.pkl")
        assert pipeline.atlantic is None

    def test_data_processing_rejects_non_dataframe(self, quick_eval):
        atl = fit(make_frame())
        with pytest.raises(ValidationError, match="pandas DataFrame"):
            atl.data_processing([[1, 2, 3]])


# ============================================================================
# Datetime handling
# ============================================================================


class TestDatetimeHandling:
    def test_default_resolution_datetimes_are_engineered(self, quick_eval):
        df = with_dates(make_frame())
        atl = fit(df)

        assert atl._datetime_cols == ["when"]
        assert {"when_month", "when_day_of_week", "when_day_of_year"} <= set(atl.cols)
        assert "when" not in atl.cols
        out = atl.data_processing(df.drop(columns=["target"]))
        assert all(dtype.kind in "fi" for dtype in out.dtypes)
        assert atl.fitted_components.datetime_columns == ["when"]

    def test_timezone_aware_datetimes(self, quick_eval):
        df = with_dates(make_frame(), tz="UTC")
        atl = fit(df)
        assert "when_month" in atl.cols
        assert len(atl.data_processing(df)) == len(df)

    def test_text_dates_at_inference_are_parsed(self, quick_eval):
        # A CSV round-trip turns datetime columns into text.
        df = with_dates(make_frame())
        atl = fit(df)
        future = df.drop(columns=["target"]).head(15)
        as_text = future.copy()
        as_text["when"] = as_text["when"].astype(str)

        pd.testing.assert_frame_equal(atl.data_processing(as_text), atl.data_processing(future))

    def test_missing_datetime_column_error_names_the_source(self, quick_eval):
        df = with_dates(make_frame())
        atl = fit(df)
        with pytest.raises(SchemaValidationError, match="datetime column") as excinfo:
            atl.data_processing(df.drop(columns=["target", "when"]))
        assert "'when'" in str(excinfo.value)

    def test_date_engineering_disabled(self, quick_eval):
        # Previously the raw datetime column reached the models and crashed.
        df = with_dates(make_frame())
        atl = fit(df, date_engineering=False)

        assert not any(col.startswith("when") for col in atl.cols)
        assert atl.dropped_columns["datetime"] == ["when"]
        out = atl.data_processing(df.drop(columns=["target"]))
        assert list(out.columns) == atl.cols

    def test_date_drop_false_passes_raw_column_through(self, quick_eval):
        df = with_dates(make_frame())
        atl = fit(df, date_drop=False)

        assert "when_month" in atl.cols and "when" not in atl.cols
        out = atl.data_processing(df)
        assert list(out.columns) == atl.cols + ["when", "target"]
        pd.testing.assert_series_equal(out["when"], df["when"])
        assert out[atl.cols].isnull().sum().sum() == 0

    def test_engineering_disabled_and_column_kept(self, quick_eval):
        df = with_dates(make_frame())
        pipeline = (
            AtlanticBuilder()
            .with_date_engineering(enabled=False, drop=False)
            .with_feature_selection(method="none")
            .with_optimization(optimization_level="fast", verbosity=0)
            .build()
        )
        out = pipeline.fit_transform(df, target="target")
        assert "when" in out.columns
        assert not any(col.startswith("when_") for col in out.columns)
        pd.testing.assert_series_equal(out["when"], df["when"])

    def test_refit_without_datetimes_forgets_them(self, quick_eval):
        atl = fit(with_dates(make_frame()))
        atl.X = make_frame()
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert atl._datetime_cols == []


# ============================================================================
# Dtype support
# ============================================================================


class TestDtypeSupport:
    def test_category_and_string_dtype_features(self, quick_eval):
        df = make_frame()
        rng = np.random.RandomState(1)
        df["as_category"] = pd.Categorical(rng.choice(list("xyz"), len(df), p=[0.6, 0.3, 0.1]))
        df["as_string"] = pd.array(rng.choice(["p", "q"], len(df)), dtype="string")
        df.loc[[1, 2], "as_string"] = pd.NA

        atl = fit(df)
        assert {"cat_x", "as_category", "as_string"} <= set(atl.c_cols)
        out = atl.data_processing(df.drop(columns=["target"]))
        assert all(dtype.kind in "fi" for dtype in out.dtypes)
        assert out.isnull().sum().sum() == 0

    def test_nullable_boolean_is_scaled_and_imputed(self, quick_eval):
        df = make_frame()
        rng = np.random.RandomState(2)
        df["flag"] = pd.array(rng.rand(len(df)) > 0.5, dtype="boolean")
        df.loc[[3, 4], "flag"] = pd.NA
        df["plain_flag"] = rng.rand(len(df)) > 0.5
        original_dtypes = df.dtypes.copy()

        atl = fit(df)
        assert {"flag", "plain_flag"} <= set(atl.n_cols)
        out = atl.data_processing(df.drop(columns=["target"]))
        assert out.isnull().sum().sum() == 0
        assert out["flag"].dtype == np.float64
        # neither fit nor transform touched the caller's frame
        pd.testing.assert_series_equal(df.dtypes, original_dtypes)

    def test_unsigned_integers_are_treated_as_numeric(self, quick_eval):
        df = make_frame()
        df["u8"] = np.random.RandomState(3).randint(0, 200, len(df)).astype("uint8")
        atl = fit(df)
        assert "u8" in atl.n_cols
        out = atl.data_processing(df.drop(columns=["target"]))
        assert out["u8"].dtype == np.float64
        assert abs(out["u8"].mean()) < 1.0  # scaled, no longer 0..200

    def test_numeric_text_at_inference_is_coerced(self, quick_eval):
        df = make_frame()
        atl = fit(df)
        future = df.drop(columns=["target"]).head(10)
        as_text = future.copy()
        as_text["num_b"] = as_text["num_b"].astype(str)
        pd.testing.assert_frame_equal(atl.data_processing(as_text), atl.data_processing(future))

    def test_non_numeric_values_in_numeric_column_raise_schema_error(self, quick_eval):
        df = make_frame()
        atl = fit(df)
        future = df.drop(columns=["target"]).head(5).astype({"num_b": object})
        future.loc[0, "num_b"] = "oops"
        with pytest.raises(SchemaValidationError, match="num_b"):
            atl.data_processing(future)

    def test_transform_does_not_mutate_input(self, quick_eval):
        df = with_dates(make_frame())
        df.loc[[0, 1], "num_a"] = np.nan
        atl = fit(df)
        future = df.drop(columns=["target"]).head(20)
        snapshot = future.copy()
        atl.data_processing(future)
        pd.testing.assert_frame_equal(future, snapshot)

    def test_unseen_category_and_missing_values_at_inference(self, quick_eval):
        df = make_frame()
        atl = fit(df)
        future = df.drop(columns=["target"]).head(6).astype({"cat_x": object})
        future.loc[0, "cat_x"] = np.nan
        future.loc[1, "cat_x"] = "NEVER_SEEN"
        future.loc[2, "num_a"] = np.nan
        out = atl.data_processing(future)
        assert out.isnull().sum().sum() == 0


# ============================================================================
# Reproducibility and fit state
# ============================================================================


class TestReproducibilityAndState:
    def test_seed_and_parallelism_reach_every_evaluation(self, quick_eval, monkeypatch):
        seen = []
        original_init = ModelEvaluator.__init__

        def spy(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            seen.append((self._random_state, self._n_jobs))

        monkeypatch.setattr(ModelEvaluator, "__init__", spy)

        df = make_frame()
        df.loc[[1, 2, 3], "num_a"] = np.nan  # forces the imputation comparison too
        atl = Atlantic(X=df, target="target", random_state=7, n_jobs=1, verbosity=0)
        atl.fit_processing(relevance=1.0, optimization_level="fast")

        # 4 encodings + KNN + Iterative (mean imputation reuses a known score)
        assert len(seen) >= 6
        assert {seed for seed, _ in seen} == {7}, "evaluations ignored the pipeline seed"
        assert {jobs for _, jobs in seen} == {1}
        assert atl.state.config.random_state == 7

    def test_same_seed_reproduces_the_pipeline(self, quick_eval):
        df = make_frame()
        df.loc[[5, 6], "num_b"] = np.nan
        first, second = fit(df, random_state=3), fit(df, random_state=3)
        assert (first.enc_method, first.imp_method) == (second.enc_method, second.imp_method)
        pd.testing.assert_frame_equal(first.data_processing(df), second.data_processing(df))

    # Thread-parallel forests can trip a spurious scikit-learn warning on
    # Python < 3.14; it is unrelated to what this test checks.
    @pytest.mark.filterwarnings("ignore:.*sklearn.utils.parallel.delayed.*:UserWarning")
    def test_results_do_not_depend_on_n_jobs(self, quick_eval):
        train, test = make_frame(seed=1), make_frame(n=40, seed=2)
        scores = [
            ModelEvaluator(
                train.drop(columns="cat_x"),
                test.drop(columns="cat_x"),
                "target",
                optimization_level="fast",
                n_jobs=n_jobs,
            ).auto_evaluate()
            for n_jobs in (None, 1, 2)
        ]
        pd.testing.assert_frame_equal(scores[0], scores[1])
        pd.testing.assert_frame_equal(scores[0], scores[2])

    def test_default_keeps_forests_single_threaded(self):
        assert Atlantic(X=make_frame(), target="target")._n_jobs is None
        assert AtlanticBuilder().get_config().optimizer.n_jobs is None
        evaluator = ModelEvaluator(make_frame().drop(columns="cat_x"), make_frame(n=20), "target")
        models = evaluator._get_model_instances()
        assert models["rf_regressor"].n_jobs is None and models["et_regressor"].n_jobs is None

    def test_reused_mean_imputation_score_equals_a_fresh_evaluation(self, quick_eval):
        # The imputation comparison reuses the score the encoding selection
        # measured for mean imputation; it must be exactly what a fresh
        # evaluation of that candidate would produce.
        df = make_frame()
        df.loc[[1, 2, 3, 4, 5], "num_a"] = np.nan
        atl = Atlantic(X=df, target="target", random_state=5, verbosity=0)
        prepared = atl._prepare_data(df)
        train, test = atl._split_data(prepared, 0.75)

        selection = Pattern(train, test, "target", "fast", verbosity=0, random_state=5)
        selection.encoding_selection()
        known = selection.perf

        fresh = Pattern(train, test, "target", "fast", verbosity=0, random_state=5)
        fresh.enc_method = selection.enc_method
        fresh.imputation_selection()
        reused = Pattern(train, test, "target", "fast", verbosity=0, random_state=5)
        reused.enc_method = selection.enc_method
        reused.imputation_selection(simple_score=known)

        assert (reused.imp_method, reused.perf) == (fresh.imp_method, fresh.perf)
        pd.testing.assert_frame_equal(reused.train, fresh.train)

    def test_refit_starts_from_a_clean_state(self, quick_eval):
        df = make_frame()
        atl = fit(df)
        assert atl._perf is not None and atl.enc_method.startswith("Encoding Version")

        atl.fit_processing(
            relevance=1.0,
            optimization_level="fast",
            auto_encoding=False,
            scaler="robust",
            encoder="label",
            auto_imputation=False,
            imputer="simple",
        )
        # No score was measured in this configuration: the one from the first
        # fit must not survive as the VIF baseline.
        assert atl._perf is None
        assert atl.enc_method == "Manual (robust + label)"
        assert atl.is_fitted

    def test_failed_refit_leaves_pipeline_unfitted(self, quick_eval, monkeypatch):
        df = make_frame()
        atl = fit(df)
        monkeypatch.setattr(
            atl, "_apply_vif_filtering", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with pytest.raises(RuntimeError):
            atl.fit_processing(relevance=1.0, optimization_level="fast")
        assert not atl.is_fitted and not atl.state.is_fitted
        with pytest.raises(NotFittedError):
            atl.data_processing(df)

    def test_manual_imputer_discards_score_measured_with_another_imputer(
        self, quick_eval, monkeypatch
    ):
        captured = {}
        original = Pattern.vif_performance

        def spy(self, vif_threshold=10.0, perf_=None):
            captured["perf_"] = perf_
            return original(self, vif_threshold=vif_threshold, perf_=perf_)

        monkeypatch.setattr(Pattern, "vif_performance", spy)

        df = make_frame()
        df.loc[[1, 2, 3, 4], "num_a"] = np.nan
        atl = fit(df, auto_encoding=True, auto_imputation=False, imputer="knn")

        assert atl.imp_method == "KNN" and isinstance(atl.imputer, AutoKNNImputer)
        assert captured["perf_"] is None

    def test_removed_columns_are_reported(self, quick_eval, capsys):
        df = make_frame()
        df["const"] = 1
        df["row_id"] = [f"id_{i}" for i in range(len(df))]
        df["mostly_null"] = np.nan
        df.loc[:10, "mostly_null"] = 1.5
        df.loc[5, "mostly_null"] = 2.5
        df["all_null"] = np.nan

        atl = Atlantic(X=df, target="target", verbosity=1)
        atl.fit_processing(relevance=1.0, optimization_level="fast", null_threshold=0.5)

        assert atl.dropped_columns == {
            "high_null": ["mostly_null", "all_null"],
            "constant": ["const"],
            "unique_identifier": ["row_id"],
        }
        assert atl.fitted_components.dropped_columns == atl.dropped_columns
        assert set(atl.cols) == {"num_a", "num_b", "cat_x"}
        assert "Columns removed before modelling" in capsys.readouterr().out

    def test_null_threshold_of_one_only_removes_fully_null_columns(self):
        df = make_frame()
        df["sparse"] = np.nan
        df.loc[0, "sparse"] = 1.0
        df["all_null"] = np.nan
        atl = Atlantic(X=df, target="target", verbosity=0)
        kept = atl._remove_high_null_columns(df, threshold=1.0)
        assert "sparse" in kept.columns and "all_null" not in kept.columns

    def test_null_threshold_boundary_is_exact(self):
        # 90 of 100 values missing: exactly at a 0.9 threshold -> kept.
        df = pd.DataFrame({"edge": [1.0] * 10 + [np.nan] * 90, "target": np.arange(100.0)})
        atl = Atlantic(X=df, target="target", verbosity=0)
        assert "edge" in atl._remove_high_null_columns(df, threshold=0.9).columns
        assert "edge" not in atl._remove_high_null_columns(df, threshold=0.89).columns

    def test_vif_removals_are_recorded_consistently(self, quick_eval):
        df = make_frame(n=200)
        df["num_a_copy"] = df["num_a"] * 2 + np.random.RandomState(9).randn(len(df)) * 1e-6
        atl = fit(df)
        removed = atl.dropped_columns.get("multicollinearity", [])
        assert set(removed).isdisjoint(atl.cols)
        assert set(atl.cols) | set(removed) == {"num_a", "num_b", "cat_x", "num_a_copy"}

    def test_verbosity_zero_is_silent(self, quick_eval, capsys):
        df = make_frame()
        df.loc[[1, 2], "num_a"] = np.nan
        fit(df)
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_default_verbosity_reports_selection(self, quick_eval, capsys):
        atl = Atlantic(X=make_frame(), target="target")
        atl.fit_processing(relevance=1.0, optimization_level="fast")
        out = capsys.readouterr().out
        assert "Encoding Version" in out and "There are no missing values" in out

    def test_fit_does_not_mutate_training_frame(self, quick_eval):
        df = with_dates(make_frame(classification=True))
        df["flag"] = np.random.RandomState(4).rand(len(df)) > 0.5
        snapshot = df.copy()
        fit(df)
        pd.testing.assert_frame_equal(df, snapshot)


# ============================================================================
# Persistence
# ============================================================================


class TestPersistence:
    def test_roundtrip_is_silent_and_version_stamped(self, quick_eval, tmp_path):
        df = with_dates(make_frame())
        atl = fit(df, date_drop=False)
        path = tmp_path / "atl.pkl"
        atl.save(str(path))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = Atlantic.load(str(path))

        assert loaded._atlantic_version == atlantic_module.__version__
        assert loaded.X is None and atl.X is not None
        pd.testing.assert_frame_equal(loaded.data_processing(df), atl.data_processing(df))

    def test_version_mismatch_warns(self, quick_eval, tmp_path, monkeypatch):
        atl = fit(make_frame())
        path = tmp_path / "atl.pkl"
        atl.save(str(path))

        monkeypatch.setattr(atlantic_module, "__version__", "9.9.9")
        with pytest.warns(UserWarning, match="saved by Atlantic") as record:
            Atlantic.load(str(path))
        assert "9.9.9" in str(record[0].message)

    def test_pipeline_saved_by_an_older_version_still_transforms(self, quick_eval):
        df = with_dates(make_frame())
        atl = fit(df)
        expected = atl.data_processing(df)

        state = atl.__getstate__()
        for attr in (
            "_atlantic_version",
            "_date_engineering",
            "_date_drop",
            "_datetime_cols",
            "_n_jobs",
            "_verbosity",
            "dropped_columns",
            "_perf",
        ):
            state.pop(attr)

        legacy = Atlantic.__new__(Atlantic)
        with pytest.warns(UserWarning, match="older Atlantic version"):
            legacy.__setstate__(state)

        assert legacy._datetime_cols is None and legacy.dropped_columns == {}
        pd.testing.assert_frame_equal(legacy.data_processing(df), expected)

    def test_builder_pipeline_save_and_load(self, quick_eval, tmp_path):
        df = make_frame()
        df.loc[[1, 2], "num_a"] = np.nan
        pipeline = (
            AtlanticBuilder()
            .with_feature_selection(method="none")
            .with_optimization(optimization_level="fast", verbosity=0)
            .build()
        )
        assert pipeline.fit(df, target="target") is pipeline
        path = tmp_path / "pipeline.pkl"
        pipeline.save(path)

        loaded = AtlanticPipeline.load(path)
        assert loaded.is_fitted and loaded.atlantic.X is None
        assert loaded.config == pipeline.config
        pd.testing.assert_frame_equal(loaded.transform(df), pipeline.transform(df))

    def test_loading_the_wrong_kind_of_file(self, quick_eval, tmp_path):
        atl = fit(make_frame())
        path = tmp_path / "atl.pkl"
        atl.save(str(path))
        with pytest.raises(StateSerializationError, match="Atlantic.load"):
            AtlanticPipeline.load(path)

    def test_metadata_export_roundtrip(self, quick_eval, tmp_path):
        df = make_frame()
        df["const"] = 0
        atl = fit(df)

        StateManager.save_metadata(atl.state, tmp_path / "meta.json")
        loaded = StateManager.load_metadata(tmp_path / "meta.json")
        assert loaded.fitted_components.selected_columns == atl.cols
        assert loaded.fitted_components.dropped_columns == {"constant": ["const"]}
        assert loaded.config.random_state == 42

        StateManager.export_config(atl.state, tmp_path / "config.json")
        assert '"dropped_columns"' in (tmp_path / "config.json").read_text()

    def test_feature_importance_survives_json_roundtrip(self, tmp_path):
        from atlantic.core.schemas import PipelineState

        importance = pd.DataFrame({"variable": ["a", "b"], "percentage": [0.7, 0.3]})
        state = PipelineState(feature_importance=importance)
        state.save(str(tmp_path / "state.json"))
        loaded = PipelineState.load(str(tmp_path / "state.json"))
        pd.testing.assert_frame_equal(loaded.feature_importance, importance)

    def test_pickle_excludes_training_data(self, quick_eval):
        df = make_frame()
        atl = fit(df)
        restored = pickle.loads(pickle.dumps(atl))
        assert restored.X is None
        assert restored.is_fitted


# ============================================================================
# Builder end-to-end
# ============================================================================


class TestBuilderEndToEnd:
    def test_manual_classification_pipeline(self, quick_eval):
        df = make_frame(classification=True)
        df.loc[[1, 2, 3], "num_b"] = np.nan
        pipeline = (
            AtlanticBuilder()
            .with_feature_selection(method="none")
            .with_encoding(scaler="minmax", encoder="label", auto_select=False)
            .with_imputation(method="simple", auto_select=False, strategy="median")
            .with_optimization(optimization_level="fast", random_state=11, n_jobs=1, verbosity=0)
            .build()
        )
        out = pipeline.fit_transform(df, target="target")
        atl = pipeline.atlantic

        assert (atl._random_state, atl._n_jobs, atl._verbosity) == (11, 1, 0)
        assert atl.eval_metric == "Precision" and atl.n_classes == 2
        assert atl.imputer.strategy == "median"
        assert out["target"].tolist() == df["target"].tolist()  # target untouched
        assert out.drop(columns=["target"]).isnull().sum().sum() == 0
        # Min-max scaled with statistics from the TRAIN split only, so rows
        # outside that split may fall slightly outside [0, 1].
        assert isinstance(atl.scaler, AutoMinMaxScaler)
        assert out["num_a"].between(-0.5, 1.5).all()

    def test_multiclass_auto_pipeline(self, quick_eval):
        rng = np.random.RandomState(5)
        df = make_frame(n=150)
        df["target"] = rng.choice(["a", "b", "c"], len(df))
        atl = fit(df)
        assert atl.eval_metric == "F1" and atl.n_classes == 3
        assert len(atl.data_processing(df)) == len(df)

    def test_h2o_feature_selection_path(self, quick_eval):
        df = make_frame()
        stub = make_h2o_stub(
            {
                "gbm": importance_frame(
                    [
                        ("num_a", 9.0, 1.0, 0.60),
                        ("cat_x", 5.0, 0.5, 0.39),
                        ("num_b", 0.1, 0.01, 0.01),
                    ]
                )
            }
        )
        with patch.dict(sys.modules, stub):
            atl = Atlantic(X=df, target="target", verbosity=0)
            atl.fit_processing(relevance=0.9, h2o_fs_models=3, optimization_level="fast")

        assert atl.dropped_columns["feature_selection"] == ["num_b"]
        assert set(atl.cols) <= {"num_a", "cat_x"}
        assert atl.h2o_feature_importance["variable"].tolist() == ["num_a", "cat_x"]
        assert atl.state.feature_importance is atl.h2o_feature_importance
        assert stub["h2o"].automl_kwargs["max_models"] == 3
        out = atl.data_processing(df.drop(columns=["target"]))
        assert list(out.columns) == atl.cols


# ============================================================================
# Base template and evaluator
# ============================================================================


class _IdentityPipeline(BasePipeline):
    """Smallest concrete pipeline: every optimisation hook is a no-op."""

    def _select_features(self, train, test, relevance, h2o_fs_models, encoding_fs):
        self.selected = True
        return train, test

    def _optimize_encoding(self, train, test, optimization_level):
        return train, test

    def _optimize_imputation(self, train, test, optimization_level):
        return train, test

    def _apply_vif_filtering(self, train, test, vif_threshold):
        return train, test

    def _fit_final_processors(self, X):
        self.train_columns = list(X.columns)
        self.train_rows = len(X)

    def _transform_data(self, X):
        return X[[c for c in self.train_columns if c in X.columns]]


class TestBaseTemplate:
    def test_template_flow(self):
        df = with_dates(make_frame(n=100, classification=True))
        df["target"] = df["target"].astype(object)
        df.loc[[0, 1, 2, 3], "target"] = np.nan

        pipeline = _IdentityPipeline(X=df, target="target", random_state=3)
        with pytest.raises(NotFittedError):
            pipeline.data_processing(df)

        assert pipeline.fit_processing(split_ratio=0.75, relevance=0.9) is pipeline
        assert pipeline.is_fitted and pipeline.state.is_fitted and pipeline.selected
        # missing-target rows dropped before the cast: no phantom "nan" class
        assert pipeline.n_classes == 2
        assert pipeline.train_rows == 72  # 75% of the 96 labelled rows
        assert pipeline.train_columns[-1] == "target"
        assert "when_month" in pipeline.train_columns and "when" not in pipeline.train_columns
        assert pipeline._input_schema["n_features"] == df.shape[1]
        assert list(pipeline.data_processing(df).columns) == [
            c for c in pipeline.train_columns if c in df.columns
        ]

    def test_split_uses_the_instance_seed(self):
        df = make_frame()
        a = _IdentityPipeline(X=df, target="target", random_state=1)
        b = _IdentityPipeline(X=df, target="target", random_state=2)
        assert not a._split_data(df, 0.75)[0].equals(b._split_data(df, 0.75)[0])
        pd.testing.assert_frame_equal(a._split_data(df, 0.75)[0], a._split_data(df, 0.75)[0])


class TestModelEvaluatorRobustness:
    def test_test_only_class_raises_clear_error(self):
        train = pd.DataFrame({"x": np.arange(10.0), "target": ["a", "b"] * 5})
        test = pd.DataFrame({"x": np.arange(4.0), "target": ["a", "b", "c", "a"]})
        evaluator = ModelEvaluator(train, test, "target", optimization_level="fast")
        with pytest.raises(ValidationError, match="absent from the train split"):
            evaluator.auto_evaluate()

    def test_repeated_evaluation_is_idempotent(self, quick_eval):
        train, test = make_frame(seed=1), make_frame(n=40, seed=2)
        evaluator = ModelEvaluator(
            train.drop(columns="cat_x"),
            test.drop(columns="cat_x"),
            "target",
            optimization_level="fast",
        )
        first = evaluator.auto_evaluate()
        second = evaluator.auto_evaluate()
        pd.testing.assert_frame_equal(first, second)
        assert len(evaluator.hparameters_list) == evaluator.dimensionality_config.n_trials
        assert evaluator.detailed_metrics is not None

    def test_vif_failure_is_reported_not_swallowed(self, quick_eval):
        train, test = make_frame(seed=1), make_frame(n=40, seed=2)
        train.loc[0, "num_a"] = np.nan  # VIF cannot be computed with nulls
        pattern = Pattern(
            train.drop(columns="cat_x"),
            test.drop(columns="cat_x"),
            "target",
            optimization_level="fast",
            verbosity=0,
        )
        with pytest.warns(UserWarning, match="VIF filtering skipped"):
            filtered_train, _ = pattern.vif_performance()
        assert list(filtered_train.columns) == ["num_a", "num_b", "target"]
        assert pattern.vif_removed == [] and pattern.vif_applied is False
