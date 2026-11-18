"""Regression tests for component-level robustness fixes.

Each test pins a defect that was reproduced before the fix: dtype detection
across pandas versions, encoder / imputer / scaler edge cases, the VIF
selector, binary metrics with non-0/1 labels, the Optuna logging side effect
and the H2O selector's failure modes (exercised through a stub - no JVM).
"""

from __future__ import annotations

import logging
import sys
import warnings
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import pytest

from atlantic.core.exceptions import (
    AtlanticError,
    FeatureSelectionError,
    H2OConnectionError,
    NotFittedError,
    SchemaValidationError,
    ValidationError,
)
from atlantic.data.generator import DatasetGenerator
from atlantic.evaluation.metrics import metrics_classification
from atlantic.feature_selection.h2o_selector import (
    H2O_INSTALL_HINT,
    H2OFeatureSelector,
    is_h2o_available,
)
from atlantic.feature_selection.vif_selector import VIFFeatureSelector, compute_vif
from atlantic.optimization.optuna_adapter import OptunaAdapter
from atlantic.preprocessing.encoders import (
    AutoIFrequencyEncoder,
    AutoLabelEncoder,
    AutoOneHotEncoder,
)
from atlantic.preprocessing.imputers import AutoIterativeImputer, AutoKNNImputer, AutoSimpleImputer
from atlantic.preprocessing.scalers import AutoMinMaxScaler, AutoRobustScaler, AutoStandardScaler
from atlantic.utils.columns import (
    get_boolean_columns,
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    is_numeric_target,
)
from atlantic.utils.datetime import engineer_datetime_features
from atlantic.utils.validation import validate_numeric_columns
from tests._h2o_stub import importance_frame, make_h2o_stub

IMPUTERS = [AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer]
SCALERS = [AutoStandardScaler, AutoMinMaxScaler, AutoRobustScaler]


@pytest.fixture
def typed_df():
    """One column of every dtype family the detection helpers must classify."""
    n = 6
    return pd.DataFrame(
        {
            "i8": np.arange(n, dtype="int8"),
            "u8": np.arange(n, dtype="uint8"),
            "f32": np.arange(n, dtype="float32"),
            "Int64": pd.array([1, 2, None, 4, 5, 6], dtype="Int64"),
            "Float64": pd.array([1.0, None, 3, 4, 5, 6], dtype="Float64"),
            "flag": [True, False] * 3,
            "flag_nullable": pd.array([True, None, False] * 2, dtype="boolean"),
            "text": list("abcabc"),
            "obj": pd.Series(list("abcabc"), dtype=object),
            "cat": pd.Categorical(list("abcabc")),
            "string": pd.array(list("abcabc"), dtype="string"),
            "dt_ns": pd.date_range("2023-01-01", periods=n, freq="h").astype("datetime64[ns]"),
            "dt_s": pd.date_range("2023-01-01", periods=n, freq="h").astype("datetime64[s]"),
            "dt_tz": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "delta": pd.to_timedelta(np.arange(n), unit="D"),
        }
    )


# ============================================================================
# Column type detection
# ============================================================================


class TestColumnTypeDetection:
    def test_numeric_covers_all_widths_unsigned_and_nullable(self, typed_df):
        assert get_numeric_columns(typed_df) == ["i8", "u8", "f32", "Int64", "Float64"]

    def test_numeric_excludes_bool_timedelta_and_categoricals(self, typed_df):
        numeric = get_numeric_columns(typed_df)
        for col in ("flag", "flag_nullable", "delta", "cat", "text", "dt_ns"):
            assert col not in numeric

    def test_categorical_covers_object_category_and_string_dtypes(self, typed_df):
        assert get_categorical_columns(typed_df) == ["text", "obj", "cat", "string"]

    def test_boolean_columns(self, typed_df):
        assert get_boolean_columns(typed_df) == ["flag", "flag_nullable"]

    def test_boolean_valued_category_is_categorical_not_boolean(self):
        df = pd.DataFrame({"c": pd.Categorical([True, False, True])})
        assert get_boolean_columns(df) == []
        assert get_categorical_columns(df) == ["c"]

    def test_datetime_any_resolution_and_timezone(self, typed_df):
        assert get_datetime_columns(typed_df) == ["dt_ns", "dt_s", "dt_tz"]

    def test_default_date_range_is_detected(self):
        df = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=4, freq="D")})
        assert get_datetime_columns(df) == ["when"]

    def test_exclude_is_honoured(self, typed_df):
        assert "i8" not in get_numeric_columns(typed_df, exclude=["i8"])
        assert "text" not in get_categorical_columns(typed_df, exclude=["text"])
        assert "dt_ns" not in get_datetime_columns(typed_df, exclude=["dt_ns"])

    def test_detection_emits_no_pandas_deprecation_warnings(self, typed_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_numeric_columns(typed_df)
            get_categorical_columns(typed_df)
            get_datetime_columns(typed_df)
            engineer_datetime_features(typed_df)

    @pytest.mark.parametrize(
        "values, dtype, expected",
        [
            ([1, 2, 3], "uint16", True),
            ([1.5, 2.5, 3.5], "float32", True),
            ([True, False, True], "bool", False),
            ([True, None, False], "boolean", False),
            (["a", "b", "a"], "category", False),
        ],
    )
    def test_is_numeric_target(self, values, dtype, expected):
        assert is_numeric_target(pd.Series(values, dtype=dtype)) is expected

    def test_validate_numeric_columns_accepts_narrow_and_nullable(self, typed_df):
        validate_numeric_columns(typed_df, ["i8", "u8", "f32", "Int64", "Float64"])
        with pytest.raises(ValidationError):
            validate_numeric_columns(typed_df, ["flag"])


# ============================================================================
# Datetime engineering
# ============================================================================


class TestDatetimeEngineering:
    @pytest.mark.parametrize("column", ["dt_ns", "dt_s", "dt_tz"])
    def test_every_datetime_flavour_is_expanded(self, typed_df, column):
        out = engineer_datetime_features(typed_df[[column]])
        assert column not in out.columns
        assert f"{column}_year" in out.columns
        assert out[f"{column}_year"].tolist() == [2023] * 6
        assert out[f"{column}_hour"].tolist() == list(range(6))

    def test_weekend_flag_and_weekday_numbering(self):
        # 2024-01-06 is a Saturday, 2024-01-08 a Monday.
        df = pd.DataFrame({"d": pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"])})
        out = engineer_datetime_features(df)
        assert out["d_day_of_week"].tolist() == [6, 7, 1]
        assert out["d_is_wknd"].tolist() == [1, 1, 0]

    def test_input_is_not_mutated(self, typed_df):
        before = typed_df.copy()
        engineer_datetime_features(typed_df)
        pd.testing.assert_frame_equal(typed_df, before)


# ============================================================================
# Encoders
# ============================================================================


class TestIFrequencyEncoderRobustness:
    def test_category_dtype_yields_float_output(self):
        df = pd.DataFrame({"c": pd.Categorical(list("aaabbc") * 5)})
        out = AutoIFrequencyEncoder().fit(df).transform(df)
        assert out["c"].dtype == np.float64

    def test_unobserved_categories_do_not_divide_by_zero(self):
        cat = pd.Categorical(["a", "a", "b"], categories=["a", "b", "never_seen"])
        df = pd.DataFrame({"c": cat})
        out = AutoIFrequencyEncoder().fit(df).transform(df)
        assert np.isfinite(out["c"]).all()

    @pytest.mark.parametrize("dtype", [object, "category", "string"])
    def test_matches_reference_formula(self, dtype):
        train = pd.DataFrame({"c": pd.Series(["a"] * 6 + ["b"] * 3 + ["c"], dtype=dtype)})
        new = pd.DataFrame({"c": pd.Series(["a", "b", "c", "zzz", None], dtype=object)})
        out = AutoIFrequencyEncoder().fit(train).transform(new)["c"]

        expected = [np.log1p(10 / 6), np.log1p(10 / 3), np.log1p(10 / 1), np.log1p(10 / 6)]
        np.testing.assert_allclose(out.iloc[:4].to_numpy(), expected)
        assert np.isnan(out.iloc[4])  # missing stays missing

    def test_all_missing_column_does_not_crash(self):
        train = pd.DataFrame({"c": pd.Series([np.nan] * 4, dtype=object)})
        enc = AutoIFrequencyEncoder().fit(train)
        out = enc.transform(pd.DataFrame({"c": pd.Series([np.nan, "x"], dtype=object)}))
        assert np.isnan(out["c"].iloc[0])
        assert np.isfinite(out["c"].iloc[1])

    def test_inverse_transform_preserves_index(self):
        df = pd.DataFrame({"c": ["a"] * 3 + ["b"]}, index=[10, 11, 12, 13])
        enc = AutoIFrequencyEncoder().fit(df)
        decoded = enc.inverse_transform(enc.transform(df))
        assert decoded.index.tolist() == [10, 11, 12, 13]
        assert decoded["c"].tolist() == ["a", "a", "a", "b"]

    def test_refit_discards_previous_columns(self):
        enc = AutoIFrequencyEncoder().fit(pd.DataFrame({"old": list("ab")}))
        enc.fit(pd.DataFrame({"new": list("cd")}))
        assert list(enc._freq_fit) == ["new"]


class TestLabelEncoderRobustness:
    def test_real_unknown_category_keeps_its_own_code(self):
        train = pd.DataFrame({"c": ["Unknown", "a", "b", "Unknown"]})
        enc = AutoLabelEncoder().fit(train)
        out = enc.transform(pd.DataFrame({"c": ["Unknown", "a", "never_seen"]}))["c"].tolist()

        known_unknown, a_code, unseen_code = out
        assert known_unknown != unseen_code, "real 'Unknown' collided with the reserved token"
        assert len({known_unknown, a_code, unseen_code}) == 3
        decoded = enc.inverse_transform(pd.DataFrame({"c": out}))["c"].tolist()
        assert decoded[:2] == ["Unknown", "a"]

    def test_reserved_token_is_unknown_by_default(self):
        enc = AutoLabelEncoder().fit(pd.DataFrame({"c": ["a", "b"]}))
        out = enc.transform(pd.DataFrame({"c": ["zzz"]}))
        assert enc.inverse_transform(out)["c"].tolist() == ["Unknown"]

    @pytest.mark.parametrize("dtype", [object, "category", "string"])
    def test_missing_values_form_their_own_class(self, dtype):
        train = pd.DataFrame({"c": pd.Series(["a", None, "b", "a"], dtype=dtype)})
        enc = AutoLabelEncoder().fit(train)
        codes = enc.transform(train)["c"].tolist()
        assert codes[1] not in (codes[0], codes[2])
        # ... and the same code is produced at transform time.
        again = enc.transform(pd.DataFrame({"c": pd.Series([None], dtype=object)}))["c"].iloc[0]
        assert again == codes[1]

    def test_output_is_integer(self):
        df = pd.DataFrame({"c": pd.Categorical(list("abca"))})
        assert AutoLabelEncoder().fit(df).transform(df)["c"].dtype.kind == "i"


class TestOneHotEncoderRobustness:
    @pytest.mark.parametrize("dtype", [object, "category", "string"])
    def test_supports_every_categorical_dtype(self, dtype):
        df = pd.DataFrame({"c": pd.Series(list("abca"), dtype=dtype)})
        out = AutoOneHotEncoder().fit(df).transform(df)
        assert sorted(out.columns) == ["c_a", "c_b", "c_c"]
        assert out.to_numpy().sum() == 4


# ============================================================================
# Imputers / scalers
# ============================================================================


class TestImputerRobustness:
    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_all_missing_column_is_kept_and_filled(self, imputer_cls):
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0], "empty": [np.nan] * 4, "b": [2.0, 3.0, np.nan, 5.0]}
        )
        out = imputer_cls().fit(df).transform(df)
        assert list(out.columns) == ["a", "empty", "b"]
        assert out.isnull().sum().sum() == 0
        assert (out["empty"] == 0.0).all()

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_no_numeric_columns_is_a_noop(self, imputer_cls):
        df = pd.DataFrame({"c": list("xyz")})
        out = imputer_cls().fit(df).transform(df)
        pd.testing.assert_frame_equal(out, df)

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_missing_fitted_column_raises_clear_error(self, imputer_cls):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, np.nan]})
        imputer = imputer_cls().fit(df)
        with pytest.raises(ValidationError, match="b"):
            imputer.transform(df[["a"]])

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_target_and_non_numeric_columns_untouched(self, imputer_cls):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 5.0],
                "label": ["x", None, "y", "x"],
                "target": [np.nan, 1.0, 2.0, 3.0],
            }
        )
        out = imputer_cls(target="target").fit(df).transform(df)
        assert out["a"].isnull().sum() == 0
        assert np.isnan(out["target"].iloc[0])
        assert out["label"].isnull().sum() == 1

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_nullable_integer_columns_are_imputed(self, imputer_cls):
        df = pd.DataFrame(
            {"i": pd.array([1, None, 3, 4], dtype="Int64"), "f": [1.0, 2.0, np.nan, 4.0]}
        )
        out = imputer_cls().fit(df).transform(df)
        assert out.isnull().sum().sum() == 0

    def test_constant_strategy_uses_fill_value(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        out = AutoSimpleImputer(strategy="constant", fill_value=-1).fit(df).transform(df)
        assert out["a"].tolist() == [1.0, -1.0, 3.0]

    def test_input_is_not_mutated(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        AutoSimpleImputer().fit(df).transform(df)
        assert df["a"].isnull().sum() == 1


class TestScalerRobustness:
    @pytest.mark.parametrize("scaler_cls", SCALERS)
    def test_missing_fitted_column_raises_clear_error(self, scaler_cls):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = scaler_cls().fit(df)
        with pytest.raises(ValidationError, match="b"):
            scaler.transform(df[["a"]])

    @pytest.mark.parametrize("scaler_cls", SCALERS)
    def test_roundtrip_and_index(self, scaler_cls):
        df = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [4.0, 5.0, 9.0]}, index=[7, 8, 9])
        scaler = scaler_cls().fit(df)
        scaled = scaler.transform(df)
        assert scaled.index.tolist() == [7, 8, 9]
        np.testing.assert_allclose(scaler.inverse_transform(scaled).to_numpy(), df.to_numpy())

    @pytest.mark.parametrize("scaler_cls", SCALERS)
    def test_each_instance_owns_its_estimator(self, scaler_cls):
        assert scaler_cls()._scaler is not scaler_cls()._scaler


# ============================================================================
# VIF
# ============================================================================


class TestVIFComputation:
    def test_matches_statsmodels_reference(self):
        sm_vif = pytest.importorskip("statsmodels.stats.outliers_influence")
        add_constant = pytest.importorskip("statsmodels.tools.tools").add_constant

        rng = np.random.RandomState(0)
        for _ in range(20):
            n, k = rng.randint(20, 300), rng.randint(2, 9)
            values = rng.randn(n, k) @ rng.randn(k, k) * rng.uniform(0.1, 50, k) + rng.uniform(
                -20, 20, k
            )
            design = add_constant(pd.DataFrame(values), has_constant="add").to_numpy()
            reference = np.array(
                [sm_vif.variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
            )
            np.testing.assert_allclose(compute_vif(values), reference, rtol=1e-8)

    def test_independent_columns_have_unit_vif(self):
        rng = np.random.RandomState(1)
        vif = compute_vif(rng.randn(5000, 4))
        np.testing.assert_allclose(vif, 1.0, atol=0.01)

    def test_exact_collinearity_is_infinite(self):
        rng = np.random.RandomState(2)
        a, c = rng.randn(60), rng.randn(60)
        vif = compute_vif(np.column_stack([a, 2 * a, c]))
        assert np.isinf(vif[0]) and np.isinf(vif[1])
        assert vif[2] == pytest.approx(1.0, abs=0.2)

    def test_constant_column_has_unit_vif(self):
        rng = np.random.RandomState(3)
        vif = compute_vif(np.column_stack([rng.randn(40), np.ones(40), rng.randn(40)]))
        assert vif[1] == 1.0
        assert np.isfinite(vif).all()

    @pytest.mark.parametrize("shape", [(1, 3), (5, 0), (5, 1)])
    def test_degenerate_shapes(self, shape):
        assert compute_vif(np.zeros(shape)).shape == (shape[1],)


class TestVIFSelectorRobustness:
    def test_non_numeric_features_pass_through(self):
        rng = np.random.RandomState(0)
        n = 100
        df = pd.DataFrame(
            {
                "a": rng.randn(n),
                "label": list("ab") * (n // 2),
                "b": rng.randn(n),
                "target": rng.randn(n),
            }
        )
        selector = VIFFeatureSelector(target="target").fit(df)
        assert selector.selected_features == ["a", "label", "b"]
        assert list(selector.transform(df).columns) == ["a", "label", "b", "target"]

    def test_target_is_never_a_selected_feature(self):
        df = pd.DataFrame({"cat": ["A", "B", "C"], "target": [1.0, 2.0, 3.0]})
        selector = VIFFeatureSelector(target="target").fit(df)
        assert selector.selected_features == ["cat"]
        assert list(selector.transform(df).columns) == ["cat", "target"]
        assert selector.n_removed == 0
        assert selector.vif_dataframe is None

    def test_boolean_and_nullable_columns_take_part(self):
        rng = np.random.RandomState(0)
        n = 200
        a = rng.randn(n)
        df = pd.DataFrame(
            {
                "a": a,
                "flag": rng.rand(n) > 0.5,
                "copy": pd.array((a * 1000).astype(int), dtype="Int64"),
                "target": rng.randn(n),
            }
        )
        selector = VIFFeatureSelector(target="target", vif_threshold=5.0).fit(df)
        assert "flag" in selector.selected_features
        assert selector.n_removed == 1
        assert selector.removed_features[0] in ("a", "copy")

    def test_removed_features_are_reported_in_order(self):
        rng = np.random.RandomState(4)
        n = 300
        a, b = rng.randn(n), rng.randn(n)
        df = pd.DataFrame(
            {"a": a, "a2": a * 2, "b": b, "b2": b + rng.randn(n) * 1e-3, "target": rng.randn(n)}
        )
        selector = VIFFeatureSelector(target="target").fit(df)
        assert selector.n_removed == 2 == len(selector.removed_features)
        assert len(selector.selected_features) == 2
        assert set(selector.vif_dataframe["variables"]) == set(selector.selected_features)
        assert (selector.vif_dataframe["VIF"] < 10).all()

    def test_infinite_values_rejected(self):
        df = pd.DataFrame(
            {"a": [1.0, np.inf, 3.0], "b": [3.0, 1.0, 2.0], "target": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(ValidationError, match="Infinite"):
            VIFFeatureSelector(target="target").fit(df)

    def test_nulls_still_rejected(self):
        df = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64"), "target": [1.0, 2.0, 3.0]})
        with pytest.raises(ValidationError, match="Null"):
            VIFFeatureSelector(target="target").fit(df)


# ============================================================================
# Metrics
# ============================================================================


class TestBinaryMetricLabels:
    def test_string_labels_no_longer_crash(self):
        y_true = np.array(["no", "yes", "no", "yes"])
        y_pred = np.array(["no", "yes", "yes", "yes"])
        m = metrics_classification(y_true, y_pred)
        # positive class = last label in sorted order ("yes")
        assert m["Precision"].iloc[0] == pytest.approx(2 / 3)
        assert m["Recall"].iloc[0] == pytest.approx(1.0)
        assert m["Accuracy"].iloc[0] == pytest.approx(0.75)

    def test_zero_one_labels_unchanged(self):
        y_true = np.array([1, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 0, 1])
        m = metrics_classification(y_true, y_pred, n_classes=2)
        assert m["Precision"].iloc[0] == pytest.approx(1.0)
        assert m["Recall"].iloc[0] == pytest.approx(0.75)

    def test_labels_containing_one_keep_sklearn_default_positive(self):
        m = metrics_classification(np.array([1, 2, 1, 2]), np.array([1, 2, 2, 2]))
        assert m["Precision"].iloc[0] == pytest.approx(1.0)
        assert m["Recall"].iloc[0] == pytest.approx(0.5)

    def test_integer_codes_without_one(self):
        m = metrics_classification(np.array([0, 2, 0, 2]), np.array([0, 2, 2, 2]))
        assert m["Precision"].iloc[0] == pytest.approx(2 / 3)

    def test_lists_are_accepted(self):
        m = metrics_classification(["a", "b", "a"], ["a", "a", "a"])
        assert m["Accuracy"].iloc[0] == pytest.approx(2 / 3)

    def test_multiclass_uses_weighted_average(self):
        m = metrics_classification(np.array(["a", "b", "c", "c"]), np.array(["a", "b", "c", "b"]))
        assert m["Recall"].iloc[0] == pytest.approx(0.75)


# ============================================================================
# Optuna adapter
# ============================================================================


class TestOptunaLoggingIsScoped:
    def test_global_logger_is_not_disabled(self):
        logger = logging.getLogger("optuna")
        was_disabled = logger.disabled
        adapter = OptunaAdapter(random_state=1, verbosity=0)
        adapter.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2, show_progress=False)
        assert logger.disabled is was_disabled

    @pytest.mark.parametrize("level", [optuna.logging.INFO, optuna.logging.WARNING])
    def test_verbosity_is_restored_after_use(self, level):
        previous = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(level)
        try:
            adapter = OptunaAdapter(random_state=1, verbosity=0)
            adapter.create_study()
            adapter.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2, show_progress=False)
            assert optuna.logging.get_verbosity() == level
        finally:
            optuna.logging.set_verbosity(previous)

    def test_verbosity_restored_when_objective_fails(self):
        previous = optuna.logging.get_verbosity()
        adapter = OptunaAdapter(random_state=1, verbosity=2)

        def boom(trial):
            raise RuntimeError("objective failed")

        with pytest.raises(RuntimeError):
            adapter.optimize(boom, n_trials=1, show_progress=False)
        assert optuna.logging.get_verbosity() == previous

    def test_progress_bar_path(self):
        adapter = OptunaAdapter(random_state=1, verbosity=0)
        adapter.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2, show_progress=True)
        assert adapter.best_value is not None


# ============================================================================
# Exceptions
# ============================================================================


class TestExceptionHierarchy:
    def test_validation_and_not_fitted_are_value_errors(self):
        assert issubclass(ValidationError, ValueError)
        assert issubclass(NotFittedError, ValueError)

    def test_h2o_connection_error_is_a_feature_selection_error(self):
        assert issubclass(H2OConnectionError, FeatureSelectionError)

    def test_schema_error_is_a_validation_error(self):
        assert issubclass(SchemaValidationError, ValidationError)
        assert issubclass(SchemaValidationError, AtlanticError)

    def test_not_fitted_custom_message(self):
        err = NotFittedError("Thing", message="custom text")
        assert str(err) == "custom text"
        assert err.component == "Thing"


# ============================================================================
# H2O selector (stubbed - no JVM required)
# ============================================================================


@pytest.fixture
def h2o_train():
    rng = np.random.RandomState(0)
    n = 40
    return pd.DataFrame(
        {
            "num_a": rng.randn(n),
            "num_b": rng.randn(n),
            "city": rng.choice(["lisbon", "porto"], n),
            "noise": rng.randn(n),
            "target": rng.randn(n),
        }
    )


class TestH2OSelector:
    def test_missing_dependency_gives_actionable_error(self, h2o_train):
        selector = H2OFeatureSelector(target="target")
        with patch.dict(sys.modules, {"h2o": None}):  # makes ``import h2o`` fail
            with pytest.raises(H2OConnectionError) as excinfo:
                selector.fit(h2o_train)
        message = str(excinfo.value)
        assert message == H2O_INSTALL_HINT
        assert "pip install" in message and "relevance=1.0" in message
        assert isinstance(excinfo.value, FeatureSelectionError)
        assert not selector.is_fitted

    def test_cluster_start_failure_is_a_connection_error(self, h2o_train):
        stub = make_h2o_stub({}, init_error=RuntimeError("no java"))
        with patch.dict(sys.modules, stub):
            with pytest.raises(H2OConnectionError, match="Java"):
                H2OFeatureSelector(target="target").fit(h2o_train)

    def test_is_h2o_available_reflects_importability(self):
        with patch.dict(sys.modules, make_h2o_stub({})):
            assert is_h2o_available() is True
        with patch.dict(sys.modules, {"h2o": None}):  # import explicitly blocked
            assert is_h2o_available() is False

    @pytest.mark.parametrize("kwargs", [{"relevance": 0.1}, {"relevance": 1.5}, {"max_models": 0}])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValidationError) as excinfo:
            H2OFeatureSelector(target="t", **kwargs)
        assert isinstance(excinfo.value, ValueError)  # backwards compatible
        assert "between" in str(excinfo.value)

    def test_fit_uses_first_model_with_variableimportance_frame(self, h2o_train):
        importances = {
            "stacked": None,  # leader without varimp must be skipped
            "gbm": importance_frame(
                [("num_a", 9.0, 1.0, 0.60), ("city", 5.0, 0.5, 0.39), ("num_b", 0.1, 0.01, 0.01)]
            ),
        }
        stub = make_h2o_stub(importances)
        with patch.dict(sys.modules, stub):
            selector = H2OFeatureSelector(target="target", relevance=0.9, max_models=2, seed=7)
            selector.fit(h2o_train)

        assert selector.selected_features == ["num_a", "city"]
        assert selector.feature_importance["variable"].tolist() == ["num_a", "city"]
        assert selector.is_fitted
        h2o = stub["h2o"]
        assert h2o.automl_kwargs["seed"] == 7 and h2o.automl_kwargs["max_models"] == 2
        assert h2o.trained_on == (["num_a", "num_b", "city", "noise"], "target")
        # every key created during fit is released, nothing else
        assert h2o.removed == ["frame_key", "stacked", "gbm"]
        # categoricals were label-encoded before reaching H2O
        assert h2o.frames[0]["city"].dtype.kind == "i"

    def test_no_model_with_importance_raises(self, h2o_train):
        stub = make_h2o_stub({"stacked": None})
        with patch.dict(sys.modules, stub):
            with pytest.raises(FeatureSelectionError, match="variable importance"):
                H2OFeatureSelector(target="target").fit(h2o_train)
        assert stub["h2o"].removed == ["frame_key", "stacked"]

    def test_expanded_categorical_levels_fold_back_to_their_column(self, h2o_train):
        importances = {
            "xgb": importance_frame(
                [
                    ("city.lisbon", 4.0, 0.8, 0.30),
                    ("num_a", 5.0, 1.0, 0.40),
                    ("city.porto", 3.0, 0.6, 0.25),
                    ("not_a_column", 0.5, 0.1, 0.05),
                ]
            )
        }
        with patch.dict(sys.modules, make_h2o_stub(importances)):
            selector = H2OFeatureSelector(
                target="target", relevance=0.9, encoding_before_selection=False
            ).fit(h2o_train)

        assert selector.selected_features == ["city", "num_a"]
        folded = selector.feature_importance.set_index("variable")["percentage"]
        assert folded["city"] == pytest.approx(0.55)

    def test_empty_selection_keeps_all_features(self, h2o_train):
        importances = {"gbm": importance_frame([("num_a", 0.0, 0.0, np.nan)])}
        with patch.dict(sys.modules, make_h2o_stub(importances)):
            with pytest.warns(UserWarning, match="keeping all features"):
                selector = H2OFeatureSelector(target="target").fit(h2o_train)
        assert selector.selected_features == ["num_a", "num_b", "city", "noise"]

    def test_unexpected_failure_is_wrapped_and_chained(self, h2o_train):
        stub = make_h2o_stub({"gbm": importance_frame([("num_a", 1.0, 1.0, 1.0)])})
        stub["h2o"].get_model = lambda model_id: (_ for _ in ()).throw(KeyError("boom"))
        with patch.dict(sys.modules, stub):
            with pytest.raises(FeatureSelectionError, match="H2O feature selection failed") as exc:
                H2OFeatureSelector(target="target").fit(h2o_train)
        assert isinstance(exc.value.__cause__, KeyError)

    def test_classification_target_is_passed_as_object(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"x": rng.randn(20), "target": rng.choice(["a", "b"], 20)})
        stub = make_h2o_stub({"gbm": importance_frame([("x", 1.0, 1.0, 1.0)])})
        with patch.dict(sys.modules, stub):
            selector = H2OFeatureSelector(target="target").fit(df)
        assert selector.pred_type == "Class"
        assert stub["h2o"].frames[0]["target"].dtype == object


# ============================================================================
# Dataset generator
# ============================================================================


class TestDatasetGeneratorIsolation:
    def test_global_numpy_random_state_is_untouched(self):
        np.random.seed(123)
        expected = np.random.rand(3)

        np.random.seed(123)
        DatasetGenerator.generate_classification(n_samples=60, random_state=1)
        DatasetGenerator.generate_regression(n_samples=60, random_state=2)
        DatasetGenerator.generate_with_datetime(n_samples=60, random_state=3)
        DatasetGenerator.generate_high_cardinality(n_samples=60, random_state=4)
        np.testing.assert_array_equal(np.random.rand(3), expected)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda seed: DatasetGenerator.generate_classification(n_samples=80, random_state=seed),
            lambda seed: DatasetGenerator.generate_regression(n_samples=80, random_state=seed),
            lambda seed: DatasetGenerator.generate_with_datetime(n_samples=80, random_state=seed),
            lambda seed: DatasetGenerator.generate_high_cardinality(
                n_samples=80, random_state=seed
            ),
        ],
    )
    def test_reproducible_and_seed_sensitive(self, factory):
        first, _ = factory(5)
        second, _ = factory(5)
        other, _ = factory(6)
        pd.testing.assert_frame_equal(first, second)
        assert not first.equals(other)

    def test_introduce_nulls_defaults_to_global_generator(self):
        df = pd.DataFrame({"a": np.arange(20.0), "target": np.arange(20.0)})
        out = DatasetGenerator._introduce_nulls(df, "target", 0.25)
        assert out["a"].isnull().sum() == 5
        assert out["target"].isnull().sum() == 0
