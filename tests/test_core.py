"""Tests for atlantic.core module: enums, exceptions, schemas, mixins, base."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.enums import (
    ClassificationMetric,
    DimensionalityTier,
    EncoderType,
    EncodingVersionType,
    FeatureSelectorType,
    ImputerType,
    OptimizationLevel,
    RegressionMetric,
    ScalerType,
    SimpleImputerStrategy,
    TaskType,
)
from atlantic.core.exceptions import (
    AtlanticError,
    ConfigurationError,
    EncodingError,
    FeatureSelectionError,
    H2OConnectionError,
    ImputationError,
    NotFittedError,
    RegistryError,
    SchemaValidationError,
    StateSerializationError,
    ValidationError,
)
from atlantic.core.mixins import (
    ColumnTypeMixin,
    DataValidationMixin,
    DateEngineeringMixin,
    SchemaValidationMixin,
    TargetTypeMixin,
)
from atlantic.core.schemas import (
    BuilderConfig,
    EncodingConfig,
    EncodingSelectionResult,
    EvaluationResult,
    FeatureImportanceResult,
    FeatureSelectionConfig,
    FittedComponents,
    ImputationConfig,
    ImputationSelectionResult,
    OptimizerConfig,
    PipelineConfig,
    PipelineState,
)


# ============================================================================
# Enums
# ============================================================================


class TestEnums:
    def test_task_type_values(self):
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.CLASSIFICATION.value == "classification"

    def test_dimensionality_tier_values(self):
        assert DimensionalityTier.SMALL.value == "small"
        assert DimensionalityTier.XLARGE.value == "xlarge"

    def test_optimization_level_values(self):
        assert OptimizationLevel.FAST.value == "fast"
        assert OptimizationLevel.BALANCED.value == "balanced"
        assert OptimizationLevel.THOROUGH.value == "thorough"

    def test_encoding_version_type_values(self):
        # Renamed from EncodingVersion to avoid shadow-collision with the
        # helper class atlantic.encoding.versions.EncodingVersion.
        assert EncodingVersionType.V1.value == "v1"
        assert EncodingVersionType.V4.value == "v4"

    def test_encoding_version_type_is_distinct_from_helper(self):
        # Regression test for the shadow-import bug: the package-level
        # ``atlantic.EncodingVersion`` resolves to the helper class, while
        # ``EncodingVersionType`` resolves to the enum. They are distinct.
        import atlantic

        assert atlantic.EncodingVersion is not atlantic.EncodingVersionType
        assert hasattr(atlantic.EncodingVersionType, "V1")
        # Helper class is callable with (train, test, target)
        assert callable(atlantic.EncodingVersion)

    def test_scaler_type_values(self):
        assert ScalerType.STANDARD.value == "standard"
        assert ScalerType.MINMAX.value == "minmax"
        assert ScalerType.ROBUST.value == "robust"

    def test_encoder_type_values(self):
        assert EncoderType.LABEL.value == "label"
        assert EncoderType.IFREQUENCY.value == "ifrequency"
        assert EncoderType.ONEHOT.value == "onehot"

    def test_imputer_type_values(self):
        assert ImputerType.SIMPLE.value == "simple"
        assert ImputerType.KNN.value == "knn"
        assert ImputerType.ITERATIVE.value == "iterative"

    def test_feature_selector_type_values(self):
        assert FeatureSelectorType.H2O.value == "h2o"
        assert FeatureSelectorType.VIF.value == "vif"

    def test_simple_imputer_strategy(self):
        assert SimpleImputerStrategy.MEAN.value == "mean"
        assert SimpleImputerStrategy.MEDIAN.value == "median"
        assert SimpleImputerStrategy.MOST_FREQUENT.value == "most_frequent"
        assert SimpleImputerStrategy.CONSTANT.value == "constant"

    def test_regression_metric(self):
        assert RegressionMetric.MAE.value == "mae"
        assert RegressionMetric.R2.value == "r2"

    def test_classification_metric(self):
        assert ClassificationMetric.ACCURACY.value == "accuracy"
        assert ClassificationMetric.F1.value == "f1"


# ============================================================================
# Exceptions
# ============================================================================


class TestExceptions:
    def test_atlantic_error_is_base(self):
        assert issubclass(AtlanticError, Exception)

    def test_all_exceptions_inherit_from_atlantic_error(self):
        for exc_class in [
            NotFittedError,
            ValidationError,
            FeatureSelectionError,
            EncodingError,
            ImputationError,
            RegistryError,
            ConfigurationError,
            H2OConnectionError,
            SchemaValidationError,
            StateSerializationError,
        ]:
            assert issubclass(exc_class, AtlanticError)

    def test_not_fitted_error_message(self):
        err = NotFittedError("MyEncoder")
        assert "MyEncoder" in str(err)
        assert "fit" in str(err).lower()

    def test_not_fitted_error_component_attr(self):
        err = NotFittedError("Scaler")
        assert err.component == "Scaler"

    def test_validation_error_raises(self):
        with pytest.raises(ValidationError, match="bad input"):
            raise ValidationError("bad input")


# ============================================================================
# Schemas – PipelineConfig
# ============================================================================


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.split_ratio == 0.75
        assert cfg.relevance == 0.99
        assert cfg.h2o_fs_models == 7
        assert cfg.encoding_fs is True
        assert cfg.vif_threshold == 10.0
        assert cfg.optimization_level == "balanced"

    def test_valid_custom(self):
        cfg = PipelineConfig(split_ratio=0.8, relevance=0.95, optimization_level="fast")
        assert cfg.split_ratio == 0.8
        assert cfg.optimization_level == "fast"

    def test_invalid_split_ratio_low(self):
        with pytest.raises(Exception):
            PipelineConfig(split_ratio=0.1)

    def test_invalid_split_ratio_high(self):
        with pytest.raises(Exception):
            PipelineConfig(split_ratio=0.99)

    def test_invalid_relevance(self):
        with pytest.raises(Exception):
            PipelineConfig(relevance=0.1)

    def test_model_dump(self):
        cfg = PipelineConfig()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "split_ratio" in d


class TestFeatureSelectionConfig:
    def test_defaults(self):
        cfg = FeatureSelectionConfig()
        assert cfg.method == "both"
        assert cfg.h2o_relevance == 0.99

    def test_excluded_algorithms_default(self):
        cfg = FeatureSelectionConfig()
        assert "GLM" in cfg.h2o_excluded_algorithms


class TestEncodingConfig:
    def test_defaults(self):
        cfg = EncodingConfig()
        assert cfg.scaler == "standard"
        assert cfg.encoder == "ifrequency"
        assert cfg.auto_select is True


class TestImputationConfig:
    def test_defaults(self):
        cfg = ImputationConfig()
        assert cfg.method == "simple"
        assert cfg.auto_select is True


class TestOptimizerConfig:
    def test_defaults(self):
        cfg = OptimizerConfig()
        assert cfg.optimization_level == "balanced"
        assert cfg.n_folds == 3


class TestBuilderConfig:
    def test_defaults(self):
        cfg = BuilderConfig()
        assert cfg.date_engineering is True
        assert isinstance(cfg.feature_selection, FeatureSelectionConfig)
        assert isinstance(cfg.encoding, EncodingConfig)
        assert isinstance(cfg.imputation, ImputationConfig)
        assert isinstance(cfg.optimizer, OptimizerConfig)


# ============================================================================
# Schemas – Dataclasses
# ============================================================================


class TestFittedComponents:
    def test_defaults(self):
        fc = FittedComponents()
        assert fc.encoder is None
        assert fc.scaler is None
        assert fc.selected_columns == []

    def test_to_dict(self):
        fc = FittedComponents(selected_columns=["a", "b"], encoding_version="v1")
        d = fc.to_dict()
        assert d["selected_columns"] == ["a", "b"]
        assert d["encoding_version"] == "v1"


class TestPipelineState:
    def test_defaults(self):
        ps = PipelineState()
        assert ps.is_fitted is False
        assert ps.task_type == "regression"

    def test_serialization_roundtrip(self, tmp_path):
        ps = PipelineState(
            config=PipelineConfig(),
            task_type="classification",
            n_classes=3,
            is_fitted=True,
            target="y",
        )
        path = tmp_path / "state.json"
        ps.save(str(path))
        loaded = PipelineState.load(str(path))
        assert loaded.task_type == "classification"
        assert loaded.n_classes == 3
        assert loaded.is_fitted is True
        assert loaded.target == "y"

    def test_to_dict_from_dict_roundtrip(self):
        ps = PipelineState(
            config=PipelineConfig(split_ratio=0.8),
            target="y",
            is_fitted=True,
        )
        d = ps.to_dict()
        ps2 = PipelineState.from_dict(d)
        assert ps2.config.split_ratio == 0.8
        assert ps2.target == "y"


class TestEvaluationResult:
    def test_to_dict(self):
        er = EvaluationResult(
            metrics=pd.DataFrame({"mae": [0.5]}),
            best_params={"n_estimators": 100},
            best_model="rf",
            iteration=1,
        )
        d = er.to_dict()
        assert d["best_model"] == "rf"
        assert d["iteration"] == 1


class TestFeatureImportanceResult:
    def test_n_selected(self):
        fir = FeatureImportanceResult(
            importance_df=pd.DataFrame(),
            selected_features=["a", "b", "c"],
            threshold_used=0.99,
        )
        assert fir.n_selected == 3


class TestEncodingSelectionResult:
    def test_fields(self):
        esr = EncodingSelectionResult(
            selected_version="v1",
            performance_scores={"v1": 0.9, "v2": 0.8},
            best_score=0.9,
            metric_name="mae",
        )
        assert esr.selected_version == "v1"
        assert esr.best_score == 0.9


class TestImputationSelectionResult:
    def test_fields(self):
        isr = ImputationSelectionResult(
            selected_method="knn",
            performance_scores={"simple": 0.8, "knn": 0.9},
            best_score=0.9,
            metric_name="mae",
        )
        assert isr.selected_method == "knn"


# ============================================================================
# Mixins
# ============================================================================


class ConcreteValidation(DataValidationMixin):
    def __init__(self, target="target"):
        self.target = target


class ConcreteColumnType(ColumnTypeMixin):
    def __init__(self, target="target"):
        self.target = target


class ConcreteDateEng(DateEngineeringMixin):
    pass


class ConcreteSchema(SchemaValidationMixin):
    pass


class ConcreteTargetType(TargetTypeMixin):
    def __init__(self, target="target"):
        self.target = target


class TestDataValidationMixin:
    def test_validate_dataframe_ok(self, sample_regression_df):
        v = ConcreteValidation()
        v._validate_dataframe(sample_regression_df)  # should not raise

    def test_validate_dataframe_not_df(self):
        v = ConcreteValidation()
        with pytest.raises(ValidationError, match="pandas DataFrame"):
            v._validate_dataframe("not a dataframe")

    def test_validate_dataframe_empty(self):
        v = ConcreteValidation()
        with pytest.raises(ValidationError, match="empty"):
            v._validate_dataframe(pd.DataFrame())

    def test_validate_dataframe_missing_target(self):
        v = ConcreteValidation(target="missing_col")
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValidationError, match="missing_col"):
            v._validate_dataframe(df)

    def test_validate_split_ratio_ok(self):
        v = ConcreteValidation()
        v._validate_split_ratio(0.75)  # no error

    def test_validate_split_ratio_bad(self):
        v = ConcreteValidation()
        with pytest.raises(ValidationError):
            v._validate_split_ratio(0.1)

    def test_validate_columns_exist(self):
        v = ConcreteValidation()
        df = pd.DataFrame({"a": [1], "b": [2]})
        v._validate_columns_exist(df, ["a", "b"])  # ok
        with pytest.raises(ValidationError):
            v._validate_columns_exist(df, ["a", "z"])


class TestColumnTypeMixin:
    def test_get_numeric_columns(self, sample_regression_df):
        m = ConcreteColumnType()
        nums = m._get_numeric_columns(sample_regression_df)
        assert "num_a" in nums
        assert "num_b" in nums
        assert "target" not in nums
        assert "cat_x" not in nums

    def test_get_categorical_columns(self, sample_regression_df):
        m = ConcreteColumnType()
        cats = m._get_categorical_columns(sample_regression_df)
        assert "cat_x" in cats
        assert "cat_y" in cats
        assert "num_a" not in cats

    def test_get_datetime_columns(self, datetime_df):
        m = ConcreteColumnType()
        dts = m._get_datetime_columns(datetime_df)
        assert "event_date" in dts

    def test_get_all_feature_columns(self, sample_regression_df):
        m = ConcreteColumnType()
        feats = m._get_all_feature_columns(sample_regression_df)
        assert "target" not in feats
        assert "num_a" in feats
        assert "cat_x" in feats


class TestDateEngineeringMixin:
    def test_engineer_features_drop(self, datetime_df):
        m = ConcreteDateEng()
        result = m._engineer_datetime_features(datetime_df, drop=True)
        assert "event_date" not in result.columns
        assert "event_date_year" in result.columns
        assert "event_date_month" in result.columns
        assert "event_date_day_of_week" in result.columns
        assert "event_date_is_wknd" in result.columns

    def test_engineer_features_keep(self, datetime_df):
        m = ConcreteDateEng()
        result = m._engineer_datetime_features(datetime_df, drop=False)
        assert "event_date" in result.columns
        assert "event_date_year" in result.columns

    def test_no_datetime_cols(self, numeric_df):
        m = ConcreteDateEng()
        result = m._engineer_datetime_features(numeric_df)
        assert list(result.columns) == list(numeric_df.columns)


class TestSchemaValidationMixin:
    def test_store_and_validate(self, sample_regression_df):
        m = ConcreteSchema()
        m._store_input_schema(sample_regression_df)
        m._validate_input_schema(sample_regression_df)  # should not raise

    def test_validate_missing_col(self, sample_regression_df):
        m = ConcreteSchema()
        m._store_input_schema(sample_regression_df)
        subset = sample_regression_df.drop(columns=["num_a"])
        with pytest.raises(ValidationError, match="Missing"):
            m._validate_input_schema(subset)

    def test_validate_no_schema_stored(self, sample_regression_df):
        m = ConcreteSchema()
        with pytest.raises(ValidationError, match="No input schema"):
            m._validate_input_schema(sample_regression_df)


class TestTargetTypeMixin:
    def test_regression_detection(self, sample_regression_df):
        m = ConcreteTargetType()
        pred_type, metric = m._detect_target_type(sample_regression_df)
        assert pred_type == "Reg"
        assert metric == "Mean Absolute Error"

    def test_binary_classification_detection(self, sample_classification_df):
        m = ConcreteTargetType()
        pred_type, metric = m._detect_target_type(sample_classification_df)
        assert pred_type == "Class"
        assert metric == "Precision"
        assert m.n_classes == 2

    def test_multiclass_detection(self, sample_multiclass_df):
        m = ConcreteTargetType()
        pred_type, metric = m._detect_target_type(sample_multiclass_df)
        assert pred_type == "Class"
        assert metric == "F1"
        assert m.n_classes == 4

    def test_missing_target(self, numeric_df):
        m = ConcreteTargetType(target="nonexistent")
        with pytest.raises(ValidationError):
            m._detect_target_type(numeric_df)

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64",
                                        "float16", "float32", "float64"])
    def test_numeric_target_dtypes_classified_as_regression(self, dtype):
        """Regression test: previously the dtype-vs-string-list ``in``
        check only matched int64/float64 by coincidence and would
        misclassify int32/float32 etc. as classification."""
        df = pd.DataFrame({"target": np.array([1, 2, 3, 4, 5], dtype=dtype)})
        m = ConcreteTargetType()
        pred_type, metric = m._detect_target_type(df)
        assert pred_type == "Reg", f"dtype {dtype} should be regression"
        assert metric == "Mean Absolute Error"

    def test_pandas_nullable_int_target_classified_as_regression(self):
        """Pandas extension dtype ``Int64`` should also be detected as numeric."""
        df = pd.DataFrame({"target": pd.array([1, 2, 3, None, 5], dtype="Int64")})
        m = ConcreteTargetType()
        pred_type, _ = m._detect_target_type(df)
        assert pred_type == "Reg"
