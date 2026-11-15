"""Tests for atlantic.optimization module: dimensionality, optuna_adapter, evaluation."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.enums import DimensionalityTier, OptimizationLevel
from atlantic.optimization.dimensionality import (
    DimensionalityConfig,
    DimensionalityStrategy,
    ModelHyperparameterRanges,
)
from atlantic.optimization.optuna_adapter import OptunaAdapter


# ============================================================================
# DimensionalityStrategy
# ============================================================================


class TestDimensionalityStrategy:
    def test_detect_tier_small(self):
        tier = DimensionalityStrategy.detect_tier(1000, 10)
        assert tier == DimensionalityTier.SMALL

    def test_detect_tier_medium(self):
        tier = DimensionalityStrategy.detect_tier(10000, 50)
        assert tier == DimensionalityTier.MEDIUM

    def test_detect_tier_large(self):
        tier = DimensionalityStrategy.detect_tier(50000, 200)
        assert tier == DimensionalityTier.LARGE

    def test_detect_tier_xlarge(self):
        tier = DimensionalityStrategy.detect_tier(200000, 600)
        assert tier == DimensionalityTier.XLARGE

    def test_get_config_small_balanced(self):
        cfg = DimensionalityStrategy.get_config(1000, 10, OptimizationLevel.BALANCED)
        assert cfg.tier == DimensionalityTier.SMALL
        assert cfg.n_trials == 10
        assert cfg.n_folds == 5
        assert cfg.model_complexity == "high"

    def test_get_config_large_fast(self):
        cfg = DimensionalityStrategy.get_config(50000, 200, OptimizationLevel.FAST)
        assert cfg.tier == DimensionalityTier.LARGE
        assert cfg.n_trials == 3
        assert cfg.model_complexity == "medium"

    def test_get_config_from_string(self):
        cfg = DimensionalityStrategy.get_config_from_string(1000, 10, "thorough")
        assert cfg.optimization_level == OptimizationLevel.THOROUGH
        assert cfg.n_trials == 20

    def test_get_config_from_string_default(self):
        cfg = DimensionalityStrategy.get_config_from_string(1000, 10, "unknown_level")
        assert cfg.optimization_level == OptimizationLevel.BALANCED

    def test_describe_tier(self):
        desc = DimensionalityStrategy.describe_tier(DimensionalityTier.SMALL)
        assert "Small" in desc
        assert "5K" in desc

    def test_early_stopping_config(self):
        cfg_small = DimensionalityStrategy.get_config(100, 5, OptimizationLevel.BALANCED)
        assert cfg_small.early_stopping_rounds is None
        cfg_medium = DimensionalityStrategy.get_config(10000, 50, OptimizationLevel.BALANCED)
        assert cfg_medium.early_stopping_rounds == 5


class TestDimensionalityConfig:
    def test_repr(self):
        cfg = DimensionalityConfig(
            tier=DimensionalityTier.SMALL,
            optimization_level=OptimizationLevel.BALANCED,
            n_trials=10,
            n_folds=5,
            early_stopping_rounds=None,
            model_complexity="high",
        )
        r = repr(cfg)
        assert "small" in r
        assert "balanced" in r
        assert "10" in r


class TestModelHyperparameterRanges:
    def test_low_complexity(self):
        ranges = ModelHyperparameterRanges.for_complexity("low")
        assert ranges.rf_n_estimators[1] <= 100
        assert ranges.xgb_learning_rate[0] >= 0.05

    def test_medium_complexity(self):
        ranges = ModelHyperparameterRanges.for_complexity("medium")
        assert ranges.rf_n_estimators[1] <= 150

    def test_high_complexity(self):
        ranges = ModelHyperparameterRanges.for_complexity("high")
        assert ranges.rf_n_estimators[1] <= 200
        assert ranges.xgb_learning_rate[0] <= 0.01


# ============================================================================
# OptunaAdapter
# ============================================================================


class TestOptunaAdapter:
    def test_create_study(self):
        adapter = OptunaAdapter(random_state=42, verbosity=0)
        study = adapter.create_study(direction="minimize", study_name="test")
        assert study is not None
        assert adapter.study is not None

    def test_optimize(self):
        adapter = OptunaAdapter(random_state=42, verbosity=0)
        adapter.create_study(direction="minimize")

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        adapter.optimize(objective, n_trials=5, show_progress=False)
        assert adapter.best_params is not None
        assert adapter.best_value is not None
        assert adapter.best_value >= 0

    def test_get_best_trial(self):
        adapter = OptunaAdapter(random_state=42, verbosity=0)
        assert adapter.get_best_trial() is None
        adapter.create_study(direction="minimize")

        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        adapter.optimize(objective, n_trials=3, show_progress=False)
        assert adapter.get_best_trial() is not None

    def test_get_trials_dataframe(self):
        adapter = OptunaAdapter(random_state=42, verbosity=0)
        assert adapter.get_trials_dataframe() is None
        adapter.create_study()

        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        adapter.optimize(objective, n_trials=3, show_progress=False)
        df = adapter.get_trials_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_auto_create_study(self):
        adapter = OptunaAdapter(random_state=42, verbosity=0)

        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        adapter.optimize(objective, n_trials=2, show_progress=False)
        assert adapter.study is not None

    def test_suggest_float_log_uses_modern_api(self, recwarn):
        """Regression test: the adapter must not emit Optuna deprecation
        warnings about ``suggest_loguniform``. The shim now forwards to
        ``trial.suggest_float(..., log=True)``.
        """
        import warnings

        adapter = OptunaAdapter(random_state=42, verbosity=0)
        adapter.create_study(direction="minimize")

        def objective(trial):
            # Exercise the log branch through the adapter shim.
            x = adapter.suggest_float(trial, "x", 1e-3, 1.0, log=True)
            return x

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            warnings.simplefilter("error", DeprecationWarning)
            adapter.optimize(objective, n_trials=3, show_progress=False)

        assert adapter.best_value is not None
        # No suggest_loguniform deprecation warning should have been raised.
        assert not any(
            "suggest_loguniform" in str(w.message) for w in recwarn.list
        )


# ============================================================================
# ModelEvaluator (lightweight test — full optimization is slow)
# ============================================================================


class TestModelEvaluator:
    def test_auto_evaluate_regression(self):
        from atlantic.optimization.evaluation import ModelEvaluator

        np.random.seed(42)
        n = 100
        train = pd.DataFrame(
            {
                "a": np.random.randn(n),
                "b": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )
        test = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
                "target": np.random.randn(30),
            }
        )
        evaluator = ModelEvaluator(
            train=train,
            test=test,
            target="target",
            optimization_level="fast",
            verbosity=0,
        )
        metrics = evaluator.auto_evaluate()
        assert isinstance(metrics, pd.DataFrame)
        assert "Mean Absolute Error" in metrics.columns

    def test_auto_evaluate_classification(self):
        from atlantic.optimization.evaluation import ModelEvaluator

        np.random.seed(42)
        n = 100
        train = pd.DataFrame(
            {
                "a": np.random.randn(n),
                "b": np.random.randn(n),
                "target": np.random.choice(["c0", "c1"], n),
            }
        )
        test = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
                "target": np.random.choice(["c0", "c1"], 30),
            }
        )
        evaluator = ModelEvaluator(
            train=train,
            test=test,
            target="target",
            optimization_level="fast",
            verbosity=0,
        )
        metrics = evaluator.auto_evaluate()
        assert isinstance(metrics, pd.DataFrame)
        assert "Precision" in metrics.columns

    def test_dimensionality_config_property(self):
        from atlantic.optimization.evaluation import ModelEvaluator

        np.random.seed(42)
        train = pd.DataFrame({"a": np.random.randn(50), "target": np.random.randn(50)})
        test = pd.DataFrame({"a": np.random.randn(10), "target": np.random.randn(10)})
        evaluator = ModelEvaluator(
            train=train, test=test, target="target", optimization_level="fast", verbosity=0
        )
        assert evaluator.dimensionality_config is not None
        assert evaluator.dimensionality_config.tier == DimensionalityTier.SMALL

    def test_xgb_classifier_instantiates_under_xgboost_2x(self):
        """Regression test: ``use_label_encoder`` was removed in xgboost 2.0
        and instantiation with that kwarg raises ``TypeError``. The
        evaluator must construct XGBClassifier without it.
        """
        from atlantic.optimization.evaluation import ModelEvaluator

        np.random.seed(42)
        n = 60
        train = pd.DataFrame(
            {
                "a": np.random.randn(n),
                "target": np.random.choice(["c0", "c1"], n),
            }
        )
        test = pd.DataFrame(
            {
                "a": np.random.randn(20),
                "target": np.random.choice(["c0", "c1"], 20),
            }
        )
        evaluator = ModelEvaluator(
            train=train, test=test, target="target", optimization_level="fast", verbosity=0
        )
        # _get_model_instances will raise TypeError under xgboost>=2 if
        # use_label_encoder is still passed.
        models = evaluator._get_model_instances()
        assert "xgb_classifier" in models
