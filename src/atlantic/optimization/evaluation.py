from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import LabelEncoder

from atlantic.core.enums import OptimizationLevel, TaskType
from atlantic.core.mixins import ColumnTypeMixin, TargetTypeMixin
from atlantic.evaluation.metrics import MetricRegistry
from atlantic.optimization.base import BaseOptimizerAdapter
from atlantic.optimization.dimensionality import (
    DimensionalityConfig,
    DimensionalityStrategy,
    ModelHyperparameterRanges,
)
from atlantic.optimization.optuna_adapter import OptunaAdapter


@dataclass
class ModelConfig:
    """Configuration for model hyperparameter ranges."""

    rf_regressor: Dict = field(
        default_factory=lambda: {
            "n_estimators": (50, 200),
            "max_depth": (5, 32),
            "min_samples_split": (2, 25),
        }
    )

    et_regressor: Dict = field(
        default_factory=lambda: {
            "n_estimators": (50, 200),
            "max_depth": (5, 32),
            "min_samples_split": (2, 25),
        }
    )

    xgb_regressor: Dict = field(
        default_factory=lambda: {
            "n_estimators": (50, 200),
            "max_depth": (5, 25),
            "learning_rate": (0.01, 0.1),
        }
    )

    rf_classifier: Dict = field(
        default_factory=lambda: {
            "n_estimators": (60, 250),
            "max_depth": (10, 50),
            "min_samples_split": (2, 20),
        }
    )

    et_classifier: Dict = field(
        default_factory=lambda: {
            "n_estimators": (60, 250),
            "max_depth": (10, 50),
            "min_samples_split": (2, 20),
        }
    )

    xgb_classifier: Dict = field(
        default_factory=lambda: {
            "n_estimators": (60, 250),
            "max_depth": (10, 20),
            "learning_rate": (0.05, 0.1),
        }
    )

    def adjust_for_complexity(self, complexity: str) -> None:
        """Adjust ranges based on complexity level."""
        ranges = ModelHyperparameterRanges.for_complexity(complexity)

        # Update regression models
        self.rf_regressor["n_estimators"] = ranges.rf_n_estimators
        self.rf_regressor["max_depth"] = ranges.rf_max_depth
        self.et_regressor["n_estimators"] = ranges.rf_n_estimators
        self.et_regressor["max_depth"] = ranges.rf_max_depth
        self.xgb_regressor["n_estimators"] = ranges.xgb_n_estimators
        self.xgb_regressor["max_depth"] = ranges.xgb_max_depth
        self.xgb_regressor["learning_rate"] = ranges.xgb_learning_rate

        # Update classification models
        self.rf_classifier["n_estimators"] = ranges.rf_n_estimators
        self.rf_classifier["max_depth"] = ranges.rf_max_depth
        self.et_classifier["n_estimators"] = ranges.rf_n_estimators
        self.et_classifier["max_depth"] = ranges.rf_max_depth
        self.xgb_classifier["n_estimators"] = ranges.xgb_n_estimators
        self.xgb_classifier["max_depth"] = ranges.xgb_max_depth
        self.xgb_classifier["learning_rate"] = ranges.xgb_learning_rate


class ModelEvaluator(ColumnTypeMixin, TargetTypeMixin):
    """
    Advanced model evaluation and hyperparameter optimization.

    Supports both regression and classification with automatic
    dimensionality-based configuration.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        optimization_level: str = "balanced",
        random_state: int = 42,
        verbosity: int = 0,
    ):
        """
        Initialize ModelEvaluator.

        Args:
            train: Training DataFrame.
            test: Testing DataFrame.
            target: Target column name.
            optimization Optimization optimization level.
            random_state: Random seed for reproducibility.
            verbosity: Logging verbosity level.
        """
        self.train = train
        self.test = test
        self.target = target
        self._random_state = random_state
        self._verbosity = verbosity

        # Detect task type
        self._detect_target_type(train)

        # Get dimensionality config
        n_rows = len(train)
        n_features = len(train.columns) - 1
        self._dim_config = DimensionalityStrategy.get_config_from_string(
            n_rows, n_features, optimization_level
        )

        # Initialize model config with complexity adjustment
        self._model_config = ModelConfig()
        self._model_config.adjust_for_complexity(self._dim_config.model_complexity)

        # Initialize optimizer
        self._optimizer = OptunaAdapter(random_state=random_state, verbosity=verbosity)

        # Results storage
        self.metrics: Optional[pd.DataFrame] = None
        self._detailed_metrics: Optional[pd.DataFrame] = None
        self.hparameters_list: List[Dict] = []
        self.metrics_list: List[pd.DataFrame] = []

        # Label encoder for classification
        self._label_encoder: Optional[LabelEncoder] = None

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare train/test data with proper encoding."""
        X_train = self.train.drop(columns=[self.target])
        X_test = self.test.drop(columns=[self.target])
        y_train = self.train[self.target].values
        y_test = self.test[self.target].values

        if self.pred_type == "Class":
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            y_test = self._label_encoder.transform(y_test)

        return X_train, X_test, y_train, y_test

    def _get_model_instances(self) -> Dict[str, Any]:
        """Get model instances based on task type."""
        if self.pred_type == "Reg":
            return {
                "rf_regressor": RandomForestRegressor(random_state=self._random_state),
                "et_regressor": ExtraTreesRegressor(random_state=self._random_state),
                "xgb_regressor": xgb.XGBRegressor(random_state=self._random_state, verbosity=0),
            }
        return {
            "rf_classifier": RandomForestClassifier(random_state=self._random_state),
            "et_classifier": ExtraTreesClassifier(random_state=self._random_state),
            "xgb_classifier": xgb.XGBClassifier(
                random_state=self._random_state,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
        }

    def _suggest_hyperparameters(self, trial: "optuna.Trial", model_type: str) -> Dict[str, Any]:
        """Generate hyperparameter suggestions for a model."""
        config = getattr(self._model_config, model_type)
        params = {}

        for param_name, value_range in config.items():
            if param_name == "num_class":
                params[param_name] = self.n_classes
                continue
            elif param_name == "objective":
                params[param_name] = "multi:softmax"
                continue

            full_name = f"{model_type}_{param_name}"

            if param_name == "learning_rate":
                params[param_name] = trial.suggest_float(
                    full_name, value_range[0], value_range[1], log=True
                )
            else:
                params[param_name] = trial.suggest_int(full_name, value_range[0], value_range[1])

        return params

    def _objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective function."""
        X_train, X_test, y_train, y_test = self._prepare_data()

        models = self._get_model_instances()
        task_type = TaskType.REGRESSION if self.pred_type == "Reg" else TaskType.CLASSIFICATION

        results = []
        hparams = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for model_name, model in models.items():
                # Skip extra trees for high complexity
                if self._dim_config.model_complexity == "low" and "et_" in model_name:
                    continue

                params = self._suggest_hyperparameters(trial, model_name)
                model.set_params(**params)
                model.fit(X_train, y_train)

                pred = model.predict(X_test)
                metrics_df = MetricRegistry.compute_all(y_test, pred, task_type, self.n_classes)
                metrics_df["Model"] = model_name.upper()
                results.append(metrics_df)

                hparams[f"{model_name}_params"] = params

        # Combine results
        metrics_df = pd.concat(results, axis=0)
        metrics_df["iteration"] = len(self.metrics_list) + 1
        self.metrics_list.append(metrics_df)

        hparams["iteration"] = len(self.hparameters_list) + 1
        self.hparameters_list.append(hparams)

        # Return primary metric value
        primary_metric = MetricRegistry.get_primary_metric_name(task_type, self.n_classes)
        return metrics_df[primary_metric].mean()

    def auto_evaluate(self) -> pd.DataFrame:
        """
        Perform automated model evaluation with hyperparameter optimization.

        Returns:
            DataFrame with aggregated evaluation metrics.
        """
        # Determine optimization direction
        task_type = TaskType.REGRESSION if self.pred_type == "Reg" else TaskType.CLASSIFICATION
        direction = "minimize" if self.pred_type == "Reg" else "maximize"

        # Create study
        self._optimizer.create_study(direction=direction, study_name=f"{self.pred_type} Evaluation")

        # Run optimization
        self._optimizer.optimize(
            objective=self._objective,
            n_trials=self._dim_config.n_trials,
            show_progress=self._verbosity > 0,
        )

        # Aggregate results
        self.metrics = pd.concat(self.metrics_list)

        # Sort by primary metric
        primary_metric = MetricRegistry.get_primary_metric_name(task_type, self.n_classes)
        ascending = self.pred_type == "Reg"

        self.metrics = self.metrics.sort_values(["Model", primary_metric], ascending=ascending)

        self._detailed_metrics = self.metrics.copy()

        # Get best per model and average
        self.metrics = self.metrics.groupby("Model").first().mean(axis=0).to_frame().T

        if "iteration" in self.metrics.columns:
            self.metrics = self.metrics.drop(columns=["iteration"])

        return self.metrics

    @property
    def dimensionality_config(self) -> DimensionalityConfig:
        """Get dimensionality configuration."""
        return self._dim_config

    @property
    def detailed_metrics(self) -> Optional[pd.DataFrame]:
        """Get detailed metrics from all trials."""
        return self._detailed_metrics
