"""
Atlantic - Pattern Optimizer.

Sequential, performance-driven selection of the preprocessing steps:
encoding version, imputation method and VIF filtering.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from atlantic.core.enums import TaskType
from atlantic.core.exceptions import ValidationError
from atlantic.core.mixins import TargetTypeMixin
from atlantic.encoding.optimizer import EncodingOptimizer
from atlantic.encoding.versions import EncodingVersion
from atlantic.evaluation.metrics import MetricRegistry
from atlantic.feature_selection.vif_selector import VIFFeatureSelector
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.preprocessing.imputers import AutoIterativeImputer, AutoKNNImputer, AutoSimpleImputer


class Pattern(TargetTypeMixin):
    """
    Sequential optimization of preprocessing methods.

    Provides automated selection of optimal encoding and imputation
    strategies based on model performance evaluation.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        optimization_level: str = "balanced",
        verbosity: int = 1,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize Pattern optimizer.

        Args:
            train: Training DataFrame.
            test: Testing DataFrame.
            target: Target column name.
            optimization_level: Optimization level ("fast", "balanced", "thorough").
            verbosity: Logging verbosity level.
            random_state: Seed shared by every model evaluation and by the
                iterative imputer, so all candidates are scored under
                identical conditions.
            n_jobs: Parallelism of the evaluation models (see ``ModelEvaluator``).
        """
        self.train = train
        self.test = test
        self.target = target
        self._optimization_level = optimization_level
        self._verbosity = verbosity
        self._random_state = random_state
        self._n_jobs = n_jobs

        # Detect target type
        self._detect_target_type(train)

        # Selection results
        self.enc_method: Optional[str] = None
        self.imp_method: Optional[str] = None
        self._imputer = None
        self.perf: Optional[float] = None
        self.vif_removed: List[str] = []
        self.vif_applied: bool = False

        # Encoding optimizer
        self._encoding_optimizer = EncodingOptimizer(
            target=target,
            optimization_level=optimization_level,
            verbosity=verbosity,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _get_eval_metric(self) -> str:
        """Get evaluation metric based on task type."""
        task_type = TaskType.REGRESSION if self.pred_type == "Reg" else TaskType.CLASSIFICATION
        return MetricRegistry.get_primary_metric_name(task_type, self.n_classes)

    def _evaluate(self, train: pd.DataFrame, test: pd.DataFrame, eval_metric: str) -> float:
        """Score a prepared train/test pair on the primary metric."""
        evaluator = ModelEvaluator(
            train=train,
            test=test,
            target=self.target,
            optimization_level=self._optimization_level,
            random_state=self._random_state,
            verbosity=0,
            n_jobs=self._n_jobs,
        )
        return float(evaluator.auto_evaluate()[eval_metric].iloc[0])

    def encoding_selection(self) -> str:
        """
        Select optimal encoding version based on performance.

        Returns:
            Selected encoding version name.
        """
        result = self._encoding_optimizer.select_best_version(self.train.copy(), self.test.copy())

        # Map version to method name
        version_map = {
            "v1": "Encoding Version 1",
            "v2": "Encoding Version 2",
            "v3": "Encoding Version 3",
            "v4": "Encoding Version 4",
        }

        self.enc_method = version_map.get(result.selected_version, result.selected_version)
        self.perf = result.best_score

        return self.enc_method

    def imputation_selection(
        self, simple_score: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select optimal imputation method based on performance.

        Args:
            simple_score: Score already measured for mean imputation on this
                exact encoding (the encoding selection scores every version
                with mean imputation, under the same seed and optimization
                level). When given, the "Simple" candidate reuses it instead
                of repeating an evaluation whose result is already known.

        Returns:
            Tuple of (imputed_train, imputed_test).
        """
        eval_metric = self._get_eval_metric()

        # Apply selected encoding
        ev = EncodingVersion(train=self.train.copy(), test=self.test.copy(), target=self.target)

        version_map = {
            "Encoding Version 1": ev.encoding_v1,
            "Encoding Version 2": ev.encoding_v2,
            "Encoding Version 3": ev.encoding_v3,
            "Encoding Version 4": ev.encoding_v4,
        }

        if self.enc_method in version_map:
            self.train, self.test = version_map[self.enc_method]()
        else:
            self.train, self.test = ev.encoding_v4()

        # Test different imputation methods. Each imputer must receive the
        # target name so it excludes the target column from
        # ``_numeric_columns`` - otherwise a numeric regression target would
        # be imputed at inference time, which would silently overwrite real
        # target values when transform is run on data with NaNs.
        imputation_methods = {
            "Simple": AutoSimpleImputer(strategy="mean", target=self.target),
            "KNN": AutoKNNImputer(n_neighbors=3, target=self.target),
            "Iterative": AutoIterativeImputer(
                max_iter=10, random_state=self._random_state, target=self.target
            ),
        }

        performance_scores = {}
        imputed_data = {}

        if self._verbosity >= 1:
            print("\nEvaluating Imputation Methods...")

        for method_name, imputer in imputation_methods.items():
            if self._verbosity >= 2:
                print(f"{method_name} Imputation Loading")

            # Fit and transform
            imputer.fit(self.train)
            train_imp = imputer.transform(self.train.copy())
            test_imp = imputer.transform(self.test.copy())

            # Evaluate
            if method_name == "Simple" and simple_score is not None:
                performance_scores[method_name] = float(simple_score)
            else:
                performance_scores[method_name] = self._evaluate(train_imp, test_imp, eval_metric)
            imputed_data[method_name] = (train_imp, test_imp, imputer)

        # Select best method
        if self.pred_type == "Reg":
            best_method = min(performance_scores, key=performance_scores.get)
        else:
            best_method = max(performance_scores, key=performance_scores.get)

        self.imp_method = best_method
        self.perf = performance_scores[best_method]
        self.train, self.test, self._imputer = imputed_data[best_method]

        # Print results
        if self._verbosity >= 1:
            print("\nPredictive Performance Null Imputation Versions:")
            for method, score in performance_scores.items():
                print(f"  {method} Performance: {score:.4f}")
            print(f"\n{best_method} Imputation selected with {eval_metric} of: {self.perf:.4f}")

        return self.train, self.test

    def vif_performance(
        self, vif_threshold: float = 10.0, perf_: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate and apply VIF-based feature selection.

        VIF filtering is only kept when it improves the primary metric over
        the unfiltered baseline. ``vif_removed`` lists the columns that were
        dropped when it is applied (empty otherwise) and ``vif_applied``
        records the decision.

        Returns:
            Tuple of (filtered_train, filtered_test).
        """
        eval_metric = self._get_eval_metric()
        self.vif_removed = []
        self.vif_applied = False

        train_vif = self.train.copy()
        test_vif = self.test.copy()

        # Get VIF-selected columns
        vif_selector = VIFFeatureSelector(target=self.target, vif_threshold=vif_threshold)

        try:
            vif_selector.fit(train_vif)
            cols_vif = vif_selector.selected_features + [self.target]
        except (ValidationError, np.linalg.LinAlgError) as e:
            # VIF cannot be computed on this data (e.g. remaining nulls):
            # keep every column, but say so instead of failing silently.
            warnings.warn(
                f"VIF filtering skipped: {e}",
                UserWarning,
                stacklevel=2,
            )
            cols_vif = list(train_vif.columns)

        removed = [col for col in train_vif.columns if col not in set(cols_vif)]

        if self._verbosity >= 1:
            print(f"\nSelected VIF Columns: {len(cols_vif) - 1}")
            print(f"Removed Columns by VIF: {len(removed)}")

        train_vif = train_vif[cols_vif]
        test_vif = test_vif[cols_vif]

        # If no columns removed, just update and return
        if not removed:
            self.train = self.train[cols_vif]
            self.test = self.test[cols_vif]
            return self.train, self.test

        # Compare performance
        if perf_ is not None:
            self.perf = perf_

        if self.perf is None:
            self.perf = self._evaluate(self.train.copy(), self.test.copy(), eval_metric)

        # Evaluate VIF-filtered version
        perf_vif = self._evaluate(train_vif, test_vif, eval_metric)

        if self._verbosity >= 1:
            print(f"Default Performance: {self.perf:.4f}")
            print(f"VIF Performance: {perf_vif:.4f}")

        # Decide whether to apply VIF
        apply_vif = (self.pred_type == "Reg" and perf_vif < self.perf) or (
            self.pred_type == "Class" and perf_vif > self.perf
        )

        if apply_vif:
            if self._verbosity >= 1:
                print("\nThe VIF filtering method was applied")
            self.train = train_vif
            self.test = test_vif
            self.vif_removed = removed
            self.vif_applied = True
        else:
            if self._verbosity >= 1:
                print("\nThe VIF filtering method was not applied")

        return self.train, self.test

    @property
    def imputer(self):
        """Get fitted imputer."""
        return self._imputer
