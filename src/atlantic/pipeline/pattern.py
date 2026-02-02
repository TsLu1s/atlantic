from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from atlantic.core.enums import TaskType
from atlantic.core.mixins import TargetTypeMixin
from atlantic.core.schemas import EncodingSelectionResult, ImputationSelectionResult
from atlantic.encoding.optimizer import EncodingOptimizer
from atlantic.encoding.versions import EncodingVersion
from atlantic.evaluation.metrics import MetricRegistry
from atlantic.feature_selection.vif_selector import VIFFeatureSelector
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.preprocessing.imputers import AutoIterativeImputer, AutoKNNImputer, AutoSimpleImputer
from atlantic.preprocessing.registry import ImputerRegistry


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
    ):
        """
        Initialize Pattern optimizer.

        Args:
            train: Training DataFrame.
            test: Testing DataFrame.
            target: Target column name.
            optimization Optimization optimization level.
            verbosity: Logging verbosity level.
        """
        self.train = train
        self.test = test
        self.target = target
        self._optimization_level = optimization_level
        self._verbosity = verbosity

        # Detect target type
        self._detect_target_type(train)

        # Selection results
        self.enc_method: Optional[str] = None
        self.imp_method: Optional[str] = None
        self._imputer = None
        self.perf: Optional[float] = None

        # Encoding optimizer
        self._encoding_optimizer = EncodingOptimizer(
            target=target, optimization_level=optimization_level, verbosity=verbosity
        )

    def _get_eval_metric(self) -> str:
        """Get evaluation metric based on task type."""
        task_type = TaskType.REGRESSION if self.pred_type == "Reg" else TaskType.CLASSIFICATION
        return MetricRegistry.get_primary_metric_name(task_type, self.n_classes)

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

    def imputation_selection(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select optimal imputation method based on performance.

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

        # Test different imputation methods
        imputation_methods = {
            "Simple": AutoSimpleImputer(strategy="mean"),
            "KNN": AutoKNNImputer(n_neighbors=3),
            "Iterative": AutoIterativeImputer(max_iter=10, random_state=42),
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
            evaluator = ModelEvaluator(
                train=train_imp,
                test=test_imp,
                target=self.target,
                optimization_level=self._optimization_level,
                verbosity=0,
            )

            metrics = evaluator.auto_evaluate()
            score = metrics[eval_metric].iloc[0]

            performance_scores[method_name] = score
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
            print(f"\n{best_method} Imputation selected with " f"{eval_metric} of: {self.perf:.4f}")

        return self.train, self.test

    def vif_performance(
        self, vif_threshold: float = 10.0, perf_: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate and apply VIF-based feature selection.

        Returns:
            Tuple of (filtered_train, filtered_test).
        """
        eval_metric = self._get_eval_metric()

        train_vif = self.train.copy()
        test_vif = self.test.copy()

        # Get VIF-selected columns
        vif_selector = VIFFeatureSelector(target=self.target, vif_threshold=vif_threshold)

        try:
            vif_selector.fit(train_vif)
            cols_vif = vif_selector.selected_features + [self.target]
        except Exception:
            # If VIF fails, keep all columns
            cols_vif = list(train_vif.columns)

        if self._verbosity >= 1:
            print(f"\nSelected VIF Columns: {len(cols_vif) - 1}")
            print(f"Removed Columns by VIF: {train_vif.shape[1] - len(cols_vif)}")

        train_vif = train_vif[cols_vif]
        test_vif = test_vif[cols_vif]

        apply_vif = False

        # If no columns removed, just update and return
        if self.train.shape[1] - len(cols_vif) == 0:
            self.train = self.train[cols_vif]
            self.test = self.test[cols_vif]
            return self.train, self.test

        # Compare performance
        if perf_ is not None:
            self.perf = perf_

        if self.perf is None:
            evaluator = ModelEvaluator(
                train=self.train.copy(),
                test=self.test.copy(),
                target=self.target,
                optimization_level=self._optimization_level,
                verbosity=0,
            )
            self.perf = evaluator.auto_evaluate()[eval_metric].iloc[0]

        # Evaluate VIF-filtered version
        evaluator_vif = ModelEvaluator(
            train=train_vif,
            test=test_vif,
            target=self.target,
            optimization_level=self._optimization_level,
            verbosity=0,
        )
        perf_vif = evaluator_vif.auto_evaluate()[eval_metric].iloc[0]

        if self._verbosity >= 1:
            print(f"Default Performance: {self.perf:.4f}")
            print(f"VIF Performance: {perf_vif:.4f}")

        # Decide whether to apply VIF
        if self.pred_type == "Reg" and perf_vif < self.perf:
            apply_vif = True
        elif self.pred_type == "Class" and perf_vif > self.perf:
            apply_vif = True

        if apply_vif:
            if self._verbosity >= 1:
                print("\nThe VIF filtering method was applied")
            self.train = train_vif
            self.test = test_vif
        else:
            if self._verbosity >= 1:
                print("\nThe VIF filtering method was not applied")

        return self.train, self.test

    @property
    def imputer(self):
        """Get fitted imputer."""
        return self._imputer
