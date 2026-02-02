from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from atlantic.core.enums import TaskType
from atlantic.core.schemas import EncodingSelectionResult
from atlantic.encoding.versions import EncodingVersion, EncodingVersionFactory
from atlantic.evaluation.metrics import MetricRegistry
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.preprocessing.imputers import AutoSimpleImputer


class EncodingOptimizer:
    """
    Automated encoding version selection.

    Evaluates all encoding versions and selects the best
    based on model performance.
    """

    def __init__(self, target: str, optimization_level: str = "balanced", verbosity: int = 1):
        """
        Initialize EncodingOptimizer.

        Args:
            target: Target column name.
            optimization Optimization optimization level.
            verbosity: Logging verbosity (0=silent, 1=progress, 2=detailed).
        """
        self.target = target
        self._optimization_level = optimization_level
        self._verbosity = verbosity
        self._selected_version: Optional[str] = None
        self._performance_scores: Dict[str, float] = {}
        self._pred_type: Optional[str] = None
        self._n_classes: Optional[int] = None

    def _detect_task_type(self, train: pd.DataFrame) -> None:
        """Detect prediction task type."""
        target_dtype = train[self.target].dtype

        if target_dtype in ["int", "int32", "int64", "float", "float32", "float64"]:
            self._pred_type = "Reg"
            self._n_classes = None
        else:
            self._pred_type = "Class"
            self._n_classes = train[self.target].nunique()

    def _get_eval_metric(self) -> str:
        """Get evaluation metric based on task type."""
        task_type = TaskType.REGRESSION if self._pred_type == "Reg" else TaskType.CLASSIFICATION
        return MetricRegistry.get_primary_metric_name(task_type, self._n_classes)

    def _impute_if_needed(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply simple imputation if nulls present."""
        if train.isnull().sum().sum() > 0 or test.isnull().sum().sum() > 0:
            imputer = AutoSimpleImputer(strategy="mean", target=self.target)
            imputer.fit(train)
            train = imputer.transform(train.copy())
            test = imputer.transform(test.copy())
        return train, test

    def select_best_version(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> EncodingSelectionResult:
        """
        Evaluate all versions and select the best.

        Returns:
            EncodingSelectionResult with selection details.
        """
        self._detect_task_type(train)
        eval_metric = self._get_eval_metric()

        # Create encoding version helper
        ev = EncodingVersion(train.copy(), test.copy(), self.target)

        # Get all versions
        if self._verbosity >= 1:
            print("Fitting Encoding Versions...")

        versions = {
            "v1": ev.encoding_v1,
            "v2": ev.encoding_v2,
            "v3": ev.encoding_v3,
            "v4": ev.encoding_v4,
        }

        encoded_data = {}
        iterator = (
            tqdm(versions.items(), desc="Encoding", ncols=75)
            if self._verbosity >= 1
            else versions.items()
        )

        for version_name, version_func in iterator:
            train_enc, test_enc = version_func()
            train_enc, test_enc = self._impute_if_needed(train_enc, test_enc)
            encoded_data[version_name] = (train_enc, test_enc)

        # Evaluate each version
        if self._verbosity >= 1:
            print("\nEvaluating Encoding Versions...")

        for version_name, (train_enc, test_enc) in encoded_data.items():
            if self._verbosity >= 2:
                print(f"Encoding Version {version_name.upper()} Loading")

            evaluator = ModelEvaluator(
                train=train_enc,
                test=test_enc,
                target=self.target,
                optimization_level=self._optimization_level,
                verbosity=0,
            )

            metrics = evaluator.auto_evaluate()
            score = metrics[eval_metric].iloc[0]
            self._performance_scores[version_name] = score

        # Select best version
        if self._pred_type == "Reg":
            # Lower is better for regression
            best_version = min(self._performance_scores, key=self._performance_scores.get)
        else:
            # Higher is better for classification
            best_version = max(self._performance_scores, key=self._performance_scores.get)

        self._selected_version = best_version
        best_score = self._performance_scores[best_version]

        # Print results
        if self._verbosity >= 1:
            print("\nPredictive Performance Encoding Versions:")
            for v, score in self._performance_scores.items():
                desc = EncodingVersionFactory.describe_version(v)
                print(f"  Version {v.upper()} [{desc}]: {score:.4f}")

            print(
                f"\nEncoding Version {best_version.upper()} selected "
                f"with {eval_metric} of: {best_score:.4f}"
            )

        return EncodingSelectionResult(
            selected_version=best_version,
            performance_scores=self._performance_scores,
            best_score=best_score,
            metric_name=eval_metric,
        )

    @property
    def selected_version(self) -> Optional[str]:
        """Get selected encoding version."""
        return self._selected_version

    @property
    def performance_scores(self) -> Dict[str, float]:
        """Get performance scores for all versions."""
        return self._performance_scores
