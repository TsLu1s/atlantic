"""Tests for atlantic.evaluation.metrics module."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.enums import TaskType
from atlantic.core.exceptions import RegistryError
from atlantic.evaluation.metrics import (
    AccuracyMetric,
    EVSMetric,
    F1Metric,
    MAEMetric,
    MAPEMetric,
    MaxErrorMetric,
    MetricRegistry,
    MSEMetric,
    PrecisionMetric,
    R2Metric,
    RecallMetric,
    RMSEMetric,
    metrics_classification,
    metrics_regression,
)


class TestIndividualMetrics:
    def test_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert MAEMetric().compute(y_true, y_pred) == pytest.approx(0.5)

    def test_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        assert MSEMetric().compute(y_true, y_pred) == pytest.approx(1.0 / 3.0)

    def test_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3.0)
        assert RMSEMetric().compute(y_true, y_pred) == pytest.approx(expected)

    def test_r2_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert R2Metric().compute(y_true, y_pred) == pytest.approx(1.0)

    def test_evs(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert EVSMetric().compute(y_true, y_pred) == pytest.approx(1.0)

    def test_mape(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 22.0, 33.0])
        result = MAPEMetric().compute(y_true, y_pred)
        assert result == pytest.approx(0.1)

    def test_max_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        assert MaxErrorMetric().compute(y_true, y_pred) == pytest.approx(2.0)

    def test_precision_binary(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = PrecisionMetric().compute(y_true, y_pred)
        assert 0.0 <= result <= 1.0

    def test_recall_binary(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = RecallMetric().compute(y_true, y_pred)
        assert 0.0 <= result <= 1.0

    def test_f1_binary(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = F1Metric().compute(y_true, y_pred)
        assert 0.0 <= result <= 1.0

    def test_accuracy(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        assert AccuracyMetric().compute(y_true, y_pred) == pytest.approx(1.0)

    def test_metric_properties(self):
        m = MAEMetric()
        assert m.name == "Mean Absolute Error"
        assert m.key == "mae"
        assert m.greater_is_better is False
        assert m.task_type == TaskType.REGRESSION


class TestMetricRegistry:
    def test_compute_all_regression(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = MetricRegistry.compute_all(y_true, y_pred, TaskType.REGRESSION)
        assert isinstance(result, pd.DataFrame)
        assert "Mean Absolute Error" in result.columns
        assert "R2 Score" in result.columns

    def test_compute_all_classification(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = MetricRegistry.compute_all(y_true, y_pred, TaskType.CLASSIFICATION, n_classes=2)
        assert isinstance(result, pd.DataFrame)
        assert "Precision" in result.columns

    def test_accuracy_included_for_binary(self):
        """Regression test: accuracy must be present in the binary
        classification metric output. The previous implementation skipped
        it for ``n_classes == 2``, producing a different schema for
        binary vs. multiclass and breaking callers that index by metric
        name."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        binary = MetricRegistry.compute_all(
            y_true, y_pred, TaskType.CLASSIFICATION, n_classes=2
        )
        multi = MetricRegistry.compute_all(
            np.array([0, 1, 2, 0, 1]),
            np.array([0, 1, 2, 0, 1]),
            TaskType.CLASSIFICATION,
            n_classes=3,
        )
        assert "Accuracy" in binary.columns
        assert "Accuracy" in multi.columns
        # The two should have the same set of metric columns (schema parity).
        assert set(binary.columns) == set(multi.columns)

    def test_get_primary_metric_regression(self):
        assert MetricRegistry.get_primary_metric(TaskType.REGRESSION) == "mae"

    def test_get_primary_metric_binary(self):
        assert MetricRegistry.get_primary_metric(TaskType.CLASSIFICATION, n_classes=2) == "precision"

    def test_get_primary_metric_multiclass(self):
        assert MetricRegistry.get_primary_metric(TaskType.CLASSIFICATION, n_classes=5) == "f1"

    def test_get_primary_metric_name(self):
        assert MetricRegistry.get_primary_metric_name(TaskType.REGRESSION) == "Mean Absolute Error"
        assert MetricRegistry.get_primary_metric_name(TaskType.CLASSIFICATION, 2) == "Precision"
        assert MetricRegistry.get_primary_metric_name(TaskType.CLASSIFICATION, 5) == "F1"

    def test_list_available(self):
        reg = MetricRegistry.list_available(TaskType.REGRESSION)
        assert "mae" in reg
        clf = MetricRegistry.list_available(TaskType.CLASSIFICATION)
        assert "precision" in clf
        all_metrics = MetricRegistry.list_available()
        assert len(all_metrics) >= len(reg) + len(clf)

    def test_get_metric(self):
        m = MetricRegistry.get("mae")
        assert m.key == "mae"

    def test_get_unknown_raises(self):
        with pytest.raises(RegistryError):
            MetricRegistry.get("nonexistent", TaskType.REGRESSION)


class TestConvenienceFunctions:
    def test_metrics_regression(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = metrics_regression(y_true, y_pred)
        assert "Mean Absolute Error" in result.columns

    def test_metrics_classification(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        result = metrics_classification(y_true, y_pred)
        assert isinstance(result, pd.DataFrame)
