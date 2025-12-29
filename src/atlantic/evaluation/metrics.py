from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    explained_variance_score,
    max_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from atlantic.core.enums import TaskType
from atlantic.core.exceptions import RegistryError


class BaseMetricStrategy(ABC):
    """Abstract base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name."""
        ...
    
    @property
    @abstractmethod
    def key(self) -> str:
        """Short identifier key."""
        ...
    
    @property
    @abstractmethod
    def greater_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        ...
    
    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Task type this metric applies to."""
        ...
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute metric value.

        Returns:
            Computed metric value.
        """
        ...


# =============================================================================
# Regression Metrics
# =============================================================================

class MAEMetric(BaseMetricStrategy):
    """Mean Absolute Error metric."""
    
    @property
    def name(self) -> str:
        return "Mean Absolute Error"
    
    @property
    def key(self) -> str:
        return "mae"
    
    @property
    def greater_is_better(self) -> bool:
        return False
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_error(y_true, y_pred))


class MAPEMetric(BaseMetricStrategy):
    """Mean Absolute Percentage Error metric."""
    
    @property
    def name(self) -> str:
        return "Mean Absolute Percentage Error"
    
    @property
    def key(self) -> str:
        return "mape"
    
    @property
    def greater_is_better(self) -> bool:
        return False
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_percentage_error(y_true, y_pred))


class MSEMetric(BaseMetricStrategy):
    """Mean Squared Error metric."""
    
    @property
    def name(self) -> str:
        return "Mean Squared Error"
    
    @property
    def key(self) -> str:
        return "mse"
    
    @property
    def greater_is_better(self) -> bool:
        return False
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_squared_error(y_true, y_pred))


class RMSEMetric(BaseMetricStrategy):
    """Root Mean Squared Error metric."""
    
    @property
    def name(self) -> str:
        return "Root Mean Squared Error"
    
    @property
    def key(self) -> str:
        return "rmse"
    
    @property
    def greater_is_better(self) -> bool:
        return False
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class R2Metric(BaseMetricStrategy):
    """R-squared (coefficient of determination) metric."""
    
    @property
    def name(self) -> str:
        return "R2 Score"
    
    @property
    def key(self) -> str:
        return "r2"
    
    @property
    def greater_is_better(self) -> bool:
        return True
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(r2_score(y_true, y_pred))


class EVSMetric(BaseMetricStrategy):
    """Explained Variance Score metric."""
    
    @property
    def name(self) -> str:
        return "Explained Variance Score"
    
    @property
    def key(self) -> str:
        return "evs"
    
    @property
    def greater_is_better(self) -> bool:
        return True
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(explained_variance_score(y_true, y_pred))


class MaxErrorMetric(BaseMetricStrategy):
    """Maximum Error metric."""
    
    @property
    def name(self) -> str:
        return "Max Error"
    
    @property
    def key(self) -> str:
        return "max_error"
    
    @property
    def greater_is_better(self) -> bool:
        return False
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(max_error(y_true, y_pred))


# =============================================================================
# Classification Metrics
# =============================================================================

class PrecisionMetric(BaseMetricStrategy):
    """Precision metric."""
    
    def __init__(self, average: str = "weighted"):
        self._average = average
    
    @property
    def name(self) -> str:
        return "Precision"
    
    @property
    def key(self) -> str:
        return "precision"
    
    @property
    def greater_is_better(self) -> bool:
        return True
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes == 2 else self._average
        return float(precision_score(y_true, y_pred, average=avg, zero_division=0))


class RecallMetric(BaseMetricStrategy):
    """Recall metric."""
    
    def __init__(self, average: str = "weighted"):
        self._average = average
    
    @property
    def name(self) -> str:
        return "Recall"
    
    @property
    def key(self) -> str:
        return "recall"
    
    @property
    def greater_is_better(self) -> bool:
        return True
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes == 2 else self._average
        return float(recall_score(y_true, y_pred, average=avg, zero_division=0))


class F1Metric(BaseMetricStrategy):
    """F1 Score metric."""
    
    def __init__(self, average: str = "weighted"):
        self._average = average
    
    @property
    def name(self) -> str:
        return "F1"
    
    @property
    def key(self) -> str:
        return "f1"
    
    @property
    def greater_is_better(self) -> bool:
        return True
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes == 2 else self._average
        return float(f1_score(y_true, y_pred, average=avg, zero_division=0))


class AccuracyMetric(BaseMetricStrategy):
    """Accuracy metric."""
    
    @property
    def name(self) -> str:
        return "Accuracy"
    
    @property
    def key(self) -> str:
        return "accuracy"
    
    @property
    def greater_is_better(self) -> bool:
        return True
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(accuracy_score(y_true, y_pred))


# =============================================================================
# Metric Registry
# =============================================================================

class MetricRegistry:
    """Registry for metric strategies."""
    
    _regression_metrics: Dict[str, Type[BaseMetricStrategy]] = {}
    _classification_metrics: Dict[str, Type[BaseMetricStrategy]] = {}
    _instances: Dict[str, BaseMetricStrategy] = {}
    
    @classmethod
    def register(cls, metric_class: Type[BaseMetricStrategy]) -> None:
        """
        Register a metric strategy class.
        
        Args:
            metric_class: Metric class to register.
        """
        try:
            instance = metric_class()
        except TypeError:
            instance = metric_class(average="weighted")
        
        key = instance.key
        task = instance.task_type
        
        if task == TaskType.REGRESSION:
            cls._regression_metrics[key] = metric_class
        else:
            cls._classification_metrics[key] = metric_class
        
        cls._instances[key] = instance
    
    @classmethod
    def get(cls, key: str, task_type: Optional[TaskType] = None) -> BaseMetricStrategy:
        """
        Get metric instance by key.

        Returns:
            Metric instance.
        """
        if key in cls._instances:
            return cls._instances[key]
        
        if task_type == TaskType.REGRESSION:
            registry = cls._regression_metrics
        elif task_type == TaskType.CLASSIFICATION:
            registry = cls._classification_metrics
        else:
            registry = {**cls._regression_metrics, **cls._classification_metrics}
        
        if key not in registry:
            available = list(registry.keys())
            raise RegistryError(f"Metric '{key}' not found. Available: {available}")
        
        return registry[key]()
    
    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        n_classes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute all metrics for given task type.

        Returns:
            DataFrame with all metric values.
        """
        registry = (
            cls._regression_metrics 
            if task_type == TaskType.REGRESSION 
            else cls._classification_metrics
        )
        
        results = {}
        for key, metric_class in registry.items():
            if key == "accuracy" and n_classes == 2:
                continue
            
            try:
                metric = metric_class()
            except TypeError:
                metric = metric_class(average="weighted")
            
            results[metric.name] = metric.compute(y_true, y_pred)
        
        return pd.DataFrame(results, index=[0])
    
    @classmethod
    def get_primary_metric(
        cls, 
        task_type: TaskType, 
        n_classes: Optional[int] = None
    ) -> str:
        """Get primary evaluation metric key for task type."""
        if task_type == TaskType.REGRESSION:
            return "mae"
        return "f1" if n_classes and n_classes > 2 else "precision"
    
    @classmethod
    def get_primary_metric_name(
        cls,
        task_type: TaskType,
        n_classes: Optional[int] = None
    ) -> str:
        """Get primary evaluation metric name for task type."""
        if task_type == TaskType.REGRESSION:
            return "Mean Absolute Error"
        return "F1" if n_classes and n_classes > 2 else "Precision"
    
    @classmethod
    def list_available(cls, task_type: Optional[TaskType] = None) -> List[str]:
        """List available metrics."""
        if task_type == TaskType.REGRESSION:
            return list(cls._regression_metrics.keys())
        elif task_type == TaskType.CLASSIFICATION:
            return list(cls._classification_metrics.keys())
        return list(cls._regression_metrics.keys()) + list(cls._classification_metrics.keys())


# Auto-register all metrics
MetricRegistry.register(MAEMetric)
MetricRegistry.register(MAPEMetric)
MetricRegistry.register(MSEMetric)
MetricRegistry.register(RMSEMetric)
MetricRegistry.register(R2Metric)
MetricRegistry.register(EVSMetric)
MetricRegistry.register(MaxErrorMetric)
MetricRegistry.register(PrecisionMetric)
MetricRegistry.register(RecallMetric)
MetricRegistry.register(F1Metric)
MetricRegistry.register(AccuracyMetric)


# =============================================================================
# Convenience Functions
# =============================================================================

def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate regression metrics.

    Returns:
        DataFrame with all regression metrics.
    """
    return MetricRegistry.compute_all(y_true, y_pred, TaskType.REGRESSION)


def metrics_classification(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    n_classes: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate classification metrics.

    Returns:
        DataFrame with all classification metrics.
    """
    if n_classes is None:
        n_classes = len(np.unique(y_true))
    return MetricRegistry.compute_all(y_true, y_pred, TaskType.CLASSIFICATION, n_classes)