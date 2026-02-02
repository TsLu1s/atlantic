"""
Atlantic - Automated Data Preprocessing Framework for Supervised Machine Learning.

A comprehensive framework for automating data preprocessing through integration
and validated application of various preprocessing mechanisms including feature
engineering, automated feature selection, multiple encoding versions, and null
imputation methods.

Example:
    >>> from atlantic import Atlantic
    >>> 
    >>> # Direct usage
    >>> atl = Atlantic(X=data, target="target_column")
    >>> atl.fit_processing(split_ratio=0.75, relevance=0.99)
    >>> processed = atl.data_processing(X=new_data)
    >>> 
    >>> # Builder pattern
    >>> pipeline = (Atlantic.builder()
    ...     .with_feature_selection(method="h2o", relevance=0.95)
    ...     .with_encoding(scaler="standard", encoder="ifrequency")
    ...     .with_imputation(method="knn", auto_select=True)
    ...     .with_vif_filtering(threshold=10.0)
    ...     .build())
    >>> 
    >>> processed = pipeline.fit_transform(X=data, target="target")
"""

__version__ = "2.0.0"
__author__ = "Luís Fernando da Silva Santos"

# Core
from atlantic.core.enums import (
    DimensionalityTier,
    EncoderType,
    EncodingVersion,
    FeatureSelectorType,
    ImputerType,
    OptimizationLevel,
    ScalerType,
    TaskType,
)
from atlantic.core.exceptions import (
    AtlanticError,
    EncodingError,
    FeatureSelectionError,
    ImputationError,
    NotFittedError,
    ValidationError,
)

# Analysis (backward compatibility)
from atlantic.core.mixins import ColumnTypeMixin, DataValidationMixin, DateEngineeringMixin
from atlantic.core.schemas import FittedComponents, PipelineConfig, PipelineState

# Data
from atlantic.data.generator import DatasetGenerator
from atlantic.encoding.optimizer import EncodingOptimizer

# Encoding
from atlantic.encoding.versions import EncodingVersion, EncodingVersionFactory

# Evaluation
from atlantic.evaluation.metrics import MetricRegistry, metrics_classification, metrics_regression

# Feature Selection
from atlantic.feature_selection.h2o_selector import H2OFeatureSelector
from atlantic.feature_selection.registry import FeatureSelectorRegistry
from atlantic.feature_selection.vif_selector import VIFFeatureSelector

# Optimization
from atlantic.optimization.dimensionality import DimensionalityConfig, DimensionalityStrategy
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.optimization.optuna_adapter import OptunaAdapter

# Main pipeline
from atlantic.pipeline.atlantic import Atlantic
from atlantic.pipeline.builder import AtlanticBuilder, AtlanticPipeline
from atlantic.pipeline.pattern import Pattern

# Preprocessing
from atlantic.preprocessing.encoders import (
    AutoIFrequencyEncoder,
    AutoLabelEncoder,
    AutoOneHotEncoder,
)
from atlantic.preprocessing.imputers import AutoIterativeImputer, AutoKNNImputer, AutoSimpleImputer
from atlantic.preprocessing.registry import EncoderRegistry, ImputerRegistry, ScalerRegistry
from atlantic.preprocessing.scalers import AutoMinMaxScaler, AutoRobustScaler, AutoStandardScaler

# State
from atlantic.state.pipeline_state import StateManager

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Main pipeline
    "Atlantic",
    "AtlanticBuilder",
    "AtlanticPipeline",
    "Pattern",
    # Enums
    "TaskType",
    "DimensionalityTier",
    "OptimizationLevel",
    "EncodingVersion",
    "ScalerType",
    "EncoderType",
    "ImputerType",
    "FeatureSelectorType",
    # Exceptions
    "AtlanticError",
    "NotFittedError",
    "ValidationError",
    "FeatureSelectionError",
    "EncodingError",
    "ImputationError",
    # Schemas
    "PipelineConfig",
    "PipelineState",
    "FittedComponents",
    # Encoders
    "AutoLabelEncoder",
    "AutoOneHotEncoder",
    "AutoIFrequencyEncoder",
    # Scalers
    "AutoMinMaxScaler",
    "AutoStandardScaler",
    "AutoRobustScaler",
    # Imputers
    "AutoSimpleImputer",
    "AutoKNNImputer",
    "AutoIterativeImputer",
    # Registries
    "EncoderRegistry",
    "ScalerRegistry",
    "ImputerRegistry",
    "FeatureSelectorRegistry",
    "MetricRegistry",
    # Feature Selection
    "H2OFeatureSelector",
    "VIFFeatureSelector",
    # Encoding
    "EncodingVersionFactory",
    "EncodingVersion",
    "EncodingOptimizer",
    # Evaluation
    "metrics_regression",
    "metrics_classification",
    # Optimization
    "DimensionalityStrategy",
    "DimensionalityConfig",
    "ModelEvaluator",
    "OptunaAdapter",
    # Data
    "DatasetGenerator",
    # State
    "StateManager",
    # Mixins
    "DataValidationMixin",
    "ColumnTypeMixin",
    "DateEngineeringMixin",
]
