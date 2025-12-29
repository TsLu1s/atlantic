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

# Main pipeline
from atlantic.pipeline.atlantic import Atlantic
from atlantic.pipeline.builder import AtlanticBuilder, AtlanticPipeline
from atlantic.pipeline.pattern import Pattern

# Core
from atlantic.core.enums import (
    TaskType,
    DimensionalityTier,
    OptimizationLevel,
    EncodingVersion,
    ScalerType,
    EncoderType,
    ImputerType,
    FeatureSelectorType
)
from atlantic.core.exceptions import (
    AtlanticError,
    NotFittedError,
    ValidationError,
    FeatureSelectionError,
    EncodingError,
    ImputationError
)
from atlantic.core.schemas import (
    PipelineConfig,
    PipelineState,
    FittedComponents
)

# Preprocessing
from atlantic.preprocessing.encoders import (
    AutoLabelEncoder,
    AutoOneHotEncoder,
    AutoIFrequencyEncoder
)
from atlantic.preprocessing.scalers import (
    AutoMinMaxScaler,
    AutoStandardScaler,
    AutoRobustScaler
)
from atlantic.preprocessing.imputers import (
    AutoSimpleImputer,
    AutoKNNImputer,
    AutoIterativeImputer
)
from atlantic.preprocessing.registry import (
    EncoderRegistry,
    ScalerRegistry,
    ImputerRegistry
)

# Feature Selection
from atlantic.feature_selection.h2o_selector import H2OFeatureSelector
from atlantic.feature_selection.vif_selector import VIFFeatureSelector
from atlantic.feature_selection.registry import FeatureSelectorRegistry

# Encoding
from atlantic.encoding.versions import EncodingVersionFactory, EncodingVersion
from atlantic.encoding.optimizer import EncodingOptimizer

# Evaluation
from atlantic.evaluation.metrics import (
    MetricRegistry,
    metrics_regression,
    metrics_classification
)

# Optimization
from atlantic.optimization.dimensionality import (
    DimensionalityStrategy,
    DimensionalityConfig
)
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.optimization.optuna_adapter import OptunaAdapter

# Data
from atlantic.data.generator import DatasetGenerator

# State
from atlantic.state.pipeline_state import StateManager

# Analysis (backward compatibility)
from atlantic.core.mixins import (
    DataValidationMixin,
    ColumnTypeMixin,
    DateEngineeringMixin
)


__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Main pipeline
    'Atlantic',
    'AtlanticBuilder',
    'AtlanticPipeline',
    'Pattern',
    
    # Enums
    'TaskType',
    'DimensionalityTier',
    'OptimizationLevel',
    'EncodingVersion',
    'ScalerType',
    'EncoderType',
    'ImputerType',
    'FeatureSelectorType',
    
    # Exceptions
    'AtlanticError',
    'NotFittedError',
    'ValidationError',
    'FeatureSelectionError',
    'EncodingError',
    'ImputationError',
    
    # Schemas
    'PipelineConfig',
    'PipelineState',
    'FittedComponents',
    
    # Encoders
    'AutoLabelEncoder',
    'AutoOneHotEncoder',
    'AutoIFrequencyEncoder',
    
    # Scalers
    'AutoMinMaxScaler',
    'AutoStandardScaler',
    'AutoRobustScaler',
    
    # Imputers
    'AutoSimpleImputer',
    'AutoKNNImputer',
    'AutoIterativeImputer',
    
    # Registries
    'EncoderRegistry',
    'ScalerRegistry',
    'ImputerRegistry',
    'FeatureSelectorRegistry',
    'MetricRegistry',
    
    # Feature Selection
    'H2OFeatureSelector',
    'VIFFeatureSelector',
    
    # Encoding
    'EncodingVersionFactory',
    'EncodingVersion',
    'EncodingOptimizer',
    
    # Evaluation
    'metrics_regression',
    'metrics_classification',
    
    # Optimization
    'DimensionalityStrategy',
    'DimensionalityConfig',
    'ModelEvaluator',
    'OptunaAdapter',
    
    # Data
    'DatasetGenerator',
    
    # State
    'StateManager',
    
    # Mixins
    'DataValidationMixin',
    'ColumnTypeMixin',
    'DateEngineeringMixin'
]