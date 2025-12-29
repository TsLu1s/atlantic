from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class PipelineConfig(BaseModel):
    """Configuration for Atlantic pipeline."""
    
    split_ratio: float = Field(default=0.75, ge=0.5, le=0.98)
    relevance: float = Field(default=0.99, ge=0.4, le=1.0)
    h2o_fs_models: int = Field(default=7, ge=1, le=100)
    encoding_fs: bool = Field(default=True)
    vif_threshold: float = Field(default=10.0, ge=3.0, le=30.0)
    optimization_level: Literal["fast", "balanced", "thorough"] = Field(default="balanced")
    random_state: int = Field(default=42)
    
    @field_validator('split_ratio')
    @classmethod
    def validate_split_ratio(cls, v: float) -> float:
        if not 0.5 <= v <= 0.98:
            raise ValueError("split_ratio must be between 0.5 and 0.98")
        return v
    
    @field_validator('relevance')
    @classmethod
    def validate_relevance(cls, v: float) -> float:
        if not 0.4 <= v <= 1.0:
            raise ValueError("relevance must be between 0.4 and 1.0")
        return v


class FeatureSelectionConfig(BaseModel):
    """Configuration for feature selection."""
    
    method: Literal["h2o", "vif", "both", "none"] = Field(default="both")
    h2o_relevance: float = Field(default=0.99, ge=0.4, le=1.0)
    h2o_max_models: int = Field(default=7, ge=1, le=100)
    h2o_excluded_algorithms: List[str] = Field(
        default_factory=lambda: ["GLM", "DeepLearning", "StackedEnsemble"]
    )
    vif_threshold: float = Field(default=10.0, ge=3.0, le=30.0)
    encoding_for_fs: bool = Field(default=True)


class EncodingConfig(BaseModel):
    """Configuration for encoding."""
    
    scaler: Literal["standard", "minmax", "robust"] = Field(default="standard")
    encoder: Literal["label", "ifrequency", "onehot"] = Field(default="ifrequency")
    auto_select: bool = Field(default=True)


class ImputationConfig(BaseModel):
    """Configuration for imputation."""
    
    method: Literal["simple", "knn", "iterative"] = Field(default="simple")
    simple_strategy: Literal["mean", "median", "most_frequent", "constant"] = Field(default="mean")
    knn_neighbors: int = Field(default=5, ge=1, le=20)
    iterative_max_iter: int = Field(default=10, ge=1, le=100)
    iterative_random_state: Optional[int] = Field(default=42)
    auto_select: bool = Field(default=True)


class OptimizerConfig(BaseModel):
    """Configuration for hyperparameter optimization."""
    
    optimization_level: Literal["fast", "balanced", "thorough"] = Field(default="balanced")
    random_state: int = Field(default=42)
    n_folds: int = Field(default=3, ge=2, le=10)
    verbosity: int = Field(default=0, ge=0, le=2)


class BuilderConfig(BaseModel):
    """Configuration built from AtlanticBuilder."""
    
    date_engineering: bool = Field(default=True)
    date_drop: bool = Field(default=True)
    null_removal_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


@dataclass
class FittedComponents:
    """Container for fitted preprocessing components."""
    
    encoder: Optional[Any] = None
    scaler: Optional[Any] = None
    imputer: Optional[Any] = None
    selected_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    encoding_version: Optional[str] = None
    imputation_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "selected_columns": self.selected_columns,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "encoding_version": self.encoding_version,
            "imputation_method": self.imputation_method
        }


@dataclass
class PipelineState:
    """Complete state of fitted Atlantic pipeline for serialization."""
    
    config: Optional[PipelineConfig] = None
    fitted_components: FittedComponents = field(default_factory=FittedComponents)
    feature_importance: Optional[pd.DataFrame] = None
    task_type: Literal["regression", "classification"] = "regression"
    n_classes: Optional[int] = None
    evaluation_metric: str = "Mean Absolute Error"
    is_fitted: bool = False
    target: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "config": self.config.model_dump() if self.config else None,
            "fitted_components": self.fitted_components.to_dict(),
            "feature_importance": (
                self.feature_importance.to_dict() 
                if self.feature_importance is not None else None
            ),
            "task_type": self.task_type,
            "n_classes": self.n_classes,
            "evaluation_metric": self.evaluation_metric,
            "is_fitted": self.is_fitted,
            "target": self.target
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Deserialize state from dictionary."""
        config = PipelineConfig(**data["config"]) if data.get("config") else None
        
        fitted_components = FittedComponents(
            selected_columns=data["fitted_components"].get("selected_columns", []),
            numerical_columns=data["fitted_components"].get("numerical_columns", []),
            categorical_columns=data["fitted_components"].get("categorical_columns", []),
            encoding_version=data["fitted_components"].get("encoding_version"),
            imputation_method=data["fitted_components"].get("imputation_method")
        )
        
        feature_importance = (
            pd.DataFrame(data["feature_importance"]) 
            if data.get("feature_importance") else None
        )
        
        return cls(
            config=config,
            fitted_components=fitted_components,
            feature_importance=feature_importance,
            task_type=data.get("task_type", "regression"),
            n_classes=data.get("n_classes"),
            evaluation_metric=data.get("evaluation_metric", "Mean Absolute Error"),
            is_fitted=data.get("is_fitted", False),
            target=data.get("target")
        )
    
    def save(self, path: str) -> None:
        """Save state to JSON file (without fitted objects)."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PipelineState":
        """Load state from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class EvaluationResult:
    """Result from model evaluation."""
    
    metrics: pd.DataFrame
    best_params: Dict[str, Any]
    best_model: str
    iteration: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics.to_dict(),
            "best_params": self.best_params,
            "best_model": self.best_model,
            "iteration": self.iteration
        }


@dataclass
class FeatureImportanceResult:
    """Result from feature importance analysis."""
    
    importance_df: pd.DataFrame
    selected_features: List[str]
    threshold_used: float
    
    @property
    def n_selected(self) -> int:
        """Number of selected features."""
        return len(self.selected_features)


@dataclass 
class EncodingSelectionResult:
    """Result from encoding version selection."""
    
    selected_version: str
    performance_scores: Dict[str, float]
    best_score: float
    metric_name: str


@dataclass
class ImputationSelectionResult:
    """Result from imputation method selection."""
    
    selected_method: str
    performance_scores: Dict[str, float]
    best_score: float
    metric_name: str