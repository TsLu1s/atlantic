"""
Atlantic - Core Base Classes Module.

Defines abstract base classes for the preprocessing framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from atlantic.core.enums import TaskType
from atlantic.core.exceptions import NotFittedError, ValidationError
from atlantic.core.mixins import (
    DataValidationMixin,
    ColumnTypeMixin,
    DateEngineeringMixin,
    SchemaValidationMixin,
    TargetTypeMixin
)
from atlantic.core.schemas import PipelineState, FittedComponents

if TYPE_CHECKING:
    from typing import Self


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessing components."""
    
    def __init__(self):
        self._is_fitted: bool = False
        self._columns: Optional[List[str]] = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit the preprocessor to the data.

        Returns:
            Self for method chaining.
        """
        ...
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.
 
        Returns:
            Transformed DataFrame.
        """
        ...
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)
    
    def _check_is_fitted(self) -> None:
        """
        Check if preprocessor has been fitted.
        
        Raises:
            NotFittedError: If not fitted.
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
    
    @property
    def is_fitted(self) -> bool:
        """Whether the preprocessor has been fitted."""
        return self._is_fitted


class BasePipeline(ABC):
    """Abstract base class for preprocessing pipelines."""
    
    # Include mixin attributes directly
    NUMERIC_TYPES: List[str] = [
        'int', 'int8', 'int16', 'int32', 'int64',
        'float', 'float16', 'float32', 'float64'
    ]
    CATEGORICAL_TYPES: List[str] = ['object', 'category']
    
    def __init__(self, X: pd.DataFrame, target: str):
        """
        Initialize the pipeline.
        
        Args:
            X: Input DataFrame.
            target: Name of target column.
        """
        self.X = X
        self.target = target
        self._is_fitted: bool = False
        self._state = PipelineState()
        self._fitted_components = FittedComponents()
        self._input_schema: dict = None
        
        # Initialize target type attributes
        self.pred_type: str = None
        self.n_classes: int = None
        self.eval_metric: str = None
        
        # Detect target type
        self._detect_target_type(X)
        self._state.target = target
        self._state.task_type = "regression" if self.pred_type == "Reg" else "classification"
        self._state.n_classes = self.n_classes
        self._state.evaluation_metric = self.eval_metric
    
    # --- Validation Methods ---
    
    def _validate_dataframe(self, X: pd.DataFrame, require_target: bool = True) -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        if X.empty:
            raise ValidationError("Input DataFrame is empty")
        if require_target and self.target and self.target not in X.columns:
            raise ValidationError(f"Target column '{self.target}' not found")
    
    def _validate_split_ratio(self, ratio: float) -> None:
        """Validate split ratio value."""
        if not 0.5 <= ratio <= 0.98:
            raise ValidationError(f"split_ratio must be between 0.5 and 0.98, got {ratio}")
    
    # --- Schema Methods ---
    
    def _store_input_schema(self, X: pd.DataFrame) -> None:
        """Store input schema during fit."""
        self._input_schema = {
            'columns': list(X.columns),
            'dtypes': {col: str(dtype) for col, dtype in X.dtypes.items()},
            'n_features': len(X.columns)
        }
    
    # --- Target Type Detection ---
    
    def _detect_target_type(self, X: pd.DataFrame) -> tuple:
        """Detect target variable type and set appropriate metric."""
        if self.target not in X.columns:
            raise ValidationError(f"Target column '{self.target}' not found")
        
        target_dtype = str(X[self.target].dtype)
        
        if any(t in target_dtype for t in self.NUMERIC_TYPES):
            self.pred_type = 'Reg'
            self.eval_metric = 'Mean Absolute Error'
            self.n_classes = None
        else:
            self.pred_type = 'Class'
            self.n_classes = X[self.target].nunique()
            self.eval_metric = 'F1' if self.n_classes > 2 else 'Precision'
        
        return self.pred_type, self.eval_metric
    
    # --- Column Type Methods ---
    
    def _get_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of numeric columns, excluding target."""
        return [
            col for col in X.select_dtypes(include=self.NUMERIC_TYPES).columns
            if col != self.target
        ]
    
    def _get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of categorical columns, excluding target."""
        return [
            col for col in X.select_dtypes(include=self.CATEGORICAL_TYPES).columns
            if col != self.target
        ]
    
    def _get_datetime_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of datetime columns."""
        return X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    # --- Datetime Engineering ---
    
    def _engineer_datetime_features(self, X: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """Extract temporal features from datetime columns."""
        X = X.copy()
        datetime_cols = self._get_datetime_columns(X)
        
        for col in datetime_cols:
            X[col] = pd.to_datetime(X[col])
            X[f'{col}_day_of_month'] = X[col].dt.day
            X[f'{col}_day_of_week'] = X[col].dt.dayofweek + 1
            X[f'{col}_is_wknd'] = X[f'{col}_day_of_week'].isin([6, 7]).astype(int)
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day_of_year'] = X[col].dt.dayofyear
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_hour'] = X[col].dt.hour
            X[f'{col}_minute'] = X[col].dt.minute
            X[f'{col}_second'] = X[col].dt.second
            
            if drop:
                X = X.drop(columns=[col])
        
        return X
    
    def fit_processing(
        self,
        split_ratio: float = 0.75,
        relevance: float = 0.99,
        h2o_fs_models: int = 7,
        encoding_fs: bool = True,
        vif_ratio: float = 10.0,
        optimization_level: str = "balanced"
    ) -> "Self":
        """
        Template method for fitting preprocessing pipeline.
        
        Args:
            split_ratio: Train/test split ratio.
            relevance: Feature importance threshold for H2O selection.
            h2o_fs_models: Number of H2O models for feature selection.
            encoding_fs: Whether to encode before feature selection.
            vif_ratio: VIF threshold for multicollinearity filtering.
            optimization_level: Optimization level for processing.
        
        Returns:
            Self for method chaining.
        """
        # Validate inputs
        self._validate_dataframe(self.X)
        self._validate_split_ratio(split_ratio)
        
        # Store schema
        self._store_input_schema(self.X)
        
        # Prepare data
        X_ = self._prepare_data(self.X)
        
        # Split data
        train, test = self._split_data(X_, split_ratio)
        
        # Feature selection
        if relevance < 1.0:
            train, test = self._select_features(
                train, test, relevance, h2o_fs_models, encoding_fs
            )
        
        # Encoding optimization
        train, test = self._optimize_encoding(train, test, optimization_level)
        
        # Imputation optimization
        train, test = self._optimize_imputation(train, test, optimization_level)
        
        # VIF filtering
        train, test = self._apply_vif_filtering(train, test, vif_ratio)
        
        # Fit final processors
        self._fit_final_processors(self.X)
        
        self._is_fitted = True
        self._state.is_fitted = True
        
        return self
    
    def data_processing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to new data.

        Returns:
            Processed DataFrame.
        """
        self._check_is_fitted()
        return self._transform_data(X)
    
    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with initial transformations.

        Returns:
            Prepared DataFrame.
        """
        X_ = X.copy()
        
        # Convert target to string for classification
        if self.pred_type == 'Class':
            X_[self.target] = X_[self.target].astype(str)
        
        # Engineer datetime features
        X_ = self._engineer_datetime_features(X_, drop=True)
        
        # Remove high-null columns
        X_ = self._remove_high_null_columns(X_, threshold=0.999)
        
        # Remove constant and unique columns
        X_ = self._remove_uninformative_columns(X_)
        
        # Ensure target is last column
        cols = [col for col in X_.columns if col != self.target] + [self.target]
        X_ = X_[cols]
        
        return X_
    
    def _split_data(
        self, 
        X: pd.DataFrame, 
        split_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Returns:
            Tuple of (train, test) DataFrames.
        """
        X = X.dropna(subset=[self.target])
        
        if self.pred_type == 'Class':
            train, test = train_test_split(
                X, 
                train_size=split_ratio,
                stratify=X[self.target],
                random_state=42
            )
        else:
            train, test = train_test_split(
                X,
                train_size=split_ratio,
                random_state=42
            )
        
        return train.reset_index(drop=True), test.reset_index(drop=True)
    
    @abstractmethod
    def _select_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        relevance: float,
        h2o_fs_models: int,
        encoding_fs: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select features using H2O AutoML."""
        ...
    
    @abstractmethod
    def _optimize_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize encoding version selection."""
        ...
    
    @abstractmethod
    def _optimize_imputation(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize imputation method selection."""
        ...
    
    @abstractmethod
    def _apply_vif_filtering(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        vif_threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply VIF-based feature filtering."""
        ...
    
    @abstractmethod
    def _fit_final_processors(self, X: pd.DataFrame) -> None:
        """Fit final preprocessing components."""
        ...
    
    @abstractmethod
    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted components."""
        ...
    
    def _remove_high_null_columns(
        self, 
        X: pd.DataFrame, 
        threshold: float = 0.999
    ) -> pd.DataFrame:
        """
        Remove columns with null percentage above threshold.

        Returns:
            DataFrame with high-null columns removed.
        """
        min_count = int((1 - threshold) * len(X))
        return X.dropna(axis=1, thresh=min_count)
    
    def _remove_uninformative_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant and all-unique columns.

        Returns:
            DataFrame with uninformative columns removed.
        """
        cols_to_drop = []
        for col in X.columns:
            if col == self.target:
                continue
            n_unique = X[col].nunique()
            if n_unique == 1 or n_unique == len(X):
                cols_to_drop.append(col)
        
        return X.drop(columns=cols_to_drop)
    
    def _check_is_fitted(self) -> None:
        """Check if pipeline has been fitted."""
        if not self._is_fitted:
            raise NotFittedError("Pipeline")
    
    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has been fitted."""
        return self._is_fitted
    
    @property
    def state(self) -> PipelineState:
        """Get pipeline state."""
        return self._state


class BuilderPipeline:
    """Mixin for builder pattern pipeline construction."""
    
    def __init__(self):
        self._builder_config: Dict[str, Any] = {}
    
    def with_date_engineering(
        self, 
        enabled: bool = True, 
        drop: bool = True
    ) -> "Self":
        """Configure datetime feature engineering."""
        self._builder_config['date_engineering'] = enabled
        self._builder_config['date_drop'] = drop
        return self
    
    def with_null_removal(self, threshold: float = 0.99) -> "Self":
        """Configure null column removal threshold."""
        self._builder_config['null_removal_threshold'] = threshold
        return self
    
    def with_feature_selection(
        self,
        method: str = "h2o",
        relevance: float = 0.99,
        h2o_models: int = 7,
        encoding_fs: bool = True
    ) -> "Self":
        """Configure feature selection."""
        self._builder_config['feature_selection_method'] = method
        self._builder_config['h2o_relevance'] = relevance
        self._builder_config['h2o_models'] = h2o_models
        self._builder_config['encoding_fs'] = encoding_fs
        return self
    
    def with_encoding(
        self,
        scaler: str = "standard",
        encoder: str = "ifrequency",
        auto_select: bool = False
    ) -> "Self":
        """Configure encoding strategy."""
        self._builder_config['scaler'] = scaler
        self._builder_config['encoder'] = encoder
        self._builder_config['auto_encoding'] = auto_select
        return self
    
    def with_imputation(
        self,
        method: str = "simple",
        auto_select: bool = False,
        **params
    ) -> "Self":
        """Configure imputation strategy."""
        self._builder_config['imputer'] = method
        self._builder_config['auto_imputation'] = auto_select
        self._builder_config['imputer_params'] = params
        return self
    
    def with_vif_filtering(self, threshold: float = 10.0) -> "Self":
        """Configure VIF-based feature filtering."""
        self._builder_config['vif_threshold'] = threshold
        return self
    
    def with_optimization(
        self,
        optimization_level: str = "balanced",
        random_state: int = 42
    ) -> "Self":
        """Configure optimization settings."""
        self._builder_config['optimization_level'] = optimization_level
        self._builder_config['random_state'] = random_state
        return self
