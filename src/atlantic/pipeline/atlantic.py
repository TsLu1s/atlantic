"""
Atlantic - Main Pipeline Module.

Implements the main Atlantic preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Self, TYPE_CHECKING

import pandas as pd

from atlantic.core.base import BasePipeline, BuilderPipeline
from atlantic.core.enums import TaskType
from atlantic.core.schemas import (
    PipelineConfig,
    PipelineState,
    FittedComponents,
    FeatureImportanceResult
)
from atlantic.core.exceptions import NotFittedError
from atlantic.core.mixins import DateEngineeringMixin
from atlantic.preprocessing.encoders import AutoLabelEncoder, AutoIFrequencyEncoder
from atlantic.preprocessing.scalers import AutoMinMaxScaler, AutoStandardScaler
from atlantic.preprocessing.imputers import (
    AutoSimpleImputer,
    AutoKNNImputer,
    AutoIterativeImputer
)
from atlantic.preprocessing.registry import EncoderRegistry, ScalerRegistry, ImputerRegistry
from atlantic.feature_selection.h2o_selector import H2OFeatureSelector
from atlantic.feature_selection.vif_selector import VIFFeatureSelector
from atlantic.encoding.versions import EncodingVersion
from atlantic.pipeline.pattern import Pattern
from atlantic.utils.columns import get_numeric_columns, get_categorical_columns
from atlantic.utils.datetime import engineer_datetime_features


class Atlantic(BasePipeline, BuilderPipeline):
    """
    Atlantic - Automated Data Preprocessing Framework.
    
    Provides comprehensive data preprocessing capabilities including
    feature engineering, feature selection, encoding, imputation,
    and transformation for supervised machine learning.
    """
    
    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        target: Optional[str] = None
    ):
        """
        Initialize Atlantic pipeline.
        
        Args:
            X: Input DataFrame to process.
            target: Name of target column.
        """
        if X is not None and target is not None:
            BasePipeline.__init__(self, X, target)
        else:
            self.X = X
            self.target = target
            self._is_fitted = False
            self._state = PipelineState()
            self._fitted_components = FittedComponents()
        
        BuilderPipeline.__init__(self)
        
        # Preprocessing components
        self.enc_method: Optional[str] = None
        self.imp_method: Optional[str] = None
        self.encoder = None
        self.scaler = None
        self.imputer = None
        
        # Column tracking
        self.n_cols: List[str] = []
        self.c_cols: List[str] = []
        self.cols: List[str] = []
        
        # Feature importance
        self.h2o_feature_importance: Optional[pd.DataFrame] = None
    
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
        Fit the preprocessing pipeline.

        Returns:
            Self for method chaining.
        """
        # Validate inputs
        self._validate_dataframe(self.X)
        self._validate_split_ratio(split_ratio)
        
        # Store configuration
        self._state.config = PipelineConfig(
            split_ratio=split_ratio,
            relevance=relevance,
            h2o_fs_models=h2o_fs_models,
            encoding_fs=encoding_fs,
            vif_threshold=vif_ratio,
            optimization_level=optimization_level
        )
        
        # Data preparation
        if self.pred_type == 'Class':
            self.X[self.target] = self.X[self.target].astype(str)
        
        X_ = self.X.copy()
        
        # Date engineering
        X_ = engineer_datetime_features(X_, drop_original=True)
        
        # Remove high-null columns
        X_ = self._remove_high_null_columns(X_, threshold=0.999)
        
        # Remove uninformative columns
        X_ = self._remove_uninformative_columns(X_)
        
        # Ensure target is last
        cols = [col for col in X_.columns if col != self.target] + [self.target]
        X_ = X_[cols]
        
        data = X_.copy()
        
        # Split data
        train, test = self._split_data(X_, split_ratio)
        
        # Feature selection with H2O
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
        self._fit_final_processors(data)
        
        self._is_fitted = True
        self._state.is_fitted = True
        
        return self
    
    def _select_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        relevance: float,
        h2o_fs_models: int,
        encoding_fs: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select features using H2O AutoML."""
        selector = H2OFeatureSelector(
            target=self.target,
            relevance=relevance,
            max_models=h2o_fs_models,
            encoding_before_selection=encoding_fs
        )
        
        selector.fit(train)
        
        sel_cols = selector.selected_features + [self.target]
        self.h2o_feature_importance = selector.feature_importance
        
        return train[sel_cols], test[sel_cols]
    
    def _optimize_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize encoding version selection."""
        ptn = Pattern(
            train=train,
            test=test,
            target=self.target,
            optimization_level=optimization_level,
            verbosity=1
        )
        
        self.enc_method = ptn.encoding_selection()
        self._perf = ptn.perf
        
        return train, test
    
    def _optimize_imputation(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize imputation method selection."""
        # Check if imputation is needed
        has_nulls = (
            train.isnull().sum().sum() > 0 or 
            test.isnull().sum().sum() > 0
        )
        
        if not has_nulls:
            self.imp_method = 'Undefined'
            
            # Apply encoding without imputation optimization
            ev = EncodingVersion(train=train, test=test, target=self.target)
            
            version_map = {
                'Encoding Version 1': ev.encoding_v1,
                'Encoding Version 2': ev.encoding_v2,
                'Encoding Version 3': ev.encoding_v3,
                'Encoding Version 4': ev.encoding_v4
            }
            
            if self.enc_method in version_map:
                train, test = version_map[self.enc_method]()
            
            print('There are no missing values in the Dataset')
            return train, test
        
        # Run imputation optimization
        ptn = Pattern(
            train=train,
            test=test,
            target=self.target,
            optimization_level=optimization_level,
            verbosity=1
        )
        ptn.enc_method = self.enc_method
        
        train, test = ptn.imputation_selection()
        self.imp_method = ptn.imp_method
        self._perf = ptn.perf
        
        return train, test
    
    def _apply_vif_filtering(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        vif_threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply VIF-based feature filtering."""
        ptn = Pattern(
            train=train,
            test=test,
            target=self.target,
            verbosity=1
        )
        ptn.perf = getattr(self, '_perf', None)
        
        train, test = ptn.vif_performance(
            vif_threshold=vif_threshold,
            perf_=getattr(self, '_perf', None)
        )
        
        self.cols = [col for col in train.columns if col != self.target]
        
        return train, test
    
    def _fit_final_processors(self, X: pd.DataFrame) -> None:
        """Fit final preprocessing components on full data."""
        # Get data with selected columns
        data = X[self.cols].copy()
        
        self.n_cols = get_numeric_columns(data)
        self.c_cols = get_categorical_columns(data)
        
        # Update fitted components
        self._fitted_components.selected_columns = self.cols
        self._fitted_components.numerical_columns = self.n_cols
        self._fitted_components.categorical_columns = self.c_cols
        self._fitted_components.encoding_version = self.enc_method
        self._fitted_components.imputation_method = self.imp_method
        
        # Fit scaler on numerical columns
        if self.n_cols:
            if self.enc_method in ['Encoding Version 1', 'Encoding Version 3']:
                self.scaler = AutoStandardScaler()
            else:
                self.scaler = AutoMinMaxScaler()
            
            self.scaler.fit(data[self.n_cols])
            data[self.n_cols] = self.scaler.transform(data[self.n_cols])
        
        # Fit encoder on categorical columns
        if self.c_cols:
            if self.enc_method in ['Encoding Version 1', 'Encoding Version 2']:
                self.encoder = AutoIFrequencyEncoder()
            else:
                self.encoder = AutoLabelEncoder()
            
            self.encoder.fit(data[self.c_cols])
            data = self.encoder.transform(data)
        
        # Fit imputer
        if self.imp_method == 'Simple':
            self.imputer = AutoSimpleImputer(strategy='mean')
            self.imputer.fit(data)
        elif self.imp_method == 'KNN':
            self.imputer = AutoKNNImputer(n_neighbors=3, weights="uniform")
            self.imputer.fit(data)
        elif self.imp_method == 'Iterative':
            self.imputer = AutoIterativeImputer(
                max_iter=10,
                random_state=42,
                initial_strategy="mean",
                imputation_order="ascending"
            )
            self.imputer.fit(data)
        
        # Store fitted components
        self._fitted_components.encoder = self.encoder
        self._fitted_components.scaler = self.scaler
        self._fitted_components.imputer = self.imputer
    
    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted components."""
        return self.data_processing(X)
    
    def data_processing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to new data.

        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise NotFittedError("Atlantic")
        
        data = X.copy()
        
        # Date engineering
        data = engineer_datetime_features(data, drop_original=True)
        
        # Select columns
        if self.target in data.columns:
            data = data[self.cols + [self.target]]
        else:
            data = data[self.cols]
        
        # Apply scaler
        if self.n_cols and self.scaler is not None:
            data[self.n_cols] = self.scaler.transform(data[self.n_cols])
        
        # Apply encoder
        if self.c_cols and self.encoder is not None:
            data = self.encoder.transform(data)
        
        # Apply imputer
        if (self.imp_method != "Undefined" and 
            self.imputer is not None and
            data[self.c_cols + self.n_cols].isnull().sum().sum() > 0):
            data = self.imputer.transform(data.copy())
        
        return data
    
    @classmethod
    def builder(cls) -> "AtlanticBuilder":
        """
        Get builder for granular pipeline construction.
        
        Returns:
            AtlanticBuilder instance.
        """
        from atlantic.pipeline.builder import AtlanticBuilder
        return AtlanticBuilder()
    
    @property
    def fitted_components(self) -> FittedComponents:
        """Get fitted preprocessing components."""
        return self._fitted_components
    
    def save(self, path: str) -> None:
        """
        Save fitted pipeline to file.
        
        Args:
            path: File path for saving (pickle format).
        """
        from atlantic.state.pipeline_state import StateManager
        StateManager.save_full_state(self, path)
    
    @classmethod
    def load(cls, path: str) -> "Atlantic":
        """
        Load fitted pipeline from file.

        Returns:
            Loaded Atlantic instance.
        """
        from atlantic.state.pipeline_state import StateManager
        return StateManager.load_full_state(path)