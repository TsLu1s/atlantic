from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Self

import pandas as pd

from atlantic.core.schemas import (
    BuilderConfig,
    EncodingConfig,
    FeatureSelectionConfig,
    ImputationConfig,
    OptimizerConfig,
)


class AtlanticBuilder:
    """
    Fluent builder for granular Atlantic pipeline construction.

    Allows step-by-step configuration of preprocessing components
    with full customization control.
    """

    def __init__(self):
        """Initialize builder with default configuration."""
        self._date_engineering: bool = True
        self._date_drop: bool = True
        self._null_removal_threshold: float = 0.99

        # Feature selection
        self._feature_selection_method: Optional[str] = "h2o"
        self._h2o_relevance: float = 0.99
        self._h2o_models: int = 7
        self._encoding_fs: bool = True
        self._vif_threshold: Optional[float] = 10.0

        # Encoding
        self._scaler: Optional[str] = None
        self._encoder: Optional[str] = None
        self._auto_encoding: bool = True

        # Imputation
        self._imputer: Optional[str] = None
        self._imputer_params: Dict[str, Any] = {}
        self._auto_imputation: bool = True

        # Optimization
        self._optimization_level: str = "balanced"
        self._random_state: int = 42

    def with_date_engineering(self, enabled: bool = True, drop: bool = True) -> "Self":
        """
        Configure datetime feature engineering.

        Returns:
            Self for method chaining.
        """
        self._date_engineering = enabled
        self._date_drop = drop
        return self

    def with_null_removal(self, threshold: float = 0.99) -> "Self":
        """
        Configure null column removal threshold.

        Returns:
            Self for method chaining.
        """
        self._null_removal_threshold = threshold
        return self

    def with_feature_selection(
        self,
        method: Literal["h2o", "none"] = "h2o",
        relevance: float = 0.99,
        h2o_models: int = 7,
        encoding_fs: bool = True,
    ) -> "Self":
        """
        Configure feature selection.

        Returns:
            Self for method chaining.
        """
        self._feature_selection_method = method if method != "none" else None
        self._h2o_relevance = relevance
        self._h2o_models = h2o_models
        self._encoding_fs = encoding_fs
        return self

    def with_encoding(
        self,
        scaler: Literal["standard", "minmax", "robust"] = "standard",
        encoder: Literal["label", "ifrequency", "onehot"] = "ifrequency",
        auto_select: bool = False,
    ) -> "Self":
        """
        Configure encoding strategy.

        Returns:
            Self for method chaining.
        """
        self._scaler = scaler
        self._encoder = encoder
        self._auto_encoding = auto_select
        return self

    def with_imputation(
        self,
        method: Literal["simple", "knn", "iterative"] = "simple",
        auto_select: bool = False,
        **params,
    ) -> "Self":
        """
        Configure imputation strategy.

        Returns:
            Self for method chaining.
        """
        self._imputer = method
        self._auto_imputation = auto_select
        self._imputer_params = params
        return self

    def with_vif_filtering(self, threshold: float = 10.0) -> "Self":
        """
        Configure VIF-based feature filtering.

        Returns:
            Self for method chaining.
        """
        self._vif_threshold = threshold
        return self

    def with_optimization(
        self,
        optimization_level: Literal["fast", "balanced", "thorough"] = "balanced",
        random_state: int = 42,
    ) -> "Self":
        """
        Configure optimization settings.

        Returns:
            Self for method chaining.
        """
        self._optimization_level = optimization_level
        self._random_state = random_state
        return self

    def _create_config(self) -> BuilderConfig:
        """Create configuration from builder state."""
        return BuilderConfig(
            date_engineering=self._date_engineering,
            date_drop=self._date_drop,
            null_removal_threshold=self._null_removal_threshold,
            feature_selection=FeatureSelectionConfig(
                method=self._feature_selection_method or "none",
                h2o_relevance=self._h2o_relevance,
                h2o_max_models=self._h2o_models,
                encoding_for_fs=self._encoding_fs,
                vif_threshold=self._vif_threshold or 10.0,
            ),
            encoding=EncodingConfig(
                scaler=self._scaler or "standard",
                encoder=self._encoder or "ifrequency",
                auto_select=self._auto_encoding,
            ),
            imputation=ImputationConfig(
                method=self._imputer or "simple",
                auto_select=self._auto_imputation,
                **self._imputer_params,
            ),
            optimizer=OptimizerConfig(
                optimization_level=self._optimization_level, random_state=self._random_state
            ),
        )

    def build(self) -> "AtlanticPipeline":
        """
        Build the configured pipeline.

        Returns:
            Configured AtlanticPipeline instance.
        """
        config = self._create_config()
        return AtlanticPipeline(config)

    def get_config(self) -> BuilderConfig:
        """
        Get current configuration without building.

        Returns:
            Current BuilderConfig.
        """
        return self._create_config()


class AtlanticPipeline:
    """
    Pipeline built from AtlanticBuilder configuration.

    Provides fit_transform and transform methods for preprocessing.
    """

    def __init__(self, config: BuilderConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Builder configuration.
        """
        self._config = config
        self._is_fitted = False
        self._atlantic = None

    def fit_transform(self, X: "pd.DataFrame", target: str) -> "pd.DataFrame":
        """
        Fit pipeline and transform data.

        Returns:
            Transformed DataFrame.
        """
        from atlantic.pipeline.atlantic import Atlantic

        # Create Atlantic instance
        self._atlantic = Atlantic(X=X, target=target)

        # Determine parameters from config
        relevance = (
            self._config.feature_selection.h2o_relevance
            if self._config.feature_selection.method == "h2o"
            else 1.0
        )

        # Fit processing
        self._atlantic.fit_processing(
            split_ratio=0.75,
            relevance=relevance,
            h2o_fs_models=self._config.feature_selection.h2o_max_models,
            encoding_fs=self._config.feature_selection.encoding_for_fs,
            vif_ratio=self._config.feature_selection.vif_threshold,
            optimization_level=self._config.optimizer.optimization_level,
        )

        self._is_fitted = True

        return self._atlantic.data_processing(X)

    def transform(self, X: "pd.DataFrame") -> "pd.DataFrame":
        """
        Transform data using fitted pipeline.

        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted or self._atlantic is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")

        return self._atlantic.data_processing(X)

    @property
    def is_fitted(self) -> bool:
        """Whether pipeline has been fitted."""
        return self._is_fitted

    @property
    def config(self) -> BuilderConfig:
        """Get pipeline configuration."""
        return self._config
