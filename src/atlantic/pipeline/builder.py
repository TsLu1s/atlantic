"""
Atlantic - Builder Pattern.

Fluent, validated configuration of an Atlantic pipeline and the
``AtlanticPipeline`` object it produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from pydantic import ValidationError as PydanticValidationError

from atlantic.core.exceptions import NotFittedError, StateSerializationError, ValidationError
from atlantic.core.schemas import (
    BuilderConfig,
    EncodingConfig,
    FeatureSelectionConfig,
    ImputationConfig,
    OptimizerConfig,
)

if TYPE_CHECKING:
    from typing import Self

    import pandas as pd

    from atlantic.pipeline.atlantic import Atlantic


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
        self._split_ratio: float = 0.75
        self._n_jobs: Optional[int] = None
        self._verbosity: int = 1

    def with_date_engineering(self, enabled: bool = True, drop: bool = True) -> "Self":
        """
        Configure datetime feature engineering.

        Args:
            enabled: Expand datetime columns into numeric features (day,
                month, year, weekday, ...). When False, datetime columns are
                not used as features at all.
            drop: Drop the original datetime columns from the transformed
                output. When False they are passed through unchanged next to
                the processed features (raw datetimes are never fed to the
                internal models).

        Returns:
            Self for method chaining.
        """
        self._date_engineering = enabled
        self._date_drop = drop
        return self

    def with_null_removal(self, threshold: float = 0.99) -> "Self":
        """
        Configure null column removal threshold.

        Columns whose fraction of missing values is above ``threshold`` are
        removed. Fully-missing columns are always removed.

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

        ``method="h2o"`` requires the optional ``h2o`` dependency
        (``pip install "atlantic[h2o]"``) and a Java runtime;
        ``method="none"`` skips importance-based selection.

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

        With ``auto_select=True`` the best scaler + encoder combination is
        chosen by validation performance and ``scaler`` / ``encoder`` are
        ignored. With ``auto_select=False`` (the default of this method) they
        are applied verbatim; ``encoder="onehot"`` is rejected at fit time
        because one-hot expansion is not supported inside the pipeline.

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

        With ``auto_select=True`` the best imputer is chosen by validation
        performance; otherwise ``method`` is applied verbatim. Extra keyword
        arguments are forwarded to the imputer (e.g. ``n_neighbors=5``).

        Returns:
            Self for method chaining.
        """
        self._imputer = method
        self._auto_imputation = auto_select
        self._imputer_params = params
        return self

    def with_vif_filtering(self, threshold: float = 10.0) -> "Self":
        """
        Configure VIF-based feature filtering (threshold within [3, 30]).

        Returns:
            Self for method chaining.
        """
        self._vif_threshold = threshold
        return self

    def with_optimization(
        self,
        optimization_level: Literal["fast", "balanced", "thorough"] = "balanced",
        random_state: int = 42,
        split_ratio: float = 0.75,
        n_jobs: Optional[int] = None,
        verbosity: int = 1,
    ) -> "Self":
        """
        Configure optimization settings.

        Args:
            optimization_level: Optimization intensity for internal evaluation.
            random_state: Seed for the train/test split and models.
            split_ratio: Train fraction of the internal preprocessing split.
            n_jobs: Parallelism of the evaluation models (``None`` keeps the
                forests single-threaded, ``-1`` uses all cores).
            verbosity: ``0`` silent, ``1`` selection summary, ``2`` detailed.

        Returns:
            Self for method chaining.
        """
        self._optimization_level = optimization_level
        self._random_state = random_state
        self._split_ratio = split_ratio
        self._n_jobs = n_jobs
        self._verbosity = verbosity
        return self

    def _create_config(self) -> BuilderConfig:
        """Create configuration from builder state.

        Raises:
            ValidationError: If a configured value is out of range.
        """
        try:
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
                    params=dict(self._imputer_params),
                ),
                optimizer=OptimizerConfig(
                    optimization_level=self._optimization_level,
                    random_state=self._random_state,
                    split_ratio=self._split_ratio,
                    n_jobs=self._n_jobs,
                    verbosity=self._verbosity,
                ),
            )
        except PydanticValidationError as e:
            details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise ValidationError(f"Invalid builder configuration - {details}") from e

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

    Provides ``fit`` / ``fit_transform`` / ``transform`` plus ``save`` /
    ``load`` for the fitted pipeline.
    """

    def __init__(self, config: BuilderConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Builder configuration.
        """
        self._config = config
        self._is_fitted = False
        self._atlantic: Optional["Atlantic"] = None

    def fit(self, X: "pd.DataFrame", target: str) -> "Self":
        """
        Fit the pipeline on ``X`` without transforming it.

        Returns:
            Self for method chaining.
        """
        from atlantic.pipeline.atlantic import Atlantic

        feature_selection = self._config.feature_selection
        enc = self._config.encoding
        imp = self._config.imputation
        opt = self._config.optimizer

        # Create Atlantic instance, propagating the configured seed,
        # parallelism and verbosity.
        self._is_fitted = False
        self._atlantic = Atlantic(
            X=X,
            target=target,
            random_state=opt.random_state,
            n_jobs=getattr(opt, "n_jobs", None),
            verbosity=getattr(opt, "verbosity", 1),
        )

        # H2O importance-based selection only runs for method="h2o".
        relevance = feature_selection.h2o_relevance if feature_selection.method == "h2o" else 1.0

        # Fit processing - the full builder configuration is honoured:
        # date/null/split/encoding/imputation settings all flow through.
        self._atlantic.fit_processing(
            split_ratio=opt.split_ratio,
            relevance=relevance,
            h2o_fs_models=feature_selection.h2o_max_models,
            encoding_fs=feature_selection.encoding_for_fs,
            vif_ratio=feature_selection.vif_threshold,
            optimization_level=opt.optimization_level,
            null_threshold=self._config.null_removal_threshold,
            date_engineering=self._config.date_engineering,
            date_drop=self._config.date_drop,
            auto_encoding=enc.auto_select,
            scaler=enc.scaler,
            encoder=enc.encoder,
            auto_imputation=imp.auto_select,
            imputer=imp.method,
            imputer_params=dict(getattr(imp, "params", {}) or {}),
        )

        self._is_fitted = True
        return self

    def fit_transform(self, X: "pd.DataFrame", target: str) -> "pd.DataFrame":
        """
        Fit pipeline and transform data.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, target).transform(X)

    def transform(self, X: "pd.DataFrame") -> "pd.DataFrame":
        """
        Transform data using fitted pipeline.

        Returns:
            Transformed DataFrame.

        Raises:
            NotFittedError: If the pipeline has not been fitted (also a
                ``ValueError``, for backwards compatibility).
        """
        if not self._is_fitted or self._atlantic is None:
            raise NotFittedError(
                "AtlanticPipeline", message="Pipeline not fitted. Call fit_transform first."
            )

        return self._atlantic.data_processing(X)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted pipeline (configuration + fitted transformers).

        The training DataFrame is not stored. The file is a pickle - load only
        files you trust.
        """
        if not self._is_fitted or self._atlantic is None:
            raise NotFittedError(
                "AtlanticPipeline", message="Pipeline not fitted. Call fit_transform first."
            )

        from atlantic.state.pipeline_state import StateManager

        StateManager.save_full_state(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AtlanticPipeline":
        """
        Load a pipeline saved with :meth:`save`.

        Only load files from a trusted source: unpickling can execute
        arbitrary code.
        """
        from atlantic.state.pipeline_state import StateManager

        loaded = StateManager.load_full_state(path)
        if not isinstance(loaded, cls):
            raise StateSerializationError(
                f"{path} does not contain an AtlanticPipeline (found {type(loaded).__name__}). "
                "Use Atlantic.load() for pipelines saved with Atlantic.save()."
            )
        return loaded

    @property
    def is_fitted(self) -> bool:
        """Whether pipeline has been fitted."""
        return self._is_fitted

    @property
    def config(self) -> BuilderConfig:
        """Get pipeline configuration."""
        return self._config

    @property
    def atlantic(self) -> Optional["Atlantic"]:
        """The underlying fitted :class:`Atlantic` instance (None before fit).

        Gives access to the fit results: ``enc_method``, ``imp_method``,
        ``cols``, ``dropped_columns``, ``fitted_components`` ...
        """
        return self._atlantic
