"""
Atlantic - Core Base Classes Module.

Defines abstract base classes for the preprocessing framework.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from atlantic.core.exceptions import NotFittedError, ValidationError
from atlantic.core.mixins import (
    ColumnTypeMixin,
    DataValidationMixin,
    DateEngineeringMixin,
    SchemaValidationMixin,
    TargetTypeMixin,
)
from atlantic.core.schemas import FittedComponents, PipelineState
from atlantic.utils.columns import get_boolean_columns, get_datetime_columns
from atlantic.utils.datetime import engineer_datetime_features

if TYPE_CHECKING:
    from typing import Self

# Reasons recorded in ``BasePipeline.dropped_columns``.
DROP_REASONS: Tuple[str, ...] = (
    "datetime",
    "high_null",
    "constant",
    "unique_identifier",
    "feature_selection",
    "multicollinearity",
)


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

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
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


class BasePipeline(
    DataValidationMixin,
    ColumnTypeMixin,
    DateEngineeringMixin,
    SchemaValidationMixin,
    TargetTypeMixin,
    ABC,
):
    """Abstract base class for preprocessing pipelines.

    Validation, column-type detection, datetime engineering, schema capture
    and target-type detection come from the mixins in
    :mod:`atlantic.core.mixins` (one implementation each). This class adds the
    leakage-safe data preparation, the train/test split and the template
    ``fit_processing`` flow.
    """

    DEFAULT_RANDOM_STATE: int = 42

    def __init__(self, X: pd.DataFrame, target: str, random_state: int = 42):
        """
        Initialize the pipeline.

        Args:
            X: Input DataFrame.
            target: Name of target column.
            random_state: Seed used for the train/test split and for the
                internal model evaluation. Stored on the instance.
        """
        self.X = X
        self.target = target
        self._random_state: int = random_state
        self._is_fitted: bool = False
        self._state = PipelineState()
        self._fitted_components = FittedComponents()
        self._input_schema: Optional[dict] = None
        self._datetime_cols: List[str] = []
        self.dropped_columns: Dict[str, List[str]] = {}

        # Initialize target type attributes
        self.pred_type: Optional[str] = None
        self.n_classes: Optional[int] = None
        self.eval_metric: Optional[str] = None

        # Fail early, with a clear message, on unusable input.
        self._validate_dataframe(X)

        # Detect target type
        self._detect_target_type(X)
        self._state.target = target
        self._state.task_type = "regression" if self.pred_type == "Reg" else "classification"
        self._state.n_classes = self.n_classes
        self._state.evaluation_metric = self.eval_metric

    # --- Validation -----------------------------------------------------

    def _validate_dataframe(self, X: pd.DataFrame, require_target: bool = True) -> None:
        """Validate input DataFrame, additionally rejecting duplicate column names."""
        super()._validate_dataframe(X, require_target=require_target)

        duplicated = X.columns[X.columns.duplicated()].unique().tolist()
        if duplicated:
            raise ValidationError(
                f"Input DataFrame has duplicate column names: {duplicated}. "
                "Column names must be unique."
            )

    # --- Bookkeeping ----------------------------------------------------

    def _record_dropped(self, reason: str, columns: List[str]) -> None:
        """Record columns removed during fit, grouped by reason."""
        if not columns:
            return
        if not hasattr(self, "dropped_columns") or self.dropped_columns is None:
            self.dropped_columns = {}
        bucket = self.dropped_columns.setdefault(reason, [])
        bucket.extend(col for col in columns if col not in bucket)

    # --- Data preparation -----------------------------------------------

    def _normalize_feature_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cast boolean feature columns (``bool`` / nullable ``boolean``) to float.

        Booleans become ``0.0`` / ``1.0`` and missing values become ``NaN``,
        so they flow through scaling and imputation like any other numeric
        feature instead of bypassing them (which previously let nullable
        boolean ``NA`` values reach the output untouched). Applied identically
        at fit and at transform time.
        """
        bool_cols = get_boolean_columns(X, exclude=[self.target] if self.target else None)
        for col in bool_cols:
            X[col] = X[col].astype("float64")
        return X

    def _prepare_data(
        self,
        X: pd.DataFrame,
        null_threshold: float = 0.999,
        date_engineering: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare data with the initial, split-independent transformations.

        The caller's DataFrame is never mutated. Steps:

        1. Drop rows with a missing target *before* any casting (casting first
           would turn ``NaN`` into the literal class ``"nan"``).
        2. Cast a classification target to ``str`` and refresh class metadata.
        3. Normalise boolean features to float.
        4. Expand datetime columns into features (or drop them when
           ``date_engineering`` is False) - raw datetime columns can never be
           consumed by the downstream models.
        5. Remove high-null, constant and identifier-like columns.
        6. Move the target to the last position.

        Returns:
            Prepared DataFrame.
        """
        self.dropped_columns = {}

        X_ = X.copy()
        X_ = X_.dropna(subset=[self.target]).reset_index(drop=True)

        if self.pred_type == "Class":
            X_[self.target] = X_[self.target].astype(str)
            self.n_classes = int(X_[self.target].nunique())
            self.eval_metric = "F1" if self.n_classes > 2 else "Precision"
            self._state.n_classes = self.n_classes
            self._state.evaluation_metric = self.eval_metric

        X_ = self._normalize_feature_dtypes(X_)

        # Datetime handling.
        self._datetime_cols = get_datetime_columns(X_, exclude=[self.target])
        if self._datetime_cols:
            if date_engineering:
                X_ = engineer_datetime_features(X_, columns=self._datetime_cols, drop_original=True)
            else:
                X_ = X_.drop(columns=self._datetime_cols)
                self._record_dropped("datetime", self._datetime_cols)

        # Remove high-null columns
        X_ = self._remove_high_null_columns(X_, threshold=null_threshold)

        # Remove constant and unique columns
        X_ = self._remove_uninformative_columns(X_)

        # Ensure target is last column
        cols = [col for col in X_.columns if col != self.target] + [self.target]
        X_ = X_[cols]

        if len(cols) == 1:
            raise ValidationError(
                "No usable feature columns remain after removing datetime, "
                "high-null, constant and identifier-like columns. "
                f"Removed columns by reason: {self.dropped_columns}"
            )

        return X_

    def _split_data(self, X: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets using ``self._random_state``.

        Classification targets are split with stratification, which requires
        at least two samples per class.

        Returns:
            Tuple of (train, test) DataFrames.
        """
        X = X.dropna(subset=[self.target])
        seed = getattr(self, "_random_state", self.DEFAULT_RANDOM_STATE)

        if self.pred_type == "Class":
            counts = X[self.target].value_counts()
            rare = counts[counts < 2].index.tolist()
            if rare:
                raise ValidationError(
                    f"Target '{self.target}' has class(es) with a single sample, which "
                    f"cannot be stratified into train/test: {rare}. Remove or merge "
                    "these classes before fitting."
                )
            train, test = train_test_split(
                X, train_size=split_ratio, stratify=X[self.target], random_state=seed
            )
        else:
            train, test = train_test_split(X, train_size=split_ratio, random_state=seed)

        return train.reset_index(drop=True), test.reset_index(drop=True)

    def _remove_high_null_columns(self, X: pd.DataFrame, threshold: float = 0.999) -> pd.DataFrame:
        """
        Remove columns whose null fraction is above ``threshold``.

        Fully-null columns are always removed regardless of the threshold
        (they carry no information and break imputer fitting). The target is
        never removed.

        Returns:
            DataFrame with high-null columns removed.
        """
        if len(X) == 0:
            return X

        null_fraction = X.isnull().mean()
        # Small tolerance so a column sitting exactly on the threshold is kept
        # despite floating-point rounding.
        to_drop = null_fraction[(null_fraction > threshold + 1e-12) | (null_fraction >= 1.0)]
        dropped = [col for col in to_drop.index if col != self.target]

        self._record_dropped("high_null", dropped)
        return X.drop(columns=dropped)

    def _remove_uninformative_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant columns and all-unique identifier-like columns.

        A column is dropped when it is constant (a single unique value) or
        when every value is unique. The all-unique rule is deliberately NOT
        applied to floating-point columns: a continuous numeric feature on
        clean data almost always has ``nunique == len(X)`` yet is a perfectly
        valid predictor. Restricting the all-unique rule to non-float dtypes
        (object / integer / category - i.e. likely ID columns) prevents the
        silent deletion of every continuous feature.

        Every removal is recorded in ``self.dropped_columns``.

        Returns:
            DataFrame with uninformative columns removed.
        """
        n_rows = len(X)
        constant: List[str] = []
        identifiers: List[str] = []
        for col in X.columns:
            if col == self.target:
                continue
            n_unique = X[col].nunique()
            if n_unique <= 1:
                constant.append(col)
            elif n_unique == n_rows and not pd.api.types.is_float_dtype(X[col]):
                identifiers.append(col)

        self._record_dropped("constant", constant)
        self._record_dropped("unique_identifier", identifiers)
        return X.drop(columns=constant + identifiers)

    # --- Template method ------------------------------------------------

    def fit_processing(
        self,
        split_ratio: float = 0.75,
        relevance: float = 0.99,
        h2o_fs_models: int = 7,
        encoding_fs: bool = True,
        vif_ratio: float = 10.0,
        optimization_level: str = "balanced",
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
            train, test = self._select_features(train, test, relevance, h2o_fs_models, encoding_fs)

        # Snapshot pre-encoded train for fitting inference-time transformers.
        pre_encoded_train = train.copy()

        # Encoding optimization
        train, test = self._optimize_encoding(train, test, optimization_level)

        # Imputation optimization
        train, test = self._optimize_imputation(train, test, optimization_level)

        # VIF filtering
        train, test = self._apply_vif_filtering(train, test, vif_ratio)

        # Fit final processors on the pre-encoded training split - never on
        # the full dataframe (would leak test rows) and never on the
        # post-encoding train (would lose raw categorical state).
        self._fit_final_processors(pre_encoded_train)

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

    @abstractmethod
    def _select_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        relevance: float,
        h2o_fs_models: int,
        encoding_fs: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select features using H2O AutoML."""
        ...

    @abstractmethod
    def _optimize_encoding(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize encoding version selection."""
        ...

    @abstractmethod
    def _optimize_imputation(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize imputation method selection."""
        ...

    @abstractmethod
    def _apply_vif_filtering(
        self, train: pd.DataFrame, test: pd.DataFrame, vif_threshold: float
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
    """Legacy chained-configuration mixin.

    .. deprecated::
        These ``with_*`` methods only record values in ``_builder_config``;
        nothing in :meth:`Atlantic.fit_processing` reads them, so calling them
        on an ``Atlantic`` instance has never changed how the pipeline is
        fitted. Each call now emits a ``FutureWarning`` saying so. Use
        :meth:`Atlantic.builder` (``AtlanticBuilder``) for a configuration
        that is actually honoured, or pass the options to
        ``fit_processing`` directly.
    """

    def __init__(self):
        self._builder_config: Dict[str, Any] = {}

    @staticmethod
    def _warn_inert(method: str) -> None:
        warnings.warn(
            f"{method}() on an Atlantic/BuilderPipeline instance only records the value and "
            "does not affect fit_processing(). Use Atlantic.builder() (AtlanticBuilder) or pass "
            "the option to fit_processing() instead.",
            FutureWarning,
            stacklevel=3,
        )

    def with_date_engineering(self, enabled: bool = True, drop: bool = True) -> "Self":
        """Record datetime feature engineering settings (not consumed; see class docs)."""
        self._warn_inert("with_date_engineering")
        self._builder_config["date_engineering"] = enabled
        self._builder_config["date_drop"] = drop
        return self

    def with_null_removal(self, threshold: float = 0.99) -> "Self":
        """Record null column removal threshold (not consumed; see class docs)."""
        self._warn_inert("with_null_removal")
        self._builder_config["null_removal_threshold"] = threshold
        return self

    def with_feature_selection(
        self,
        method: str = "h2o",
        relevance: float = 0.99,
        h2o_models: int = 7,
        encoding_fs: bool = True,
    ) -> "Self":
        """Record feature selection settings (not consumed; see class docs)."""
        self._warn_inert("with_feature_selection")
        self._builder_config["feature_selection_method"] = method
        self._builder_config["h2o_relevance"] = relevance
        self._builder_config["h2o_models"] = h2o_models
        self._builder_config["encoding_fs"] = encoding_fs
        return self

    def with_encoding(
        self, scaler: str = "standard", encoder: str = "ifrequency", auto_select: bool = False
    ) -> "Self":
        """Record encoding strategy (not consumed; see class docs)."""
        self._warn_inert("with_encoding")
        self._builder_config["scaler"] = scaler
        self._builder_config["encoder"] = encoder
        self._builder_config["auto_encoding"] = auto_select
        return self

    def with_imputation(
        self, method: str = "simple", auto_select: bool = False, **params
    ) -> "Self":
        """Record imputation strategy (not consumed; see class docs)."""
        self._warn_inert("with_imputation")
        self._builder_config["imputer"] = method
        self._builder_config["auto_imputation"] = auto_select
        self._builder_config["imputer_params"] = params
        return self

    def with_vif_filtering(self, threshold: float = 10.0) -> "Self":
        """Record VIF threshold (not consumed; see class docs)."""
        self._warn_inert("with_vif_filtering")
        self._builder_config["vif_threshold"] = threshold
        return self

    def with_optimization(
        self, optimization_level: str = "balanced", random_state: int = 42
    ) -> "Self":
        """Record optimization settings (not consumed; see class docs)."""
        self._warn_inert("with_optimization")
        self._builder_config["optimization_level"] = optimization_level
        self._builder_config["random_state"] = random_state
        return self
