"""
Atlantic - Main Pipeline Module.

Implements the main Atlantic preprocessing pipeline.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_integer_dtype
from pydantic import ValidationError as PydanticValidationError

from atlantic._version import __version__
from atlantic.core.base import BasePipeline, BuilderPipeline
from atlantic.core.enums import TaskType
from atlantic.core.exceptions import (
    H2OConnectionError,
    NotFittedError,
    SchemaValidationError,
    ValidationError,
)
from atlantic.core.schemas import FittedComponents, PipelineConfig, PipelineState
from atlantic.encoding.versions import EncodingVersionFactory
from atlantic.evaluation.metrics import MetricRegistry
from atlantic.feature_selection.h2o_selector import (
    H2O_INSTALL_HINT,
    H2OFeatureSelector,
    is_h2o_available,
)
from atlantic.optimization.evaluation import ModelEvaluator
from atlantic.pipeline.pattern import Pattern
from atlantic.preprocessing.imputers import AutoIterativeImputer, AutoKNNImputer, AutoSimpleImputer
from atlantic.preprocessing.registry import EncoderRegistry, ScalerRegistry
from atlantic.utils.columns import (
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    is_boolean_column,
    is_datetime_column,
    is_numeric_column,
)
from atlantic.utils.datetime import engineer_datetime_features

if TYPE_CHECKING:
    from typing import Self

    from atlantic.pipeline.builder import AtlanticBuilder

# Encoding-version display labels mapped to version ids.
_DISPLAY_TO_VERSION: Dict[str, str] = {
    "Encoding Version 1": "v1",
    "Encoding Version 2": "v2",
    "Encoding Version 3": "v3",
    "Encoding Version 4": "v4",
}

# Canonical display names for imputation methods.
_IMPUTER_DISPLAY: Dict[str, str] = {
    "simple": "Simple",
    "knn": "KNN",
    "iterative": "Iterative",
}

_SCALER_CHOICES: Tuple[str, ...] = ("standard", "minmax", "robust")
_ENCODER_CHOICES: Tuple[str, ...] = ("label", "ifrequency")
_IMPUTER_CHOICES: Tuple[str, ...] = ("simple", "knn", "iterative")

_LABEL_LIKE_MAX_UNIQUE: int = 10

# Defaults for attributes missing from pipelines saved by older versions.
_STATE_DEFAULTS: Dict[str, Any] = {
    "_date_engineering": True,
    "_date_drop": True,
    "_datetime_cols": None,
    "_n_jobs": None,
    "_verbosity": 1,
    "_input_schema": None,
    "_perf": None,
}


class Atlantic(BasePipeline, BuilderPipeline):
    """
    Atlantic - Automated Data Preprocessing Framework.

    Provides comprehensive data preprocessing capabilities including
    feature engineering, feature selection, encoding, imputation,
    and transformation for supervised machine learning.

    The task type is inferred from the target column: a numeric target is a
    regression task, anything else (strings, categories, booleans) is a
    classification task. Integer-coded class labels (``0``/``1``) are
    therefore treated as regression - cast them to ``str`` first to get
    classification.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        verbosity: int = 1,
    ):
        """
        Initialize Atlantic pipeline.

        Args:
            X: Input DataFrame to process.
            target: Name of target column.
            random_state: Seed for the train/test split and for every internal
                model evaluation.
            n_jobs: Parallelism of the evaluation models. ``None`` (default)
                keeps the scikit-learn forests single-threaded; ``-1`` uses all
                cores for them too. Affects runtime only, never the result
                (see ``ModelEvaluator`` for a caveat on Python < 3.14).
            verbosity: ``0`` silences progress output, ``1`` (default) prints
                the selection summary, ``2`` adds per-candidate detail.
        """
        self._random_state: int = random_state
        self._n_jobs: Optional[int] = n_jobs
        self._verbosity: int = verbosity

        if X is not None and target is not None:
            BasePipeline.__init__(self, X, target, random_state=random_state)
            self._warn_if_label_like_target()
        else:
            # Deferred construction (e.g. ``Atlantic()`` before ``Atlantic.load``).
            self.X = X
            self.target = target
            self._is_fitted = False
            self._state = PipelineState()
            self._fitted_components = FittedComponents()
            self._input_schema = None
            self._datetime_cols = []
            self.dropped_columns = {}
            self.pred_type = None
            self.n_classes = None
            self.eval_metric = None

        BuilderPipeline.__init__(self)

        # Preprocessing components
        self.enc_method: Optional[str] = None
        self.imp_method: Optional[str] = None
        self.encoder = None
        self.scaler = None
        self.imputer = None

        # Resolved scaler/encoder registry keys
        self._scaler_type: Optional[str] = None
        self._encoder_type: Optional[str] = None

        # Builder configuration
        self._auto_encoding: bool = True
        self._auto_imputation: bool = True
        self._manual_scaler: Optional[str] = None
        self._manual_encoder: Optional[str] = None
        self._manual_imputer: Optional[str] = None
        self._imputer_params: Dict[str, Any] = {}

        # Datetime handling chosen at fit time and replayed at transform time.
        self._date_engineering: bool = True
        self._date_drop: bool = True

        self._perf: Optional[float] = None

        # Column tracking
        self.n_cols: List[str] = []
        self.c_cols: List[str] = []
        self.cols: List[str] = []

        # Feature importance
        self.h2o_feature_importance: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _warn_if_label_like_target(self) -> None:
        """Warn when a numeric target looks like integer-coded class labels."""
        if self.pred_type != "Reg":
            return
        target = self.X[self.target].dropna()
        if target.empty:
            return
        n_unique = int(target.nunique())
        if n_unique > _LABEL_LIKE_MAX_UNIQUE:
            return
        integer_valued = is_integer_dtype(target) or bool((target % 1 == 0).all())
        if integer_valued:
            warnings.warn(
                f"Target '{self.target}' is numeric with only {n_unique} distinct integer "
                "value(s) and is being treated as a REGRESSION target. If these are class "
                f"labels, cast the column to str first, e.g. "
                f"df['{self.target}'] = df['{self.target}'].astype(str).",
                UserWarning,
                stacklevel=3,
            )

    def _require_data(self) -> None:
        """Ensure both the DataFrame and the target name were provided."""
        if self.X is None or self.target is None:
            raise ValidationError(
                "Atlantic needs both a DataFrame and a target column name to fit: "
                "Atlantic(X=dataframe, target='target_column')."
            )
        if self.pred_type is None:
            # Constructed without data and populated afterwards.
            self._validate_dataframe(self.X)
            self._detect_target_type(self.X)
            self._state.target = self.target
            self._state.task_type = "regression" if self.pred_type == "Reg" else "classification"

    @staticmethod
    def _validate_manual_choices(
        auto_encoding: bool,
        scaler: Optional[str],
        encoder: Optional[str],
        auto_imputation: bool,
        imputer: Optional[str],
    ) -> None:
        """Validate builder-supplied component keys before fitting."""
        if not auto_encoding:
            if scaler is not None and scaler not in _SCALER_CHOICES:
                raise ValidationError(
                    f"Unknown scaler '{scaler}'. Choose from 'standard', 'minmax', 'robust'."
                )
            if encoder is not None and encoder not in _ENCODER_CHOICES:
                if encoder == "onehot":
                    raise ValidationError(
                        "encoder='onehot' is not supported in the automated "
                        "pipeline because one-hot expansion changes the column "
                        "structure the pipeline tracks. Use AutoOneHotEncoder "
                        "as a standalone component instead."
                    )
                raise ValidationError(
                    f"Unknown encoder '{encoder}'. Choose from 'label', 'ifrequency'."
                )
        if not auto_imputation and imputer is not None and imputer not in _IMPUTER_CHOICES:
            raise ValidationError(
                f"Unknown imputer '{imputer}'. Choose from 'simple', 'knn', 'iterative'."
            )

    def _reset_fit_state(self) -> None:
        """Clear everything a previous ``fit_processing`` call left behind.

        Guarantees a refit never reuses a stale baseline score, feature
        importance or transformer from an earlier fit.
        """
        self._is_fitted = False
        self._state.is_fitted = False
        self._fitted_components = FittedComponents()
        self.enc_method = None
        self.imp_method = None
        self.encoder = None
        self.scaler = None
        self.imputer = None
        self._scaler_type = None
        self._encoder_type = None
        self._perf = None
        self.n_cols = []
        self.c_cols = []
        self.cols = []
        self.h2o_feature_importance = None
        self.dropped_columns = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit_processing(
        self,
        split_ratio: float = 0.75,
        relevance: float = 0.99,
        h2o_fs_models: int = 7,
        encoding_fs: bool = True,
        vif_ratio: float = 10.0,
        optimization_level: str = "balanced",
        *,
        null_threshold: float = 0.999,
        date_engineering: bool = True,
        date_drop: bool = True,
        auto_encoding: bool = True,
        scaler: Optional[str] = None,
        encoder: Optional[str] = None,
        auto_imputation: bool = True,
        imputer: Optional[str] = None,
        imputer_params: Optional[Dict[str, Any]] = None,
    ) -> "Self":
        """
        Fit the preprocessing pipeline.

        Args:
            split_ratio: Train fraction of the internal train/validation split
                used to compare preprocessing candidates (0.5 - 0.98).
            relevance: Share of H2O feature importance to retain (0.4 - 1.0).
                ``1.0`` skips H2O feature selection entirely; any lower value
                requires the optional ``h2o`` dependency and Java.
            h2o_fs_models: Number of H2O AutoML models for feature selection.
            encoding_fs: Label-encode categoricals before H2O selection.
            vif_ratio: VIF threshold for multicollinearity filtering (3 - 30).
            optimization_level: "fast", "balanced" or "thorough".

        The keyword-only arguments are what the :class:`AtlanticBuilder` wires
        through so its ``with_encoding`` / ``with_imputation`` /
        ``with_null_removal`` / ``with_date_engineering`` configuration is
        honoured:

        Args:
            null_threshold: Drop columns whose null fraction exceeds this.
            date_engineering: Expand datetime columns into numeric features.
                When False, datetime columns are not used as features.
            date_drop: Whether ``data_processing`` drops the original datetime
                columns from its output. When False they are passed through
                unchanged (raw datetimes are never fed to the models).
            auto_encoding: When True, the best scaler+encoder combination is
                selected automatically. When False, the ``scaler`` and
                ``encoder`` arguments are applied verbatim.
            scaler: Explicit scaler key ("standard" / "minmax" / "robust");
                only used when ``auto_encoding`` is False.
            encoder: Explicit encoder key ("label" / "ifrequency"); only used
                when ``auto_encoding`` is False. "onehot" is not supported in
                the automated pipeline (use ``AutoOneHotEncoder`` standalone).
            auto_imputation: When True, the best imputation method is selected
                automatically. When False, ``imputer`` is applied verbatim.
            imputer: Explicit imputer key ("simple" / "knn" / "iterative");
                only used when ``auto_imputation`` is False.
            imputer_params: Extra keyword arguments forwarded to the imputer
                constructor (e.g. ``{"n_neighbors": 5}``).

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: On invalid input data or configuration.
            H2OConnectionError: If ``relevance < 1.0`` and H2O is unavailable.
        """
        # Validate inputs
        self._require_data()
        self._validate_dataframe(self.X)
        self._validate_split_ratio(split_ratio)
        self._validate_manual_choices(auto_encoding, scaler, encoder, auto_imputation, imputer)

        try:
            config = PipelineConfig(
                split_ratio=split_ratio,
                relevance=relevance,
                h2o_fs_models=h2o_fs_models,
                encoding_fs=encoding_fs,
                vif_threshold=vif_ratio,
                optimization_level=optimization_level,
                random_state=self._random_state,
            )
        except PydanticValidationError as e:
            details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise ValidationError(f"Invalid fit_processing configuration - {details}") from e

        if relevance < 1.0 and not is_h2o_available():
            raise H2OConnectionError(H2O_INSTALL_HINT)

        # Start from a clean slate, then record the configuration.
        self._reset_fit_state()
        self._auto_encoding = auto_encoding
        self._auto_imputation = auto_imputation
        self._manual_scaler = scaler
        self._manual_encoder = encoder
        self._manual_imputer = imputer
        self._imputer_params = dict(imputer_params or {})
        self._date_engineering = date_engineering
        self._date_drop = date_drop
        self._state.config = config

        # Capture the raw input schema for reference/debuggability.
        self._store_input_schema(self.X)

        # Data preparation
        X_ = self._prepare_data(
            self.X, null_threshold=null_threshold, date_engineering=date_engineering
        )

        if self._verbosity >= 1 and self.dropped_columns:
            print(f"Columns removed before modelling: {self.dropped_columns}")

        # Split data
        train, test = self._split_data(X_, split_ratio)

        # Feature selection with H2O
        if relevance < 1.0:
            train, test = self._select_features(train, test, relevance, h2o_fs_models, encoding_fs)

        # Snapshot pre-encoded train split
        pre_encoded_train = train.copy()

        # Encoding + imputation
        if auto_encoding and auto_imputation:
            train, test = self._optimize_encoding(train, test, optimization_level)
            train, test = self._optimize_imputation(train, test, optimization_level)
        else:
            train, test = self._configured_encode_impute(train, test, optimization_level)

        # VIF filtering
        train, test = self._apply_vif_filtering(train, test, vif_ratio)

        # Fit final processors on the pre-encoded train snapshot.
        self._fit_final_processors(pre_encoded_train)

        self._is_fitted = True
        self._state.is_fitted = True

        return self

    def _new_pattern(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Pattern:
        """Build a Pattern optimizer sharing this pipeline's seed/parallelism."""
        return Pattern(
            train=train,
            test=test,
            target=self.target,
            optimization_level=optimization_level,
            verbosity=self._verbosity,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
        )

    def _select_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        relevance: float,
        h2o_fs_models: int,
        encoding_fs: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select features using H2O AutoML."""
        selector = H2OFeatureSelector(
            target=self.target,
            relevance=relevance,
            max_models=h2o_fs_models,
            encoding_before_selection=encoding_fs,
        )

        selector.fit(train)

        features = [col for col in train.columns if col != self.target]
        selected = [col for col in selector.selected_features if col in features]
        if not selected:
            selected = features

        self.h2o_feature_importance = selector.feature_importance
        self._record_dropped("feature_selection", [c for c in features if c not in set(selected)])

        sel_cols = selected + [self.target]
        return train[sel_cols], test[sel_cols]

    def _optimize_encoding(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize encoding version selection."""
        ptn = self._new_pattern(train, test, optimization_level)

        self.enc_method = ptn.encoding_selection()
        self._perf = ptn.perf

        version_id = _DISPLAY_TO_VERSION.get(self.enc_method)
        if version_id is not None:
            self._scaler_type, self._encoder_type = EncodingVersionFactory.get_version_config(
                version_id
            )

        return train, test

    def _eval_metric_name(self) -> str:
        """Primary evaluation metric name for the detected task."""
        task_type = TaskType.REGRESSION if self.pred_type == "Reg" else TaskType.CLASSIFICATION
        return MetricRegistry.get_primary_metric_name(task_type, self.n_classes)

    def _apply_encoding_types(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        scaler_type: str,
        encoder_type: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply an explicit scaler+encoder combination to train/test.

        Fits on train only; accepts any registered scaler (including
        "robust").
        """
        return EncodingVersionFactory.apply_types(
            train, test, self.target, scaler_type, encoder_type
        )

    def _build_imputer(self, method: Optional[str], params: Optional[Dict[str, Any]] = None):
        """Construct an imputer for ``method`` honouring ``params`` overrides.

        ``method`` is the display name ("Simple"/"KNN"/"Iterative"); anything
        else (including "Undefined") yields a mean SimpleImputer fallback so a
        transformer always exists at inference time.
        """
        p = dict(params or {})
        if method == "KNN":
            return AutoKNNImputer(
                n_neighbors=p.get("n_neighbors", 3),
                weights=p.get("weights", "uniform"),
                target=self.target,
            )
        if method == "Iterative":
            return AutoIterativeImputer(
                max_iter=p.get("max_iter", 10),
                random_state=p.get("random_state", self._random_state),
                initial_strategy=p.get("initial_strategy", "mean"),
                imputation_order=p.get("imputation_order", "ascending"),
                target=self.target,
            )
        # "Simple" and the "Undefined" fallback.
        return AutoSimpleImputer(
            strategy=p.get("strategy", "mean"),
            target=self.target,
            fill_value=p.get("fill_value"),
        )

    def _configured_encode_impute(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encoding + imputation honouring the builder configuration.

        Activated whenever ``auto_encoding`` or ``auto_imputation`` is False.
        Encoding is applied exactly once, then imputation is either selected
        automatically (on the already-encoded data) or applied as configured.
        """
        # --- Resolve + apply encoding ---
        if self._auto_encoding:
            ptn = self._new_pattern(train, test, optimization_level)
            self.enc_method = ptn.encoding_selection()
            self._perf = ptn.perf
            version_id = _DISPLAY_TO_VERSION.get(self.enc_method, "v1")
            self._scaler_type, self._encoder_type = EncodingVersionFactory.get_version_config(
                version_id
            )
        else:
            self._scaler_type = self._manual_scaler or "standard"
            self._encoder_type = self._manual_encoder or "ifrequency"
            self.enc_method = f"Manual ({self._scaler_type} + {self._encoder_type})"

        train, test = self._apply_encoding_types(train, test, self._scaler_type, self._encoder_type)

        # --- Imputation on the already-encoded data ---
        has_nulls = train.isnull().sum().sum() > 0 or test.isnull().sum().sum() > 0
        if not has_nulls:
            self.imp_method = "Undefined"
            return train, test

        if self._auto_imputation:
            return self._select_imputation(train, test, optimization_level)

        self._perf = None
        self.imp_method = _IMPUTER_DISPLAY.get(self._manual_imputer or "simple", "Simple")
        imputer = self._build_imputer(self.imp_method, self._imputer_params)
        imputer.fit(train)
        return imputer.transform(train.copy()), imputer.transform(test.copy())

    def _select_imputation(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select the best imputation method on already-encoded data.

        Mirrors Pattern.imputation_selection but operates on data that is
        already encoded (so no re-encoding happens), fitting each imputer on
        train only and scoring via ModelEvaluator.
        """
        eval_metric = self._eval_metric_name()
        methods = {
            "Simple": self._build_imputer("Simple", self._imputer_params),
            "KNN": self._build_imputer("KNN", self._imputer_params),
            "Iterative": self._build_imputer("Iterative", self._imputer_params),
        }

        scores: Dict[str, float] = {}
        imputed: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for name, imputer in methods.items():
            imputer.fit(train)
            tr = imputer.transform(train.copy())
            te = imputer.transform(test.copy())
            evaluator = ModelEvaluator(
                train=tr,
                test=te,
                target=self.target,
                optimization_level=optimization_level,
                random_state=self._random_state,
                verbosity=0,
                n_jobs=self._n_jobs,
            )
            scores[name] = float(evaluator.auto_evaluate()[eval_metric].iloc[0])
            imputed[name] = (tr, te)

        best = (min if self.pred_type == "Reg" else max)(scores, key=scores.get)
        self.imp_method = best
        self._perf = scores[best]
        return imputed[best]

    def _optimize_imputation(
        self, train: pd.DataFrame, test: pd.DataFrame, optimization_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize imputation method selection."""
        # Check if imputation is needed
        has_nulls = train.isnull().sum().sum() > 0 or test.isnull().sum().sum() > 0

        if not has_nulls:
            self.imp_method = "Undefined"

            # Apply the selected encoding without imputation optimization.
            if self._scaler_type is not None and self._encoder_type is not None:
                train, test = self._apply_encoding_types(
                    train, test, self._scaler_type, self._encoder_type
                )

            if self._verbosity >= 1:
                print("There are no missing values in the Dataset")
            return train, test

        # Run imputation optimization
        ptn = self._new_pattern(train, test, optimization_level)
        ptn.enc_method = self.enc_method

        train, test = ptn.imputation_selection(simple_score=self._perf)
        self.imp_method = ptn.imp_method
        self._perf = ptn.perf

        return train, test

    def _apply_vif_filtering(
        self, train: pd.DataFrame, test: pd.DataFrame, vif_threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply VIF-based feature filtering.

        The baseline-vs-VIF performance comparison runs at the
        ``optimization_level`` and seed the rest of the pipeline uses, so both
        candidates are scored under identical conditions.
        """
        opt_level = self._state.config.optimization_level if self._state.config else "balanced"
        ptn = self._new_pattern(train, test, opt_level)

        train, test = ptn.vif_performance(vif_threshold=vif_threshold, perf_=self._perf)

        self._record_dropped("multicollinearity", ptn.vif_removed)
        self.cols = [col for col in train.columns if col != self.target]

        return train, test

    def _fit_final_processors(self, X: pd.DataFrame) -> None:
        """Fit the inference-time scaler, encoder and imputer on the train split."""
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
        self._fitted_components.datetime_columns = list(self._datetime_cols or [])
        self._fitted_components.dropped_columns = dict(self.dropped_columns)

        # Fit scaler on numerical columns
        if self.n_cols:
            self.scaler = ScalerRegistry.get(self._scaler_type or "standard")
            self.scaler.fit(data[self.n_cols])
            data[self.n_cols] = self.scaler.transform(data[self.n_cols])

        # Fit encoder on categorical columns
        if self.c_cols:
            self.encoder = EncoderRegistry.get(self._encoder_type or "ifrequency")
            self.encoder.fit(data[self.c_cols])
            data = self.encoder.transform(data)

        # Fit imputer
        self.imputer = self._build_imputer(self.imp_method, self._imputer_params)
        self.imputer.fit(data)

        # Store fitted components
        self._fitted_components.encoder = self.encoder
        self._fitted_components.scaler = self.scaler
        self._fitted_components.imputer = self.imputer

        # Sync pipeline state
        self._state.fitted_components = self._fitted_components
        self._state.feature_importance = self.h2o_feature_importance

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted components."""
        return self.data_processing(X)

    def _resolve_datetime_columns(self, data: pd.DataFrame) -> List[str]:
        """Datetime columns to handle at transform time.

        Uses the columns identified during fit. A fitted datetime column that
        arrives as text (typical after a CSV round-trip) is parsed back into
        datetimes; values that cannot be parsed become missing and are then
        imputed like any other gap.
        """
        fitted = getattr(self, "_datetime_cols", None)
        if fitted is None:
            # Pipeline saved by a version that did not record them.
            return get_datetime_columns(data, exclude=[self.target] if self.target else None)

        present = [col for col in fitted if col in data.columns]
        for col in present:
            if not is_datetime_column(data[col]):
                data[col] = pd.to_datetime(data[col], errors="coerce")
        return present

    def _coerce_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make sure columns that were numeric at fit time are numeric now.

        Numeric-looking text is converted; anything else raises a
        :class:`SchemaValidationError` naming the column instead of failing
        deep inside scikit-learn.
        """
        for col in self.n_cols:
            series = data[col]
            if is_numeric_column(series):
                continue
            if is_boolean_column(series):
                data[col] = series.astype("float64")
                continue
            try:
                data[col] = pd.to_numeric(series)
            except (ValueError, TypeError) as e:
                raise SchemaValidationError(
                    f"Column '{col}' was numeric when the pipeline was fitted but has "
                    f"dtype '{series.dtype}' with non-numeric values now."
                ) from e
        return data

    def data_processing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to new data.

        The input is never mutated. The output holds the fitted feature
        columns (scaled / encoded / imputed), followed by the original
        datetime columns when the pipeline was fitted with
        ``date_drop=False``, followed by the target column when it is present
        in ``X`` (returned untouched).

        Returns:
            Transformed DataFrame.

        Raises:
            NotFittedError: If called before ``fit_processing``.
            SchemaValidationError: If ``X`` lacks a fitted column or a numeric
                column now holds non-numeric values.
        """
        if not self._is_fitted:
            raise NotFittedError("Atlantic")
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        data = X.copy()
        has_target = bool(self.target) and self.target in data.columns

        # Same dtype normalisation as at fit time (booleans -> float).
        data = self._normalize_feature_dtypes(data)

        # Datetime handling, replaying the fit-time configuration.
        datetime_cols = self._resolve_datetime_columns(data)
        keep_raw_dates = not getattr(self, "_date_drop", True)
        raw_dates = data[datetime_cols].copy() if keep_raw_dates and datetime_cols else None

        if getattr(self, "_date_engineering", True):
            data = engineer_datetime_features(data, columns=datetime_cols, drop_original=True)

        # Validate schema
        missing = [c for c in self.cols if c not in data.columns]
        if missing:
            fitted_dates = getattr(self, "_datetime_cols", None) or []
            absent_dates = sorted(
                {d for d in fitted_dates if d not in X.columns for c in missing if c.startswith(d)}
            )
            hint = (
                f" (derived from datetime column(s) {absent_dates}, absent from the input)"
                if absent_dates
                else ""
            )
            raise SchemaValidationError(
                f"Input is missing {len(missing)} column(s) required by the "
                f"fitted pipeline: {missing}{hint}"
            )

        # Select columns (target kept if present)
        data = data[self.cols + [self.target]] if has_target else data[self.cols]

        # Apply scaler
        if self.n_cols and self.scaler is not None:
            data = self._coerce_numeric_features(data)
            data[self.n_cols] = self.scaler.transform(data[self.n_cols])

        # Apply encoder
        if self.c_cols and self.encoder is not None:
            data = self.encoder.transform(data)

        # Apply imputer
        if self.imputer is not None and data[self.cols].isnull().sum().sum() > 0:
            data = self.imputer.transform(data)

        # Pass the original datetime columns through when requested.
        if raw_dates is not None:
            for col in datetime_cols:
                data[col] = raw_dates[col]
            ordered = self.cols + datetime_cols + ([self.target] if has_target else [])
            data = data[ordered]

        return data

    # ------------------------------------------------------------------
    # Builder / accessors
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Serialization hook.

        The raw training frame ``self.X`` is deliberately excluded: it is only
        needed during ``fit_processing`` and would otherwise be pickled into
        every saved pipeline - bloating the file and shipping the user's
        training data. Inference (``data_processing``) never reads ``self.X``.
        The Atlantic version is stamped so a later load can detect a mismatch.
        """
        state = self.__dict__.copy()
        state["X"] = None
        state["_atlantic_version"] = __version__
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Deserialization hook.

        Fills in attributes introduced by newer versions and warns when the
        pipeline was saved by a different Atlantic version.
        """
        saved_version = state.get("_atlantic_version")
        self.__dict__.update(state)

        for name, default in _STATE_DEFAULTS.items():
            self.__dict__.setdefault(name, default)
        self.__dict__.setdefault("dropped_columns", {})

        if saved_version != __version__:
            origin = f"Atlantic {saved_version}" if saved_version else "an older Atlantic version"
            warnings.warn(
                f"This pipeline was saved by {origin} and is being loaded with Atlantic "
                f"{__version__}. Transformations may differ; re-fit the pipeline if results "
                "look inconsistent.",
                UserWarning,
                stacklevel=2,
            )

    def save(self, path: str) -> None:
        """
        Save fitted pipeline to file.

        The training DataFrame is not stored. The file is a pickle: share it
        only with parties who trust you, and load only files you trust.

        Args:
            path: File path for saving (pickle format).
        """
        from atlantic.state.pipeline_state import StateManager

        StateManager.save_full_state(self, path)

    @classmethod
    def load(cls, path: str) -> "Atlantic":
        """
        Load fitted pipeline from file.

        Only load files from a trusted source: unpickling can execute
        arbitrary code.

        Returns:
            Loaded Atlantic instance.
        """
        from atlantic.state.pipeline_state import StateManager

        return StateManager.load_full_state(path)
