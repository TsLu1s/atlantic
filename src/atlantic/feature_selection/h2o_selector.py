"""
Atlantic - H2O AutoML Feature Selection.

Trains H2O AutoML models, reads the leader's variable importance and keeps
the features that account for a requested share of it.

H2O ships with Atlantic, but its cluster needs a Java runtime. The package is
imported lazily, only when a selector is fitted, so the rest of Atlantic stays
usable (and fast to import) on machines without Java.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

import pandas as pd

from atlantic.core.exceptions import (
    AtlanticError,
    FeatureSelectionError,
    H2OConnectionError,
    ValidationError,
)
from atlantic.core.mixins import TargetTypeMixin
from atlantic.preprocessing.base import BaseFeatureSelector
from atlantic.preprocessing.encoders import AutoLabelEncoder
from atlantic.utils.columns import get_categorical_columns

if TYPE_CHECKING:
    from typing import Self

    import h2o
    from h2o.automl import H2OAutoML

H2O_INSTALL_HINT: str = (
    "H2O feature selection needs the 'h2o' package, which is installed with Atlantic, "
    "and a Java runtime. Reinstall it with `pip install --force-reinstall atlantic`, "
    "or skip H2O feature selection by passing relevance=1.0 to fit_processing() "
    '(builder: with_feature_selection(method="none")).'
)


def is_h2o_available() -> bool:
    """Whether the ``h2o`` package can be imported."""
    if "h2o" in sys.modules:
        # A ``None`` entry means the import has been explicitly blocked.
        return sys.modules["h2o"] is not None
    try:
        return importlib.util.find_spec("h2o") is not None
    except (ImportError, ValueError):
        return False


class H2OFeatureSelector(BaseFeatureSelector, TargetTypeMixin):
    """
    H2O AutoML based feature selection.

    Uses H2O's AutoML to train models and extract feature importance,
    then selects features based on cumulative importance threshold.

    Note: the H2O cluster is initialised inside ``fit``; afterwards only the
    frames and models this selector created are removed from it.
    """

    H2O_MIN_RELEVANCE: float = 0.4
    H2O_MAX_RELEVANCE: float = 1.0
    H2O_MIN_MODELS: int = 1
    H2O_MAX_MODELS: int = 100
    H2O_EXCLUDED_ALGORITHMS: List[str] = ["GLM", "DeepLearning", "StackedEnsemble"]

    def __init__(
        self,
        target: str,
        relevance: float = 0.99,
        max_models: int = 7,
        excluded_algorithms: Optional[List[str]] = None,
        encoding_before_selection: bool = True,
        seed: int = 1,
    ):
        """
        Initialize H2O feature selector.

        Args:
            target: Target column name.
            relevance: Share of cumulative feature importance to retain.
            max_models: Number of AutoML models to train.
            excluded_algorithms: H2O algorithms to exclude from AutoML.
            encoding_before_selection: Label-encode categoricals before
                handing the frame to H2O.
            seed: Seed passed to H2O AutoML.
        """
        super().__init__(target)

        if not self.H2O_MIN_RELEVANCE <= relevance <= self.H2O_MAX_RELEVANCE:
            raise ValidationError(
                f"relevance must be between {self.H2O_MIN_RELEVANCE} "
                f"and {self.H2O_MAX_RELEVANCE}, got {relevance}"
            )

        if not self.H2O_MIN_MODELS <= max_models <= self.H2O_MAX_MODELS:
            raise ValidationError(
                f"max_models must be between {self.H2O_MIN_MODELS} "
                f"and {self.H2O_MAX_MODELS}, got {max_models}"
            )

        self._relevance = relevance
        self._max_models = max_models
        self._excluded_algorithms = excluded_algorithms or list(self.H2O_EXCLUDED_ALGORITHMS)
        self._encoding_before_selection = encoding_before_selection
        self._seed = seed

        # H2O keys created by this selector, for scoped cleanup
        self._h2o_keys: List[str] = []

    def _initialize_h2o(self) -> None:
        """
        Initialize H2O server.

        Raises:
            H2OConnectionError: If ``h2o`` is not installed or the cluster
                cannot be started (typically a missing Java runtime).
        """
        try:
            import h2o
        except ImportError as e:
            raise H2OConnectionError(H2O_INSTALL_HINT) from e

        try:
            h2o.init()
        except Exception as e:
            raise H2OConnectionError(
                f"Failed to initialize H2O: {e}. H2O requires a working Java runtime."
            ) from e

    def _shutdown_h2o(self) -> None:
        """
        Release ONLY the H2O keys this selector created during fit.

        We deliberately avoid both ``h2o.shutdown`` (which tears down the
        entire cluster) and ``h2o.remove_all`` (which drops *every* key in the
        shared cluster - including frames and models owned by other code in the
        user's process). Only the frames and models tracked in
        ``self._h2o_keys`` are removed, leaving any other H2O state intact.
        Cleanup is best-effort: individual removal failures are swallowed.
        """
        try:
            import h2o
        except Exception:
            self._h2o_keys = []
            return

        for key in self._h2o_keys:
            try:
                h2o.remove(key)
            except Exception:  # noqa: S110 - best-effort cleanup
                pass  # Key may already be gone / cluster down.

        self._h2o_keys = []

    def _prepare_data(self, X: pd.DataFrame) -> "h2o.H2OFrame":
        """
        Prepare data for H2O processing.
        """
        import h2o

        X_processed = X.copy()

        cat_cols = get_categorical_columns(X_processed, exclude=[self.target])

        # Encode categorical columns if requested
        if self._encoding_before_selection and cat_cols:
            encoder = AutoLabelEncoder()
            encoder.fit(X_processed[cat_cols])
            X_processed = encoder.transform(X_processed)
        else:
            # H2O's pandas ingestion expects plain object columns for text.
            for col in cat_cols:
                X_processed[col] = X_processed[col].astype(object)

        if self.pred_type == "Class":
            X_processed[self.target] = X_processed[self.target].astype(object)

        # Convert to H2O Frame
        h2o_frame = h2o.H2OFrame(X_processed)

        # Track this frame's key for scoped cleanup.
        try:
            self._h2o_keys.append(h2o_frame.frame_id)
        except Exception:  # noqa: S110 - tracking is best-effort
            pass

        # Set target as factor for classification
        if self.pred_type == "Class":
            h2o_frame[self.target] = h2o_frame[self.target].asfactor()

        return h2o_frame

    def _configure_automl(self) -> "H2OAutoML":
        """
        Configure H2O AutoML settings.
        """
        from h2o.automl import H2OAutoML

        return H2OAutoML(
            max_models=self._max_models,
            nfolds=3,
            seed=self._seed,
            exclude_algos=self._excluded_algorithms,
            sort_metric="AUTO",
        )

    def _train_and_get_importance(
        self, h2o_frame: "h2o.H2OFrame", features: List[str]
    ) -> pd.DataFrame:
        """
        Train AutoML and extract feature importance.

        The leaderboard is walked best-first and the first model that exposes
        variable importance is used (some algorithms, e.g. stacked ensembles,
        do not provide it).
        """
        import h2o

        aml = self._configure_automl()
        aml.train(x=features, y=self.target, training_frame=h2o_frame)

        leaderboard = aml.leaderboard.as_data_frame()
        model_ids = [str(model_id) for model_id in leaderboard["model_id"].tolist()]

        # Track every trained model for scoped cleanup.
        self._h2o_keys.extend(model_ids)

        for model_id in model_ids:
            importance = h2o.get_model(model_id).varimp(use_pandas=True)
            if importance is not None and len(importance) > 0:
                return importance

        raise FeatureSelectionError(
            "None of the H2O AutoML models exposes variable importance; "
            "feature selection cannot be performed with the current algorithm set."
        )

    @staticmethod
    def _normalize_importance(importance_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Map importance rows back onto the input feature names.

        H2O reports one-hot expanded categoricals as ``"<column>.<level>"``.
        Those rows are folded back into their source column (importances
        summed) so the selection always yields real column names. Rows that
        cannot be attributed to an input feature are discarded.
        """
        feature_set = set(features)
        variables = importance_df["variable"].astype(str)
        if variables.isin(feature_set).all():
            return importance_df

        def resolve(name: str) -> Optional[str]:
            if name in feature_set:
                return name
            candidates = [f for f in features if name.startswith(f"{f}.")]
            return max(candidates, key=len) if candidates else None

        resolved = importance_df.copy()
        resolved["variable"] = variables.map(resolve)
        resolved = resolved.dropna(subset=["variable"])

        numeric_cols = [c for c in resolved.columns if c != "variable"]
        folded = resolved.groupby("variable", sort=False)[numeric_cols].sum().reset_index()
        return folded.sort_values("percentage", ascending=False, kind="stable").reset_index(
            drop=True
        )

    def _select_by_importance(self, importance_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features based on cumulative importance.

        Starting at a 1.5% per-feature importance cut-off, the cut-off is
        halved until the retained features account for more than
        ``relevance`` of the total importance (or the cut-off reaches its
        floor).
        """
        percentage = importance_df["percentage"].fillna(0.0)
        threshold = 0.015

        while True:
            filtered = importance_df[percentage > threshold]
            if percentage[percentage > threshold].sum() <= self._relevance:
                threshold *= 0.5
                if threshold < 0.001:
                    break
            else:
                break

        selected = filtered["variable"].tolist()
        return selected, filtered

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Self":
        """
        Fit H2O feature selector.

        Raises:
            H2OConnectionError: If H2O is not installed or cannot start.
            FeatureSelectionError: If training or importance extraction fails.
        """
        # Detect target type
        self._detect_target_type(X)

        # Reset tracked keys for this fit so cleanup only touches new state.
        self._h2o_keys = []

        try:
            # Initialize H2O
            self._initialize_h2o()

            # Prepare data
            h2o_frame = self._prepare_data(X)
            features = [col for col in X.columns if col != self.target]

            # Train and get importance
            importance_df = self._train_and_get_importance(h2o_frame, features)
            importance_df = self._normalize_importance(importance_df, features)

            # Select features
            selected, filtered_importance = self._select_by_importance(importance_df)

            if not selected:
                warnings.warn(
                    "H2O variable importance did not single out any feature; "
                    "keeping all features.",
                    UserWarning,
                    stacklevel=2,
                )
                selected, filtered_importance = list(features), importance_df

            self._selected_features = selected
            self._feature_importance = filtered_importance
            self._is_fitted = True

        except AtlanticError:
            raise
        except Exception as e:
            raise FeatureSelectionError(f"H2O feature selection failed: {e}") from e

        finally:
            # Always release this selector's H2O keys
            self._shutdown_h2o()

        return self

    @property
    def relevance_threshold(self) -> float:
        """Get relevance threshold used."""
        return self._relevance
