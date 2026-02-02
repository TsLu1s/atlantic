from __future__ import annotations

from typing import List, Optional, Self, Tuple  # , TYPE_CHECKING

import h2o
import pandas as pd
from h2o.automl import H2OAutoML

from atlantic.core.exceptions import FeatureSelectionError, H2OConnectionError
from atlantic.core.mixins import TargetTypeMixin
from atlantic.preprocessing.base import BaseFeatureSelector
from atlantic.preprocessing.encoders import AutoLabelEncoder


class H2OFeatureSelector(BaseFeatureSelector, TargetTypeMixin):
    """
    H2O AutoML based feature selection.

    Uses H2O's AutoML to train models and extract feature importance,
    then selects features based on cumulative importance threshold.

    Note: H2O server is initialized and shutdown within the fit method.
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
    ):
        """
        Initialize H2O feature selector.
        """
        super().__init__(target)

        if not self.H2O_MIN_RELEVANCE <= relevance <= self.H2O_MAX_RELEVANCE:
            raise ValueError(
                f"Relevance must be between {self.H2O_MIN_RELEVANCE}"
                f"and {self.H2O_MAX_RELEVANCE}"
            )

        if not self.H2O_MIN_MODELS <= max_models <= self.H2O_MAX_MODELS:
            raise ValueError(
                f"max_models must be between {self.H2O_MIN_MODELS} " f"and {self.H2O_MAX_MODELS}"
            )

        self._relevance = relevance
        self._max_models = max_models
        self._excluded_algorithms = excluded_algorithms or self.H2O_EXCLUDED_ALGORITHMS
        self._encoding_before_selection = encoding_before_selection

    def _initialize_h2o(self) -> None:
        """
        Initialize H2O server.
        """
        try:
            import h2o

            h2o.init()
        except Exception as e:
            raise H2OConnectionError(f"Failed to initialize H2O: {e}")

    def _shutdown_h2o(self) -> None:
        """
        Shutdown H2O server.
        """
        try:
            import h2o

            h2o.shutdown(prompt=False)
        except Exception:
            pass  # Ignore shutdown errors

    def _prepare_data(self, X: pd.DataFrame) -> "h2o.H2OFrame":
        """
        Prepare data for H2O processing.
        """
        import h2o

        X_processed = X.copy()

        # Encode categorical columns if requested
        if self._encoding_before_selection:
            cat_cols = [
                col
                for col in X.select_dtypes(include=["object", "category"]).columns
                if col != self.target
            ]
            if cat_cols:
                encoder = AutoLabelEncoder()
                encoder.fit(X_processed[cat_cols])
                X_processed = encoder.transform(X_processed)

        # Convert to H2O Frame
        h2o_frame = h2o.H2OFrame(X_processed)

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
            seed=1,
            exclude_algos=self._excluded_algorithms,
            sort_metric="AUTO",
        )

    def _train_and_get_importance(
        self, h2o_frame: "h2o.H2OFrame", features: List[str]
    ) -> pd.DataFrame:
        """
        Train AutoML and extract feature importance.
        """
        import h2o

        aml = self._configure_automl()
        aml.train(x=features, y=self.target, training_frame=h2o_frame)

        # Get best model's feature importance
        leaderboard = aml.leaderboard.as_data_frame()
        best_model = h2o.get_model(leaderboard["model_id"].iloc[0])

        return best_model.varimp(use_pandas=True)

    def _select_by_importance(self, importance_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features based on cumulative importance.
        """
        threshold = 0.015

        while True:
            filtered = importance_df[importance_df["percentage"] > threshold]
            if filtered["percentage"].sum() <= self._relevance:
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
        """
        # Detect target type
        self._detect_target_type(X)

        try:
            # Initialize H2O
            self._initialize_h2o()

            # Prepare data
            h2o_frame = self._prepare_data(X)
            features = [col for col in X.columns if col != self.target]

            # Train and get importance
            importance_df = self._train_and_get_importance(h2o_frame, features)

            # Select features
            selected, filtered_importance = self._select_by_importance(importance_df)

            self._selected_features = selected
            self._feature_importance = filtered_importance
            self._is_fitted = True

        except Exception as e:
            raise FeatureSelectionError(f"H2O feature selection failed: {e}")

        finally:
            # Always shutdown H2O
            self._shutdown_h2o()

        return self

    @property
    def relevance_threshold(self) -> float:
        """Get relevance threshold used."""
        return self._relevance
