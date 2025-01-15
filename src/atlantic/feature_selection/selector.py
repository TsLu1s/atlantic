import pandas as pd
from typing import List, Tuple
import h2o
from h2o.automl import H2OAutoML
from statsmodels.stats.outliers_influence import variance_inflation_factor
from atlantic.processing.encoders import AutoLabelEncoder  
from atlantic.processing.analysis import Analysis          
from dataclasses import dataclass, field

@dataclass
class FeatureSelectionConfig:
    """Configuration parameters for feature selection."""
    
    VIF_MIN_THRESHOLD: float = 3.0
    VIF_MAX_THRESHOLD: float = 30.0
    H2O_MIN_RELEVANCE: float = 0.4
    H2O_MAX_RELEVANCE: float = 1.0
    H2O_MIN_MODELS: int = 1
    H2O_MAX_MODELS: int = 100
    H2O_EXCLUDED_ALGORITHMS: List[str] = field(default_factory=lambda: [
        'GLM', 'DeepLearning', 'StackedEnsemble'
    ])

class Selector(Analysis):
    """
    Advanced feature selection framework for machine learning models.
    
    This class provides methods for identifying and selecting the most relevant features
    using multiple techniques including Variance Inflation Factor (VIF) analysis and
    H2O AutoML-based feature importance evaluation.
    """
    def __init__(self, 
                 X : pd.DataFrame, 
                 target : str):
        super().__init__(target)
        self.X = X
        self.vif_df = None
        self.pred_type, self.eval_metric = super().target_type(X = X)
        self._fs_config = FeatureSelectionConfig()
        
        """
        Initialize the feature selector.
        
        Args:
            X: Input DataFrame containing features and target
            target: Name of target variable column
        """
        
    def calculate_vif(self,X : pd.DataFrame):
        """
        Calculate Variance Inflation Factor for numerical features.
        
        Args:
            X: Input DataFrame with numerical features only
            
        Returns:
            DataFrame containing VIF values for each feature
            
        Raises:
            ValueError: If input contains non-numerical columns or null values
        """
        
        # Check if there are any categorical columns or null values in X
        if len([col for col in X[list(X.columns)].select_dtypes(include=['number']).columns if col != self.target]) < len(list(X.columns))-1: 
            raise ValueError("Only numerical columns are supported in VIF calculation.")
        if X.isnull().values.any():
            raise ValueError("Null values are not supported in VIF calculation.")

        vif = pd.DataFrame()
        vif['variables'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) 
                      for i in range(X.shape[1])]
        
        vif = vif.sort_values(['VIF'], ascending = False)
        
        return vif

    def feature_selection_vif(self, vif_threshold : float = 10.0):
        """
        Select features using Variance Inflation Factor analysis.
        
        Args:
            vif_threshold: Maximum allowed VIF value (between 3 and 30)
            
        Returns:
            List of selected feature names
            
        Raises:
            ValueError: If threshold is outside valid range
        """
        # Perform feature selection using VIF (Variance Inflation Factor).
        if not (self._fs_config.VIF_MIN_THRESHOLD <= vif_threshold <= self._fs_config.VIF_MAX_THRESHOLD):
            raise ValueError(
                f"VIF threshold must be between {self._fs_config.VIF_MIN_THRESHOLD} "
                f"and {self._fs_config.VIF_MAX_THRESHOLD}"
            )
        
        cols = list(self.X.columns.difference([self.target]))
        X_ = self.X[cols].copy()
        
        self.vif_df = self.calculate_vif(X_)

        while self.vif_df['VIF'].max() >= vif_threshold:
            # Iteratively remove columns with VIF above the threshold.
            self.vif_df.drop(self.vif_df['variables'].loc[self.vif_df['VIF'] == self.vif_df['VIF'].max()].index,
                             inplace = True)
            cols = [rows for rows in self.vif_df['variables']]
            X_ = X_[cols]
            self.vif_df = self.calculate_vif(X_)
        cols.append(self.target)
        
        return cols
        
    def _prepare_data_for_h2o(self, encoding_fs: bool) -> h2o.H2OFrame:
        """Prepare data for H2O processing."""
        X_processed = self.X.copy()
        
        if encoding_fs:
            categorical_cols = [col for col in self.X.select_dtypes(
                include=['object', 'category']).columns 
                if col != self.target
            ]
            if categorical_cols:
                encoder = AutoLabelEncoder()
                encoder.fit(X_processed[categorical_cols])
                X_processed = encoder.transform(X_processed)
        
        h2o_frame = h2o.H2OFrame(X_processed)
        if self.pred_type == 'Class':
            h2o_frame[self.target] = h2o_frame[self.target].asfactor()
            
        return h2o_frame

    def _configure_automl(self, max_models: int) -> H2OAutoML:
        """Configure H2O AutoML settings."""
        return H2OAutoML(
            max_models=max_models,
            nfolds=3,
            seed=1,
            exclude_algos=self._fs_config.H2O_EXCLUDED_ALGORITHMS,
            sort_metric='AUTO'
        )
    
    def _train_and_evaluate_models(
        self,
        h2o_frame: h2o.H2OFrame,
        automl: H2OAutoML
    ) -> h2o.H2OFrame:
        """Train models and get feature importance."""
        features = list(self.X.columns.difference([self.target]))
        
        automl.train(
            x=features,
            y=self.target,
            training_frame=h2o_frame
        )
        
        leaderboard = automl.leaderboard.as_data_frame()
        best_model = h2o.get_model(leaderboard['model_id'].iloc[0])
        
        return best_model.varimp(use_pandas=True)

    def _select_features_by_importance(
        self,
        varimp: pd.DataFrame,
        relevance: float
    ) -> Tuple[List[str], pd.DataFrame]:
        """Select features based on cumulative importance."""
        threshold = 0.015
        while True:
            feat_imp = varimp[varimp['percentage'] > threshold]
            if feat_imp['percentage'].sum() <= relevance:
                threshold *= 0.5
            else:
                break
                
        return feat_imp['variable'].tolist(), feat_imp
    
    def feature_selection_h2o(
        self,
        relevance: float = 0.99,
        h2o_fs_models: int = 7,
        encoding_fs: bool = True
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features using H2O AutoML importance analysis.
        
        Args:
            relevance: Minimum cumulative feature importance (between 0.4 and 1)
            h2o_fs_models: Number of models to train (between 1 and 100)
            encoding_fs: Whether to encode categorical features
            
        Returns:
            Tuple containing:
                - List of selected feature names
                - DataFrame with feature importance scores
                
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        if not (self._fs_config.H2O_MIN_RELEVANCE <= relevance <= self._fs_config.H2O_MAX_RELEVANCE):
            raise ValueError(
                f"Relevance must be between {self._fs_config.H2O_MIN_RELEVANCE} "
                f"and {self._fs_config.H2O_MAX_RELEVANCE}"
            )
            
        if not (self._fs_config.H2O_MIN_MODELS <= h2o_fs_models <= self._fs_config.H2O_MAX_MODELS):
            raise ValueError(
                f"Number of models must be between {self._fs_config.H2O_MIN_MODELS} "
                f"and {self._fs_config.H2O_MAX_MODELS}"
            )

        try:
            h2o.init()
            X_processed = self._prepare_data_for_h2o(encoding_fs)
            
            aml = self._configure_automl(h2o_fs_models)
            model_results = self._train_and_evaluate_models(X_processed, aml)
            
            selected_features, importance_df = self._select_features_by_importance(
                model_results, relevance
            )
            
            return selected_features + [self.target], importance_df
            
        finally:
            h2o.shutdown(prompt=False)