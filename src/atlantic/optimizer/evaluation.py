import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from atlantic.processing.analysis import Analysis
from atlantic.optimizer.metrics import metrics_classification, metrics_regression

from sklearn.ensemble import (RandomForestRegressor,
                              ExtraTreesRegressor,
                              RandomForestClassifier,
                              ExtraTreesClassifier)
import xgboost as xgb
import optuna

from tqdm import tqdm
import logging

import warnings
warnings.filterwarnings("ignore", category=Warning) 

@dataclass
class ModelConfig:
    """Configuration for model hyperparameter ranges."""
    
    # Regression model parameters
    rf_regressor: Dict = field(default_factory=lambda: {
        "n_estimators": (50, 200),
        "max_depth": (5, 32),
        "min_samples_split": (2, 25)
    })
    
    et_regressor: Dict = field(default_factory=lambda: {
        "n_estimators": (50, 200),
        "max_depth": (5, 32),
        "min_samples_split": (2, 25)
    })
    
    xgb_regressor: Dict = field(default_factory=lambda: {
        "n_estimators": (50, 200),
        "max_depth": (5, 25),
        "learning_rate": (0.01, 0.1)
    })
    
    # Classification model parameters
    rf_classifier: Dict = field(default_factory=lambda: {
        "n_estimators": (60, 250),
        "max_depth": (10, 50),
        "min_samples_split": (2, 20)
    })
    
    et_classifier: Dict = field(default_factory=lambda: {
        "n_estimators": (60, 250),
        "max_depth": (10, 50),
        "min_samples_split": (2, 20)
    })
    
    xgb_classifier: Dict = field(default_factory=lambda: {
        "n_estimators": (60, 250),
        "max_depth": (10, 20),
        "learning_rate": (0.05, 0.1)
    })


class Evaluation(Analysis):
    def __init__(self, 
                 train : pd.DataFrame,
                 test : pd.DataFrame,
                 target : str):
        """
        Advanced model evaluation and hyperparameter optimization framework.
        
        This class extends the Analysis framework to provide automated model evaluation
        and hyperparameter optimization for machine learning tasks. It supports both
        regression and classification problems, utilizing Random Forest, Extra Trees,
        and XGBoost algorithms.
        """
        # Constructor for the Evaluation class, inherits from Analysis and initializes class attributes.
        super().__init__(target)
        
        self.train = train
        self.test = test
        self.metrics: Optional[pd.DataFrame] = None
        self._tmetrics: Optional[pd.DataFrame] = None
        self.hparameters_list: List[Dict] = []
        self.metrics_list: List[pd.DataFrame] = []
        
        # Determine prediction type and number of classes
        self.pred_type, self.eval_metric = self.target_type(train)
                
        self.model_config = self._initialize_model_config()
        
        # Configure logging to suppress Optuna's logs
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger('optuna').setLevel(logging.ERROR)
        logging.getLogger('optuna').disabled = True
    
    def _initialize_model_config(self) -> ModelConfig:
        """Initialize model configuration based on problem type."""
        config = ModelConfig()
        
        # Adjust XGBoost parameters for multiclass
        if self.pred_type == "Class" and self.n_classes > 2:
            config.xgb_classifier.update({
                "num_class": (self.n_classes,self.n_classes),
                "objective": ("multi:softmax","multi:softmax")
            })
            
        return config
    
    def _get_model_instances(self) -> Dict:
        """Initialize model instances based on prediction type."""
        if self.pred_type == "Reg":
            return {
                "rf_regressor": RandomForestRegressor(),
                "et_regressor": ExtraTreesRegressor(),
                "xgb_regressor": xgb.XGBRegressor()
            }
        return {
            "rf_classifier": RandomForestClassifier(),
            "et_classifier": ExtraTreesClassifier(),
            "xgb_classifier": xgb.XGBClassifier()
        }


    def _suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        model_type: str,
        pred_type: str
    ) -> Dict:
        """Generate hyperparameter suggestions for a specific model."""
        config = getattr(self.model_config, model_type)  # Use the full model type name
        params = {}
        
        for param_name, value_range in config.items():
            # Special handling for num_class
            if param_name == "num_class":
                params[param_name] = self.n_classes
                continue
            elif param_name == "objective":
                params[param_name] = "multi:softmax"
                continue
            
            if param_name == "learning_rate":
                params[param_name] = trial.suggest_loguniform(
                    f"{model_type}_{param_name}",
                    value_range[0],
                    value_range[1]
                )
            else:
                params[param_name] = trial.suggest_int(
                    f"{model_type}_{param_name}",
                    value_range[0],
                    value_range[1]
                )
                
        return params


    def _determine_evaluation_settings(
        self,
        train_size: int,
        feature_count: int
    ) -> Tuple[str, int]:
        """Determine appropriate evaluation settings based on dataset characteristics."""
        if train_size <= 8000:
            return ("low", 8) if feature_count < 30 else ("medium", 6)
        return ("mid_high", 5) if feature_count < 30 else ("high", 5)

    def objective(
        self,
        trial: optuna.Trial,
        dim: Literal["normal", "high", "low", "medium", "mid_high"] = "normal"
    ) -> None:
        """
        Define the optimization objective for model evaluation.
        
        Args:
            trial: Optuna trial object for hyperparameter suggestion
            dim: Dimensionality setting affecting model configuration
        """
        X_train, X_test, y_train, y_test = self.divide_dfs(self.train, self.test)

        models = self._get_model_instances()
        metrics_func = metrics_regression if self.pred_type == "Reg" else metrics_classification
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            results = []
            hparams = {}
            
            for model_name, model in models.items():
                if dim != "high" or model_name != "et":
                    params = self._suggest_hyperparameters(
                        trial, model_name, self.pred_type
                    )
                    model.set_params(**params)
                    model.fit(X_train, y_train)
                    
                    pred = model.predict(X_test)
                    metrics = metrics_func(y_test, pred)
                    metrics["Model"] = model_name.upper()
                    results.append(metrics)
                    
                    hparams[f"{model_name}_{self.pred_type.lower()}_params"] = params
            
            metrics_df = pd.concat(results, axis=0)
            metrics_df["iteration"] = len(self.metrics_list) + 1
            self.metrics_list.append(metrics_df)
            
            hparams["iteration"] = len(self.hparameters_list) + 1
            self.hparameters_list.append(hparams)
        
    def auto_evaluate(self) -> pd.DataFrame:
            """
            Perform automated model evaluation and hyperparameter optimization.
            
            Returns:
                DataFrame containing aggregated evaluation metrics for best models
            """
            
            dim, n_trials = self._determine_evaluation_settings(
                self.train.shape[0],
                self.train.shape[1]
            )
            
            study = optuna.create_study(
                direction="minimize" if self.pred_type == "Reg" else "maximize",
                study_name=f"{self.pred_type} Evaluation"
            )
            
            with tqdm(total=n_trials, desc="", ncols=75) as pbar:
                study.optimize(
                    lambda trial: self.objective(trial, dim=dim),
                    n_trials=n_trials,
                    callbacks=[lambda study, trial: pbar.update(1)]
                )
            
            self.metrics = pd.concat(self.metrics_list)
            sort_col = ("Mean Absolute Error" if self.pred_type == "Reg" 
                   else "F1" if self.n_classes > 2 
                   else "Precision")
            
            self.metrics = self.metrics.sort_values(
                ["Model", sort_col],
                ascending=self.pred_type == "Reg"
            )
            
            self._tmetrics = self.metrics.copy()
            self.metrics = (self.metrics.groupby("Model")
                           .first()
                           .mean(axis=0)
                           .to_frame()
                           .T)
            self.metrics.drop(columns='iteration', inplace=True)
            
            return self.metrics




