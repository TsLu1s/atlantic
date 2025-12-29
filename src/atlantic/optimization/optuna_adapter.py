from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Self, TYPE_CHECKING

import optuna
from tqdm import tqdm

from atlantic.optimization.base import BaseOptimizerAdapter

class OptunaAdapter(BaseOptimizerAdapter):
    """
    Optuna-specific implementation of optimizer adapter.
    
    Wraps Optuna's optimization interface with unified API.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        verbosity: int = 0
    ):
        """
        Initialize Optuna adapter.
        
        Args:
            random_state: Random seed for reproducibility.
            verbosity: Logging verbosity level (0=silent, 1=warnings, 2=info).
        """
        super().__init__(random_state)
        self._verbosity = verbosity
        self._configure_logging()
    
    def _configure_logging(self) -> None:
        """Configure Optuna logging based on verbosity."""
        if self._verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            logging.getLogger('optuna').setLevel(logging.ERROR)
            logging.getLogger('optuna').disabled = True
        elif self._verbosity == 1:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)
    
    def create_study(
        self,
        direction: str = "minimize",
        study_name: Optional[str] = None
    ) -> optuna.Study:
        """
        Create Optuna study.

        Returns:
            Optuna Study object.
        """
        sampler = optuna.samplers.TPESampler(seed=self._random_state)
        
        self._study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=sampler
        )
        
        return self._study
    
    def optimize(
        self,
        objective: Callable,
        n_trials: int,
        callbacks: Optional[List[Callable]] = None,
        show_progress: bool = True
    ) -> optuna.Study:
        """
        Run Optuna optimization.

        Returns:
            Optuna Study object after optimization.
        """
        if self._study is None:
            self.create_study()
        
        all_callbacks = callbacks or []
        
        if show_progress:
            pbar = tqdm(total=n_trials, desc="Optimizing", ncols=75)
            
            def progress_callback(study, trial):
                pbar.update(1)
            
            all_callbacks.append(progress_callback)
        
        try:
            self._study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=all_callbacks if all_callbacks else None,
                show_progress_bar=False
            )
        finally:
            if show_progress:
                pbar.close()
        
        # Store best results
        self._best_params = self._study.best_params
        self._best_value = self._study.best_value
        
        return self._study
    
    def suggest_int(
        self,
        trial: optuna.Trial,
        name: str,
        low: int,
        high: int
    ) -> int:
        """
        Suggest integer hyperparameter.

        Returns:
            Suggested integer value.
        """
        return trial.suggest_int(name, low, high)
    
    def suggest_float(
        self,
        trial: optuna.Trial,
        name: str,
        low: float,
        high: float,
        log: bool = False
    ) -> float:
        """
        Suggest float hyperparameter.

        Returns:
            Suggested float value.
        """
        if log:
            return trial.suggest_loguniform(name, low, high)
        return trial.suggest_float(name, low, high)
    
    def suggest_categorical(
        self,
        trial: optuna.Trial,
        name: str,
        choices: List[Any]
    ) -> Any:
        """
        Suggest categorical hyperparameter.

        Returns:
            Suggested value from choices.
        """
        return trial.suggest_categorical(name, choices)
    
    def get_best_trial(self) -> Optional[optuna.trial.FrozenTrial]:
        """
        Get best trial object.
        
        Returns:
            Best trial or None if no optimization run.
        """
        if self._study is None:
            return None
        return self._study.best_trial
    
    @property
    def study(self) -> Optional[optuna.Study]:
        """Get underlying Optuna study."""
        return self._study
    
    def get_trials_dataframe(self) -> Optional["pd.DataFrame"]:
        """
        Get trials as DataFrame.
        
        Returns:
            DataFrame with trial information or None.
        """
        if self._study is None:
            return None
        return self._study.trials_dataframe()