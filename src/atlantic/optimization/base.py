from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

class BaseOptimizerAdapter(ABC):
    """
    Abstract adapter for hyperparameter optimization frameworks.
    
    Provides unified interface for different optimization backends
    (Optuna, Hyperopt, Ray Tune, etc.)
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize optimizer adapter.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self._random_state = random_state
        self._study = None
        self._best_params: Dict[str, Any] = {}
        self._best_value: Optional[float] = None
    
    @abstractmethod
    def create_study(
        self,
        direction: str = "minimize",
        study_name: Optional[str] = None
    ) -> Any:
        """
        Create optimization study.

        Returns:
            Study object.
        """
        ...
    
    @abstractmethod
    def optimize(
        self,
        objective: Callable,
        n_trials: int,
        callbacks: Optional[List[Callable]] = None,
        show_progress: bool = True
    ) -> Any:
        """
        Run optimization.

        Returns:
            Optimization result.
        """
        ...
    
    @abstractmethod
    def suggest_int(
        self,
        trial: Any,
        name: str,
        low: int,
        high: int
    ) -> int:
        """
        Suggest integer hyperparameter.

        Returns:
            Suggested integer value.
        """
        ...
    
    @abstractmethod
    def suggest_float(
        self,
        trial: Any,
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
        ...
    
    @abstractmethod
    def suggest_categorical(
        self,
        trial: Any,
        name: str,
        choices: List[Any]
    ) -> Any:
        """
        Suggest categorical hyperparameter.

        Returns:
            Suggested value from choices.
        """
        ...
    
    @property
    def best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        return self._best_params
    
    @property
    def best_value(self) -> Optional[float]:
        """Get best objective value found."""
        return self._best_value
    
    @abstractmethod
    def get_best_trial(self) -> Any:
        """Get best trial object."""
        ...