"""
Atlantic - Optuna Adapter.

Optuna implementation of the optimizer adapter interface.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional

import optuna
from tqdm import tqdm

from atlantic.optimization.base import BaseOptimizerAdapter

if TYPE_CHECKING:
    import pandas as pd


class OptunaAdapter(BaseOptimizerAdapter):
    """
    Optuna-specific implementation of optimizer adapter.

    Wraps Optuna's optimization interface with unified API.

    Optuna's log verbosity is process-global. The adapter only lowers/raises
    it *while it is creating or running a study* and restores the previous
    level afterwards, so using Atlantic never permanently silences (or
    un-silences) Optuna logging in the caller's application.
    """

    def __init__(self, random_state: Optional[int] = None, verbosity: int = 0):
        """
        Initialize Optuna adapter.

        Args:
            random_state: Random seed for reproducibility.
            verbosity: Logging verbosity level (0=silent, 1=warnings, 2=info).
        """
        super().__init__(random_state)
        self._verbosity = verbosity

    def _optuna_level(self) -> int:
        """Map the adapter verbosity to an Optuna logging level."""
        if self._verbosity <= 0:
            return optuna.logging.ERROR
        if self._verbosity == 1:
            return optuna.logging.WARNING
        return optuna.logging.INFO

    @contextmanager
    def _logging_scope(self) -> Iterator[None]:
        """Temporarily apply this adapter's Optuna verbosity, then restore it."""
        previous = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(self._optuna_level())
        try:
            yield
        finally:
            optuna.logging.set_verbosity(previous)

    def create_study(
        self, direction: str = "minimize", study_name: Optional[str] = None
    ) -> optuna.Study:
        """
        Create Optuna study.

        Returns:
            Optuna Study object.
        """
        sampler = optuna.samplers.TPESampler(seed=self._random_state)

        with self._logging_scope():
            self._study = optuna.create_study(
                direction=direction, study_name=study_name, sampler=sampler
            )

        return self._study

    def optimize(
        self,
        objective: Callable,
        n_trials: int,
        callbacks: Optional[List[Callable]] = None,
        show_progress: bool = True,
    ) -> optuna.Study:
        """
        Run Optuna optimization.

        Returns:
            Optuna Study object after optimization.
        """
        if self._study is None:
            self.create_study()

        # Copy so we never mutate a caller-supplied callbacks list.
        all_callbacks = list(callbacks) if callbacks else []

        pbar = None
        if show_progress:
            pbar = tqdm(total=n_trials, desc="Optimizing", ncols=75)

            def progress_callback(study, trial):
                pbar.update(1)

            all_callbacks.append(progress_callback)

        try:
            with self._logging_scope():
                self._study.optimize(
                    objective,
                    n_trials=n_trials,
                    callbacks=all_callbacks if all_callbacks else None,
                    show_progress_bar=False,
                )
        finally:
            if pbar is not None:
                pbar.close()

        # Store best results
        self._best_params = self._study.best_params
        self._best_value = self._study.best_value

        return self._study

    def suggest_int(self, trial: optuna.Trial, name: str, low: int, high: int) -> int:
        """
        Suggest integer hyperparameter.

        Returns:
            Suggested integer value.
        """
        return trial.suggest_int(name, low, high)

    def suggest_float(
        self, trial: optuna.Trial, name: str, low: float, high: float, log: bool = False
    ) -> float:
        """
        Suggest float hyperparameter.

        Returns:
            Suggested float value.
        """
        return trial.suggest_float(name, low, high, log=log)

    def suggest_categorical(self, trial: optuna.Trial, name: str, choices: List[Any]) -> Any:
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
