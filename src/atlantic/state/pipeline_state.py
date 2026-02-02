from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

from atlantic.core.exceptions import StateSerializationError
from atlantic.core.schemas import FittedComponents, PipelineState


class StateManager:
    """
    Manages pipeline state serialization and deserialization.

    Supports both JSON (metadata only) and pickle (full state with fitted objects).
    """

    @staticmethod
    def save_metadata(state: PipelineState, path: Union[str, Path]) -> None:
        """
        Save pipeline metadata to JSON file.

        Args:
            state: PipelineState to save.
            path: File path for JSON output.
        """
        path = Path(path)

        try:
            with open(path, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
        except Exception as e:
            raise StateSerializationError(f"Failed to save metadata: {e}")

    @staticmethod
    def load_metadata(path: Union[str, Path]) -> PipelineState:
        """
        Load pipeline metadata from JSON file.

        Returns:
            PipelineState (without fitted objects).
        """
        path = Path(path)

        if not path.exists():
            raise StateSerializationError(f"File not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return PipelineState.from_dict(data)
        except Exception as e:
            raise StateSerializationError(f"Failed to load metadata: {e}")

    @staticmethod
    def save_full_state(pipeline: Any, path: Union[str, Path]) -> None:
        """
        Save complete pipeline state including fitted objects.

        Args:
            pipeline: Atlantic pipeline instance.
            path: File path for pickle output.
        """
        path = Path(path)

        try:
            with open(path, "wb") as f:
                pickle.dump(pipeline, f)
        except Exception as e:
            raise StateSerializationError(f"Failed to save full state: {e}")

    @staticmethod
    def load_full_state(path: Union[str, Path]) -> Any:
        """
        Load complete pipeline state from pickle file.

        Returns:
            Atlantic pipeline instance.
        """
        path = Path(path)

        if not path.exists():
            raise StateSerializationError(f"File not found: {path}")

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise StateSerializationError(f"Failed to load full state: {e}")

    @staticmethod
    def export_config(state: PipelineState, path: Union[str, Path]) -> None:
        """
        Export pipeline configuration for reproducibility.

        Args:
            state: PipelineState to export.
            path: File path for JSON output.
        """
        path = Path(path)

        config_data = {
            "task_type": state.task_type,
            "target": state.target,
            "n_classes": state.n_classes,
            "evaluation_metric": state.evaluation_metric,
            "encoding_version": state.fitted_components.encoding_version,
            "imputation_method": state.fitted_components.imputation_method,
            "selected_columns": state.fitted_components.selected_columns,
            "numerical_columns": state.fitted_components.numerical_columns,
            "categorical_columns": state.fitted_components.categorical_columns,
            "config": state.config.model_dump() if state.config else None,
        }

        try:
            with open(path, "w") as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            raise StateSerializationError(f"Failed to export config: {e}")
