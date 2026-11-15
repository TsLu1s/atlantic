"""Tests for atlantic.state.pipeline_state module."""

import json
import pickle

import pandas as pd
import pytest

from atlantic.core.exceptions import StateSerializationError
from atlantic.core.schemas import FittedComponents, PipelineConfig, PipelineState
from atlantic.state.pipeline_state import StateManager


class TestStateManagerMetadata:
    def test_save_and_load_metadata(self, tmp_path):
        state = PipelineState(
            config=PipelineConfig(split_ratio=0.8),
            task_type="classification",
            n_classes=3,
            target="y",
            is_fitted=True,
        )
        path = tmp_path / "meta.json"
        StateManager.save_metadata(state, str(path))
        loaded = StateManager.load_metadata(str(path))
        assert loaded.task_type == "classification"
        assert loaded.n_classes == 3
        assert loaded.target == "y"
        assert loaded.is_fitted is True

    def test_load_metadata_file_not_found(self, tmp_path):
        with pytest.raises(StateSerializationError, match="not found"):
            StateManager.load_metadata(str(tmp_path / "nonexistent.json"))


class _FakePipeline:
    """Module-level class so it can be pickled."""
    def __init__(self):
        self.value = 42
        self.name = "test"


class TestStateManagerFullState:
    def test_save_and_load_full_state(self, tmp_path):
        pipeline = _FakePipeline()
        path = tmp_path / "pipeline.pkl"
        StateManager.save_full_state(pipeline, str(path))
        loaded = StateManager.load_full_state(str(path))
        assert loaded.value == 42
        assert loaded.name == "test"

    def test_load_full_state_file_not_found(self, tmp_path):
        with pytest.raises(StateSerializationError, match="not found"):
            StateManager.load_full_state(str(tmp_path / "nonexistent.pkl"))


class TestStateManagerExportConfig:
    def test_export_config(self, tmp_path):
        state = PipelineState(
            config=PipelineConfig(),
            fitted_components=FittedComponents(
                selected_columns=["a", "b"],
                encoding_version="v1",
            ),
            task_type="regression",
            target="y",
        )
        path = tmp_path / "config.json"
        StateManager.export_config(state, str(path))

        with open(path) as f:
            data = json.load(f)

        assert data["task_type"] == "regression"
        assert data["target"] == "y"
        assert data["selected_columns"] == ["a", "b"]
        assert data["encoding_version"] == "v1"
