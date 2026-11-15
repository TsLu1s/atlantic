"""Tests for atlantic.data.generator module."""

import numpy as np
import pandas as pd
import pytest

from atlantic.data.generator import DatasetGenerator


class TestDatasetGeneratorClassification:
    def test_binary_classification(self):
        df, target = DatasetGenerator.generate_classification(
            n_samples=100, n_features=10, n_informative=5, n_classes=2, n_categorical=2, random_state=42
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert target == "target"
        assert df[target].nunique() == 2

    def test_multiclass_classification(self):
        df, target = DatasetGenerator.generate_classification(
            n_samples=200, n_features=10, n_informative=5, n_classes=4, n_categorical=3, random_state=42
        )
        assert df[target].nunique() == 4

    def test_null_percentage(self):
        df, target = DatasetGenerator.generate_classification(
            n_samples=1000, n_features=10, n_informative=5, null_percentage=0.1, random_state=42
        )
        # Some columns should have nulls
        assert df.drop(columns=[target]).isnull().sum().sum() > 0

    def test_no_nulls(self):
        df, target = DatasetGenerator.generate_classification(
            n_samples=100, null_percentage=0.0, random_state=42
        )
        assert df.isnull().sum().sum() == 0

    def test_categorical_columns_present(self):
        df, _ = DatasetGenerator.generate_classification(n_categorical=4, random_state=42)
        cat_cols = [c for c in df.columns if c.startswith("cat_feature_")]
        assert len(cat_cols) == 4

    def test_target_not_nulled(self):
        df, target = DatasetGenerator.generate_classification(
            n_samples=500, null_percentage=0.2, random_state=42
        )
        assert df[target].isnull().sum() == 0


class TestDatasetGeneratorRegression:
    def test_basic(self):
        df, target = DatasetGenerator.generate_regression(
            n_samples=100, n_features=5, random_state=42
        )
        assert len(df) == 100
        assert target == "target"
        assert np.issubdtype(df[target].dtype, np.floating)

    def test_null_percentage(self):
        df, target = DatasetGenerator.generate_regression(
            n_samples=1000, null_percentage=0.15, random_state=42
        )
        assert df.drop(columns=[target]).isnull().sum().sum() > 0


class TestDatasetGeneratorDatetime:
    def test_datetime_columns_present(self):
        df, target = DatasetGenerator.generate_with_datetime(
            n_samples=100, random_state=42
        )
        assert "datetime_col" in df.columns
        assert "event_date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["datetime_col"])


class TestDatasetGeneratorQuick:
    def test_quick_classification(self):
        df, target = DatasetGenerator.quick_classification(n_samples=50)
        assert len(df) == 50
        assert target == "target"

    def test_quick_regression(self):
        df, target = DatasetGenerator.quick_regression(n_samples=50)
        assert len(df) == 50
        assert target == "target"


class TestDatasetGeneratorHighNull:
    def test_high_null(self):
        df, target = DatasetGenerator.generate_high_null(
            n_samples=200, null_percentage=0.3, random_state=42
        )
        null_pct = df.drop(columns=[target]).isnull().sum().sum() / (
            len(df) * (df.shape[1] - 1)
        )
        assert null_pct > 0.2


class TestDatasetGeneratorHighCardinality:
    def test_high_cardinality(self):
        df, target = DatasetGenerator.generate_high_cardinality(
            n_samples=200, n_categorical=3, cardinality_range=(50, 100), random_state=42
        )
        hc_cols = [c for c in df.columns if c.startswith("high_card_feature_")]
        assert len(hc_cols) == 3
        for col in hc_cols:
            assert df[col].nunique() >= 50
