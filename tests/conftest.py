"""Shared fixtures for Atlantic test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_regression_df():
    """DataFrame for regression tasks with mixed types and some nulls."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "num_a": np.random.randn(n),
            "num_b": np.random.rand(n) * 100,
            "num_c": np.random.randint(0, 50, n).astype(float),
            "cat_x": np.random.choice(["A", "B", "C", "D"], n),
            "cat_y": np.random.choice(["low", "mid", "high"], n),
            "target": np.random.randn(n) * 10 + 50,
        }
    )
    # Introduce some nulls
    null_idx = np.random.choice(n, 10, replace=False)
    df.loc[null_idx, "num_a"] = np.nan
    df.loc[null_idx[:5], "num_c"] = np.nan
    return df


@pytest.fixture
def sample_classification_df():
    """DataFrame for classification tasks with mixed types and some nulls."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "num_a": np.random.randn(n),
            "num_b": np.random.rand(n) * 100,
            "num_c": np.random.randint(0, 50, n).astype(float),
            "cat_x": np.random.choice(["A", "B", "C", "D"], n),
            "cat_y": np.random.choice(["low", "mid", "high"], n),
            "target": np.random.choice(["class_0", "class_1"], n),
        }
    )
    null_idx = np.random.choice(n, 10, replace=False)
    df.loc[null_idx, "num_a"] = np.nan
    df.loc[null_idx[:5], "num_c"] = np.nan
    return df


@pytest.fixture
def sample_multiclass_df():
    """DataFrame for multi-class classification."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame(
        {
            "num_a": np.random.randn(n),
            "num_b": np.random.rand(n) * 100,
            "cat_x": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice(["c0", "c1", "c2", "c3"], n),
        }
    )
    return df


@pytest.fixture
def numeric_df():
    """Pure numeric DataFrame with no nulls."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.rand(n) * 10,
            "c": np.random.randint(1, 100, n).astype(float),
        }
    )


@pytest.fixture
def categorical_df():
    """Pure categorical DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "col1": np.random.choice(["A", "B", "C", "D", "E"], n),
            "col2": np.random.choice(["low", "mid", "high"], n),
            "col3": np.random.choice(["x", "y"], n),
        }
    )


@pytest.fixture
def datetime_df():
    """DataFrame with datetime columns."""
    np.random.seed(42)
    n = 50
    base_date = pd.Timestamp("2023-01-01")
    return pd.DataFrame(
        {
            "event_date": pd.date_range(start=base_date, periods=n, freq="h"),
            "num_a": np.random.randn(n),
            "target": np.random.randn(n),
        }
    )


@pytest.fixture
def train_test_regression(sample_regression_df):
    """Pre-split train/test for regression."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(sample_regression_df, train_size=0.75, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


@pytest.fixture
def train_test_classification(sample_classification_df):
    """Pre-split train/test for classification."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(sample_classification_df, train_size=0.75, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)
