from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


class DatasetGenerator:
    """
    Generate synthetic datasets for preprocessing evaluation.

    Provides methods to create classification and regression datasets
    with configurable characteristics including mixed types and null values.
    """

    CATEGORY_POOLS = {
        "low": ["A", "B", "C", "D", "E"],
        "medium": [f"Cat_{i}" for i in range(20)],
        "high": [f"Category_{i}" for i in range(100)],
    }

    @staticmethod
    def _introduce_nulls(
        df: pd.DataFrame, target_name: str, null_percentage: float
    ) -> pd.DataFrame:
        """
        Introduce null values into DataFrame.

        Returns:
            DataFrame with nulls introduced.
        """
        df = df.copy()
        n_samples = len(df)

        for col in df.columns:
            if col == target_name:
                continue

            n_nulls = int(n_samples * null_percentage)
            if n_nulls > 0:
                null_indices = np.random.choice(df.index, size=n_nulls, replace=False)
                df.loc[null_indices, col] = np.nan

        return df

    @staticmethod
    def generate_classification(
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 10,
        n_classes: int = 2,
        n_categorical: int = 5,
        null_percentage: float = 0.05,
        categorical_cardinality: str = "low",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate synthetic classification dataset with mixed types and nulls.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        np.random.seed(random_state)

        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=random_state,
        )

        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)

        # Add categorical features
        categories = DatasetGenerator.CATEGORY_POOLS.get(
            categorical_cardinality, DatasetGenerator.CATEGORY_POOLS["low"]
        )
        for i in range(n_categorical):
            df[f"cat_feature_{i}"] = np.random.choice(categories, size=n_samples)

        # Add target
        target_name = "target"
        if n_classes == 2:
            df[target_name] = ["class_0" if v == 0 else "class_1" for v in y]
        else:
            df[target_name] = [f"class_{v}" for v in y]

        # Introduce nulls
        if null_percentage > 0:
            df = DatasetGenerator._introduce_nulls(df, target_name, null_percentage)

        return df, target_name

    @staticmethod
    def generate_regression(
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 10,
        noise: float = 0.1,
        n_categorical: int = 5,
        null_percentage: float = 0.05,
        categorical_cardinality: str = "low",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate synthetic regression dataset with mixed types and nulls.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        np.random.seed(random_state)

        # Generate base regression data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise * 100,
            random_state=random_state,
        )

        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)

        # Add categorical features
        categories = DatasetGenerator.CATEGORY_POOLS.get(
            categorical_cardinality, DatasetGenerator.CATEGORY_POOLS["low"]
        )
        for i in range(n_categorical):
            df[f"cat_feature_{i}"] = np.random.choice(categories, size=n_samples)

        # Add target
        target_name = "target"
        df[target_name] = y

        # Introduce nulls
        if null_percentage > 0:
            df = DatasetGenerator._introduce_nulls(df, target_name, null_percentage)

        return df, target_name

    @staticmethod
    def generate_with_datetime(
        n_samples: int = 1000,
        n_numeric: int = 10,
        n_categorical: int = 5,
        task_type: Literal["classification", "regression"] = "regression",
        date_range: Tuple[str, str] = ("2020-01-01", "2024-01-01"),
        null_percentage: float = 0.05,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate dataset with datetime columns for date engineering testing.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        np.random.seed(random_state)

        # Calculate informative features (must be <= n_numeric)
        n_informative = min(n_numeric // 2, n_numeric - 1)

        # Generate base dataset
        if task_type == "classification":
            df, target_name = DatasetGenerator.generate_classification(
                n_samples=n_samples,
                n_features=n_numeric,
                n_informative=n_informative,
                n_categorical=n_categorical,
                null_percentage=0,
                random_state=random_state,
            )
        else:
            df, target_name = DatasetGenerator.generate_regression(
                n_samples=n_samples,
                n_features=n_numeric,
                n_informative=n_informative,
                n_categorical=n_categorical,
                null_percentage=0,
                random_state=random_state,
            )

        # Add datetime columns
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1])
        date_range_seconds = int((end - start).total_seconds())

        random_seconds = np.random.randint(0, date_range_seconds, size=n_samples)
        df["datetime_col"] = start + pd.to_timedelta(random_seconds, unit="s")

        # Add second datetime column with different pattern
        df["event_date"] = df["datetime_col"] + pd.to_timedelta(
            np.random.randint(0, 30, size=n_samples), unit="D"
        )

        # Introduce nulls
        if null_percentage > 0:
            df = DatasetGenerator._introduce_nulls(df, target_name, null_percentage)

        return df, target_name

    @staticmethod
    def quick_classification(n_samples: int = 500, n_classes: int = 2) -> Tuple[pd.DataFrame, str]:
        """
        Quick classification dataset with sensible defaults.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        return DatasetGenerator.generate_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_classes=n_classes,
            n_categorical=3,
            null_percentage=0.02,
        )

    @staticmethod
    def quick_regression(n_samples: int = 500) -> Tuple[pd.DataFrame, str]:
        """
        Quick regression dataset with sensible defaults.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        return DatasetGenerator.generate_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_categorical=3,
            null_percentage=0.02,
        )

    @staticmethod
    def generate_high_null(
        n_samples: int = 1000,
        null_percentage: float = 0.30,
        task_type: Literal["classification", "regression"] = "regression",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate dataset with high null percentage for imputation testing.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        if task_type == "classification":
            return DatasetGenerator.generate_classification(
                n_samples=n_samples,
                n_features=15,
                n_informative=8,
                n_categorical=5,
                null_percentage=null_percentage,
                random_state=random_state,
            )
        else:
            return DatasetGenerator.generate_regression(
                n_samples=n_samples,
                n_features=15,
                n_informative=8,
                n_categorical=5,
                null_percentage=null_percentage,
                random_state=random_state,
            )

    @staticmethod
    def generate_high_cardinality(
        n_samples: int = 1000,
        n_categorical: int = 10,
        cardinality_range: Tuple[int, int] = (50, 200),
        task_type: Literal["classification", "regression"] = "regression",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate dataset with high-cardinality categorical features.

        Returns:
            Tuple of (DataFrame, target_column_name).
        """
        np.random.seed(random_state)

        n_features = 10
        n_informative = 5

        # Generate base dataset
        if task_type == "classification":
            df, target_name = DatasetGenerator.generate_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_categorical=0,
                null_percentage=0,
                random_state=random_state,
            )
        else:
            df, target_name = DatasetGenerator.generate_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_categorical=0,
                null_percentage=0,
                random_state=random_state,
            )

        # Add high-cardinality categorical features
        for i in range(n_categorical):
            cardinality = np.random.randint(cardinality_range[0], cardinality_range[1])
            categories = [f"hc_cat_{i}_{j}" for j in range(cardinality)]
            df[f"high_card_feature_{i}"] = np.random.choice(categories, size=n_samples)

        return df, target_name
