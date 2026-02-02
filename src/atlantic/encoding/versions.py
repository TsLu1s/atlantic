from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from atlantic.core.enums import EncodingVersion
from atlantic.preprocessing.encoders import AutoIFrequencyEncoder, AutoLabelEncoder
from atlantic.preprocessing.registry import EncoderRegistry, ScalerRegistry
from atlantic.preprocessing.scalers import AutoMinMaxScaler, AutoStandardScaler


class EncodingVersionFactory:
    """
    Factory for creating encoding version combinations.

    Provides predefined combinations of scalers and encoders
    for automated preprocessing optimization.
    """

    # Version definitions: (scaler_type, encoder_type)
    VERSIONS: Dict[str, Tuple[str, str]] = {
        "v1": ("standard", "ifrequency"),
        "v2": ("minmax", "ifrequency"),
        "v3": ("standard", "label"),
        "v4": ("minmax", "label"),
    }

    VERSION_DESCRIPTIONS: Dict[str, str] = {
        "v1": "StandardScaler + IFrequencyEncoder",
        "v2": "MinMaxScaler + IFrequencyEncoder",
        "v3": "StandardScaler + LabelEncoder",
        "v4": "MinMaxScaler + LabelEncoder",
    }

    @classmethod
    def get_version_config(cls, version: str) -> Tuple[str, str]:
        """
        Get scaler and encoder types for a version.

        Returns:
            Tuple of (scaler_type, encoder_type).
        """
        version_lower = version.lower()
        if version_lower not in cls.VERSIONS:
            raise ValueError(
                f"Unknown version '{version}'. " f"Available: {list(cls.VERSIONS.keys())}"
            )
        return cls.VERSIONS[version_lower]

    @classmethod
    def apply_version(
        cls, train: pd.DataFrame, test: pd.DataFrame, target: str, version: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply encoding version to train and test data.

        Returns:
            Tuple of (encoded_train, encoded_test).
        """
        scaler_type, encoder_type = cls.get_version_config(version)

        train_enc = train.copy()
        test_enc = test.copy()

        # Get column types
        num_cols = [
            col
            for col in train.select_dtypes(
                include=["int", "int32", "int64", "float", "float32", "float64"]
            ).columns
            if col != target
        ]
        cat_cols = [
            col
            for col in train.select_dtypes(include=["object", "category"]).columns
            if col != target
        ]

        # Apply scaler to numerical columns
        if num_cols:
            scaler = ScalerRegistry.get(scaler_type)
            scaler.fit(train_enc[num_cols])
            train_enc[num_cols] = scaler.transform(train_enc[num_cols])
            test_enc[num_cols] = scaler.transform(test_enc[num_cols])

        # Apply encoder to categorical columns
        if cat_cols:
            encoder = EncoderRegistry.get(encoder_type)
            encoder.fit(train_enc[cat_cols])
            train_enc = encoder.transform(train_enc)
            test_enc = encoder.transform(test_enc)

        return train_enc, test_enc

    @classmethod
    def get_all_versions(
        cls, train: pd.DataFrame, test: pd.DataFrame, target: str
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Apply all encoding versions and return results.

        Returns:
            Dictionary mapping version name to (train, test) tuples.
        """
        results = {}
        for version in cls.VERSIONS.keys():
            results[version] = cls.apply_version(train, test, target, version)
        return results

    @classmethod
    def describe_version(cls, version: str) -> str:
        """
        Get human-readable description of version.

        Returns:
            Description string.
        """
        version_lower = version.lower()
        return cls.VERSION_DESCRIPTIONS.get(version_lower, f"Unknown version: {version}")

    @classmethod
    def list_versions(cls) -> List[str]:
        """Get list of available versions."""
        return list(cls.VERSIONS.keys())


class EncodingVersion:
    """
    Helper class for managing encoding versions.

    Provides convenient interface for applying specific
    encoding combinations to datasets.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str):
        """
        Initialize EncodingVersion helper.
        """
        self.train = train
        self.test = test
        self.target = target

        # Identify column types
        self.cat_cols = [
            col
            for col in train.select_dtypes(include=["object", "category"]).columns
            if col != target
        ]
        self.num_cols = [
            col
            for col in train.select_dtypes(
                include=["int", "int32", "int64", "float", "float32", "float64"]
            ).columns
            if col != target
        ]

    def encoding_v1(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Version 1: StandardScaler + IFrequencyEncoder."""
        return EncodingVersionFactory.apply_version(self.train, self.test, self.target, "v1")

    def encoding_v2(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Version 2: MinMaxScaler + IFrequencyEncoder."""
        return EncodingVersionFactory.apply_version(self.train, self.test, self.target, "v2")

    def encoding_v3(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Version 3: StandardScaler + LabelEncoder."""
        return EncodingVersionFactory.apply_version(self.train, self.test, self.target, "v3")

    def encoding_v4(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Version 4: MinMaxScaler + LabelEncoder."""
        return EncodingVersionFactory.apply_version(self.train, self.test, self.target, "v4")

    def get_all(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get all encoding versions."""
        return EncodingVersionFactory.get_all_versions(self.train, self.test, self.target)
