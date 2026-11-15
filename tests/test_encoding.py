"""Tests for atlantic.encoding module: versions and optimizer."""

import numpy as np
import pandas as pd
import pytest

from atlantic.encoding.versions import EncodingVersion, EncodingVersionFactory


# ============================================================================
# EncodingVersionFactory
# ============================================================================


class TestEncodingVersionFactory:
    @pytest.fixture
    def mixed_train_test(self):
        np.random.seed(42)
        n = 100
        train = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "num_b": np.random.rand(n) * 10,
                "cat_x": np.random.choice(["A", "B", "C"], n),
                "target": np.random.randn(n),
            }
        )
        test = pd.DataFrame(
            {
                "num_a": np.random.randn(30),
                "num_b": np.random.rand(30) * 10,
                "cat_x": np.random.choice(["A", "B", "C"], 30),
                "target": np.random.randn(30),
            }
        )
        return train, test

    def test_list_versions(self):
        versions = EncodingVersionFactory.list_versions()
        assert set(versions) == {"v1", "v2", "v3", "v4"}

    def test_get_version_config(self):
        scaler, encoder = EncodingVersionFactory.get_version_config("v1")
        assert scaler == "standard"
        assert encoder == "ifrequency"

    def test_get_version_config_v4(self):
        scaler, encoder = EncodingVersionFactory.get_version_config("v4")
        assert scaler == "minmax"
        assert encoder == "label"

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="Unknown version"):
            EncodingVersionFactory.get_version_config("v99")

    def test_describe_version(self):
        desc = EncodingVersionFactory.describe_version("v1")
        assert "StandardScaler" in desc
        assert "IFrequencyEncoder" in desc

    def test_apply_version_v1(self, mixed_train_test):
        train, test = mixed_train_test
        train_enc, test_enc = EncodingVersionFactory.apply_version(train, test, "target", "v1")
        assert train_enc.shape[0] == train.shape[0]
        assert test_enc.shape[0] == test.shape[0]
        assert "cat_x" in train_enc.columns  # encoded but col name kept
        # Numeric columns should be scaled
        assert train_enc["num_a"].std() != train["num_a"].std()

    def test_apply_version_v2(self, mixed_train_test):
        train, test = mixed_train_test
        train_enc, test_enc = EncodingVersionFactory.apply_version(train, test, "target", "v2")
        assert train_enc.shape[0] == train.shape[0]

    def test_apply_version_v3(self, mixed_train_test):
        train, test = mixed_train_test
        train_enc, test_enc = EncodingVersionFactory.apply_version(train, test, "target", "v3")
        assert train_enc.shape[0] == train.shape[0]

    def test_apply_version_v4(self, mixed_train_test):
        train, test = mixed_train_test
        train_enc, test_enc = EncodingVersionFactory.apply_version(train, test, "target", "v4")
        assert train_enc.shape[0] == train.shape[0]

    def test_get_all_versions(self, mixed_train_test):
        train, test = mixed_train_test
        results = EncodingVersionFactory.get_all_versions(train, test, "target")
        assert len(results) == 4
        assert all(k in results for k in ["v1", "v2", "v3", "v4"])

    def test_target_not_encoded(self, mixed_train_test):
        train, test = mixed_train_test
        train_enc, _ = EncodingVersionFactory.apply_version(train, test, "target", "v1")
        # Target column should remain unchanged
        pd.testing.assert_series_equal(
            train_enc["target"], train["target"], check_names=True
        )

    def test_no_categorical_columns(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [4.0, 5.0, 6.0]})
        test = pd.DataFrame({"a": [7.0, 8.0], "target": [9.0, 10.0]})
        train_enc, test_enc = EncodingVersionFactory.apply_version(train, test, "target", "v1")
        assert train_enc.shape == train.shape

    def test_no_numeric_columns(self):
        train = pd.DataFrame({"cat": ["A", "B", "C"], "target": [1.0, 2.0, 3.0]})
        test = pd.DataFrame({"cat": ["A", "B"], "target": [4.0, 5.0]})
        train_enc, test_enc = EncodingVersionFactory.apply_version(train, test, "target", "v1")
        assert train_enc.shape[0] == train.shape[0]


# ============================================================================
# EncodingVersion Helper
# ============================================================================


class TestEncodingVersion:
    @pytest.fixture
    def ev(self):
        np.random.seed(42)
        n = 80
        train = pd.DataFrame(
            {
                "num_a": np.random.randn(n),
                "cat_x": np.random.choice(["X", "Y", "Z"], n),
                "target": np.random.randn(n),
            }
        )
        test = pd.DataFrame(
            {
                "num_a": np.random.randn(20),
                "cat_x": np.random.choice(["X", "Y", "Z"], 20),
                "target": np.random.randn(20),
            }
        )
        return EncodingVersion(train, test, "target")

    def test_encoding_v1(self, ev):
        train_enc, test_enc = ev.encoding_v1()
        assert train_enc.shape[0] == 80
        assert test_enc.shape[0] == 20

    def test_encoding_v2(self, ev):
        train_enc, test_enc = ev.encoding_v2()
        assert train_enc.shape[0] == 80

    def test_encoding_v3(self, ev):
        train_enc, test_enc = ev.encoding_v3()
        assert train_enc.shape[0] == 80

    def test_encoding_v4(self, ev):
        train_enc, test_enc = ev.encoding_v4()
        assert train_enc.shape[0] == 80

    def test_get_all(self, ev):
        results = ev.get_all()
        assert len(results) == 4

    def test_column_types_detected(self, ev):
        assert "cat_x" in ev.cat_cols
        assert "num_a" in ev.num_cols
        assert "target" not in ev.cat_cols
        assert "target" not in ev.num_cols
