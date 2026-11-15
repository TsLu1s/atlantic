"""Tests for atlantic.utils module: columns, datetime, validation."""

import numpy as np
import pandas as pd
import pytest

from atlantic.core.exceptions import ValidationError
from atlantic.utils.columns import (
    get_categorical_columns,
    get_columns_by_null_percentage,
    get_constant_columns,
    get_datetime_columns,
    get_high_cardinality_columns,
    get_numeric_columns,
    get_unique_columns,
    separate_columns_by_type,
)
from atlantic.utils.datetime import (
    detect_datetime_granularity,
    engineer_datetime_features,
    get_datetime_range,
)
from atlantic.utils.validation import (
    validate_column_exists,
    validate_columns_exist,
    validate_dataframe,
    validate_in_choices,
    validate_no_nulls,
    validate_numeric_columns,
    validate_numeric_range,
)


def test_validation_module_has_no_pandas_as_np_alias():
    """Regression test: ``utils/validation.py`` previously had
    ``import pandas as np`` which would silently alias pandas to ``np``
    and corrupt any future numpy use. The module must not expose ``np``
    bound to a pandas type."""
    import atlantic.utils.validation as v

    if hasattr(v, "np"):
        import numpy as numpy_real

        assert v.np is numpy_real, (
            "validation.np must be numpy, not pandas — guards against the "
            "previous 'import pandas as np' typo"
        )


# ============================================================================
# Column Utilities
# ============================================================================


class TestColumnUtils:
    def test_get_numeric_columns(self, sample_regression_df):
        nums = get_numeric_columns(sample_regression_df)
        assert "num_a" in nums
        assert "num_b" in nums
        assert "target" in nums  # no exclude
        assert "cat_x" not in nums

    def test_get_numeric_columns_exclude(self, sample_regression_df):
        nums = get_numeric_columns(sample_regression_df, exclude=["target"])
        assert "target" not in nums

    def test_get_categorical_columns(self, sample_regression_df):
        cats = get_categorical_columns(sample_regression_df)
        assert "cat_x" in cats
        assert "cat_y" in cats
        assert "num_a" not in cats

    def test_get_datetime_columns(self, datetime_df):
        dts = get_datetime_columns(datetime_df)
        assert "event_date" in dts

    def test_get_columns_by_null_percentage(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, None, None, None],  # 60% null
                "b": [1, 2, 3, 4, None],  # 20% null
                "c": [1, 2, 3, 4, 5],  # 0% null
            }
        )
        high_null = get_columns_by_null_percentage(df, threshold=0.5)
        assert "a" in high_null
        assert "b" not in high_null
        assert "c" not in high_null

    def test_get_constant_columns(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        assert get_constant_columns(df) == ["a"]

    def test_get_constant_columns_exclude(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        assert get_constant_columns(df, exclude=["a"]) == []

    def test_get_unique_columns(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [1, 1, 2]})
        assert "id" in get_unique_columns(df)
        assert "val" not in get_unique_columns(df)

    def test_get_high_cardinality_columns(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "low_card": np.random.choice(["A", "B"], 100),
                "high_card": [f"cat_{i}" for i in range(100)],
            }
        )
        high = get_high_cardinality_columns(df, threshold=50)
        assert "high_card" in high
        assert "low_card" not in high

    def test_separate_columns_by_type(self, datetime_df):
        result = separate_columns_by_type(datetime_df, target="target")
        assert "num_a" in result["numeric"]
        assert "target" not in result["numeric"]
        assert "event_date" in result["datetime"]


# ============================================================================
# Datetime Utilities
# ============================================================================


class TestDatetimeUtils:
    def test_engineer_datetime_features_drop(self, datetime_df):
        result = engineer_datetime_features(datetime_df, drop_original=True)
        assert "event_date" not in result.columns
        assert "event_date_year" in result.columns
        assert "event_date_month" in result.columns
        assert "event_date_hour" in result.columns
        assert "event_date_is_wknd" in result.columns

    def test_engineer_datetime_features_keep(self, datetime_df):
        result = engineer_datetime_features(datetime_df, drop_original=False)
        assert "event_date" in result.columns
        assert "event_date_year" in result.columns

    def test_engineer_specific_components(self, datetime_df):
        result = engineer_datetime_features(
            datetime_df, components=["year", "month"], drop_original=True
        )
        assert "event_date_year" in result.columns
        assert "event_date_month" in result.columns
        assert "event_date_hour" not in result.columns

    def test_engineer_specific_columns(self, datetime_df):
        result = engineer_datetime_features(
            datetime_df, columns=["event_date"], drop_original=True
        )
        assert "event_date" not in result.columns
        assert "event_date_year" in result.columns

    def test_no_datetime_columns(self, numeric_df):
        result = engineer_datetime_features(numeric_df)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_get_datetime_range(self, datetime_df):
        info = get_datetime_range(datetime_df, "event_date")
        assert "min" in info
        assert "max" in info
        assert "range_days" in info
        assert info["range_days"] >= 0

    def test_detect_datetime_granularity(self, datetime_df):
        granularity = detect_datetime_granularity(datetime_df, "event_date")
        assert granularity in ["second", "minute", "hour", "day", "month", "year"]


# ============================================================================
# Validation Utilities
# ============================================================================


class TestValidationUtils:
    def test_validate_dataframe_ok(self, sample_regression_df):
        validate_dataframe(sample_regression_df)

    def test_validate_dataframe_not_df(self):
        with pytest.raises(ValidationError):
            validate_dataframe([1, 2, 3])

    def test_validate_dataframe_empty(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_dataframe(pd.DataFrame())

    def test_validate_dataframe_empty_allowed(self):
        validate_dataframe(pd.DataFrame(), require_non_empty=False)

    def test_validate_column_exists_ok(self, sample_regression_df):
        validate_column_exists(sample_regression_df, "num_a")

    def test_validate_column_exists_missing(self, sample_regression_df):
        with pytest.raises(ValidationError, match="not found"):
            validate_column_exists(sample_regression_df, "nonexistent")

    def test_validate_columns_exist_ok(self, sample_regression_df):
        validate_columns_exist(sample_regression_df, ["num_a", "num_b"])

    def test_validate_columns_exist_missing(self, sample_regression_df):
        with pytest.raises(ValidationError):
            validate_columns_exist(sample_regression_df, ["num_a", "fake_col"])

    def test_validate_numeric_range_ok(self):
        validate_numeric_range(5, min_val=0, max_val=10)

    def test_validate_numeric_range_too_low(self):
        with pytest.raises(ValidationError):
            validate_numeric_range(-1, min_val=0)

    def test_validate_numeric_range_too_high(self):
        with pytest.raises(ValidationError):
            validate_numeric_range(100, max_val=50)

    def test_validate_in_choices_ok(self):
        validate_in_choices("a", ["a", "b", "c"])

    def test_validate_in_choices_bad(self):
        with pytest.raises(ValidationError):
            validate_in_choices("z", ["a", "b", "c"])

    def test_validate_no_nulls_ok(self, numeric_df):
        validate_no_nulls(numeric_df)

    def test_validate_no_nulls_fail(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [3, 4]})
        with pytest.raises(ValidationError, match="null"):
            validate_no_nulls(df)

    def test_validate_numeric_columns_ok(self, numeric_df):
        validate_numeric_columns(numeric_df, ["a", "b"])

    def test_validate_numeric_columns_fail(self, sample_regression_df):
        with pytest.raises(ValidationError, match="numeric"):
            validate_numeric_columns(sample_regression_df, ["cat_x"])
