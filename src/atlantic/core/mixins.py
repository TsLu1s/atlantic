from __future__ import annotations

from typing import List, TYPE_CHECKING

import pandas as pd

from atlantic.core.exceptions import ValidationError

if TYPE_CHECKING:
    pass


class DataValidationMixin:
    """Mixin for DataFrame validation."""
    
    target: str  # Expected to be defined in the implementing class
    
    def _validate_dataframe(
        self, 
        X: pd.DataFrame, 
        require_target: bool = True
    ) -> None:
        """
        Validate input DataFrame.
        
        Args:
            X: Input DataFrame to validate.
            require_target: Whether to require target column presence.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValidationError("Input DataFrame is empty")
        
        if require_target and hasattr(self, 'target') and self.target:
            if self.target not in X.columns:
                raise ValidationError(
                    f"Target column '{self.target}' not found in DataFrame. "
                    f"Available columns: {list(X.columns)}"
                )
    
    def _validate_split_ratio(self, ratio: float) -> None:
        """
        Validate split ratio value.

        Raises:
            ValidationError: If ratio is out of valid range.
        """
        if not 0.5 <= ratio <= 0.98:
            raise ValidationError(
                f"split_ratio must be between 0.5 and 0.98, got {ratio}"
            )
    
    def _validate_columns_exist(
        self, 
        X: pd.DataFrame, 
        columns: List[str]
    ) -> None:
        """
        Validate that specified columns exist in DataFrame.

        Raises:
            ValidationError: If any column is missing.
        """
        missing = set(columns) - set(X.columns)
        if missing:
            raise ValidationError(
                f"Columns not found in DataFrame: {missing}"
            )


class ColumnTypeMixin:
    """Mixin for column type detection."""
    
    NUMERIC_TYPES: List[str] = [
        'int', 'int8', 'int16', 'int32', 'int64',
        'float', 'float16', 'float32', 'float64'
    ]
    CATEGORICAL_TYPES: List[str] = ['object', 'category']
    DATETIME_TYPES: List[str] = ['datetime64[ns]', 'datetime64']
    
    target: str  # Expected to be defined in the implementing class
    
    def _get_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of numeric columns, excluding target.
        
        Args:
            X: Input DataFrame.
            
        Returns:
            List of numeric column names.
        """
        target = getattr(self, 'target', None)
        return [
            col for col in X.select_dtypes(include=self.NUMERIC_TYPES).columns
            if col != target
        ]
    
    def _get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of categorical columns, excluding target.

        Returns:
            List of categorical column names.
        """
        target = getattr(self, 'target', None)
        return [
            col for col in X.select_dtypes(include=self.CATEGORICAL_TYPES).columns
            if col != target
        ]
    
    def _get_datetime_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of datetime columns.

        Returns:
            List of datetime column names.
        """
        return X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    def _get_all_feature_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get all feature columns (excluding target).

        Returns:
            List of all feature column names.
        """
        target = getattr(self, 'target', None)
        return [col for col in X.columns if col != target]


class DateEngineeringMixin:
    """Mixin for datetime feature engineering."""
    
    DATE_COMPONENTS: List[str] = [
        'day_of_month', 'day_of_week', 'is_wknd', 'month',
        'day_of_year', 'year', 'hour', 'minute', 'second'
    ]
    
    def _engineer_datetime_features(
        self, 
        X: pd.DataFrame, 
        drop: bool = True
    ) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.

        Returns:
            DataFrame with engineered datetime features.
        """
        X = X.copy()
        datetime_cols = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        for col in datetime_cols:
            # Standardize datetime format
            X[col] = pd.to_datetime(X[col])
            
            # Extract components
            X[f'{col}_day_of_month'] = X[col].dt.day
            X[f'{col}_day_of_week'] = X[col].dt.dayofweek + 1
            X[f'{col}_is_wknd'] = X[f'{col}_day_of_week'].isin([6, 7]).astype(int)
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day_of_year'] = X[col].dt.dayofyear
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_hour'] = X[col].dt.hour
            X[f'{col}_minute'] = X[col].dt.minute
            X[f'{col}_second'] = X[col].dt.second
            
            if drop:
                X = X.drop(columns=[col])
        
        return X


class SchemaValidationMixin:
    """Mixin for input/output schema validation."""
    
    _input_schema: dict = None
    
    def _store_input_schema(self, X: pd.DataFrame) -> None:
        """
        Store input schema during fit.
        
        Args:
            X: Input DataFrame to capture schema from.
        """
        self._input_schema = {
            'columns': list(X.columns),
            'dtypes': {col: str(dtype) for col, dtype in X.dtypes.items()},
            'n_features': len(X.columns)
        }
    
    def _validate_input_schema(
        self, 
        X: pd.DataFrame, 
        strict: bool = False
    ) -> None:
        """
        Validate input schema against stored schema.

        Raises:
            ValidationError: If schema validation fails.
        """
        if self._input_schema is None:
            raise ValidationError("No input schema stored. Call fit first.")
        
        expected_cols = set(self._input_schema['columns'])
        actual_cols = set(X.columns)
        
        if strict:
            if expected_cols != actual_cols:
                missing = expected_cols - actual_cols
                extra = actual_cols - expected_cols
                raise ValidationError(
                    f"Column mismatch. Missing: {missing}, Extra: {extra}"
                )
        else:
            missing = expected_cols - actual_cols
            if missing:
                raise ValidationError(
                    f"Missing required columns: {missing}"
                )


class TargetTypeMixin:
    """Mixin for target variable type detection."""
    
    NUMERIC_TYPES: List[str] = [
        'int', 'int8', 'int16', 'int32', 'int64',
        'float', 'float16', 'float32', 'float64'
    ]
    
    target: str
    pred_type: str = None
    n_classes: int = None
    eval_metric: str = None
    
    def _detect_target_type(self, X: pd.DataFrame) -> tuple:
        """
        Detect target variable type and set appropriate metric.

        Returns:
            Tuple of (prediction_type, evaluation_metric).
        """
        if self.target not in X.columns:
            raise ValidationError(f"Target column '{self.target}' not found")
        
        target_dtype = X[self.target].dtype
        
        if target_dtype in self.NUMERIC_TYPES:
            self.pred_type = 'Reg'
            self.eval_metric = 'Mean Absolute Error'
            self.n_classes = None
        else:
            self.pred_type = 'Class'
            self.n_classes = X[self.target].nunique()
            self.eval_metric = 'F1' if self.n_classes > 2 else 'Precision'
        
        return self.pred_type, self.eval_metric
