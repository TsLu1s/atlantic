from __future__ import annotations

from typing import List, Optional

import pandas as pd


DATE_COMPONENTS = [
    'day_of_month', 'day_of_week', 'is_wknd', 'month',
    'day_of_year', 'year', 'hour', 'minute', 'second'
]


def engineer_datetime_features(
    X: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_original: bool = True,
    components: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract temporal features from datetime columns.

    Returns:
        DataFrame with engineered datetime features.
    """
    X = X.copy()
    
    if columns is None:
        columns = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    if components is None:
        components = DATE_COMPONENTS
    
    for col in columns:
        if col not in X.columns:
            continue
        
        # Ensure datetime type
        X[col] = pd.to_datetime(X[col])
        
        # Extract requested components
        if 'day_of_month' in components:
            X[f'{col}_day_of_month'] = X[col].dt.day
        
        if 'day_of_week' in components:
            X[f'{col}_day_of_week'] = X[col].dt.dayofweek + 1
        
        if 'is_wknd' in components:
            dow = X[col].dt.dayofweek + 1
            X[f'{col}_is_wknd'] = dow.isin([6, 7]).astype(int)
        
        if 'month' in components:
            X[f'{col}_month'] = X[col].dt.month
        
        if 'day_of_year' in components:
            X[f'{col}_day_of_year'] = X[col].dt.dayofyear
        
        if 'year' in components:
            X[f'{col}_year'] = X[col].dt.year
        
        if 'hour' in components:
            X[f'{col}_hour'] = X[col].dt.hour
        
        if 'minute' in components:
            X[f'{col}_minute'] = X[col].dt.minute
        
        if 'second' in components:
            X[f'{col}_second'] = X[col].dt.second
        
        if drop_original:
            X = X.drop(columns=[col])
    
    return X


def get_datetime_range(X: pd.DataFrame, column: str) -> dict:
    """
    Get datetime range information for a column.

    Returns:
        Dictionary with min, max, and range information.
    """
    col_data = pd.to_datetime(X[column])
    
    return {
        'min': col_data.min(),
        'max': col_data.max(),
        'range_days': (col_data.max() - col_data.min()).days,
        'n_unique': col_data.nunique()
    }


def detect_datetime_granularity(X: pd.DataFrame, column: str) -> str:
    """
    Detect the granularity of a datetime column.

    Returns:
        Granularity string ('second', 'minute', 'hour', 'day', 'month', 'year').
    """
    col_data = pd.to_datetime(X[column])
    
    # Check from finest to coarsest granularity
    if col_data.dt.second.nunique() > 1:
        return 'second'
    if col_data.dt.minute.nunique() > 1:
        return 'minute'
    if col_data.dt.hour.nunique() > 1:
        return 'hour'
    if col_data.dt.day.nunique() > 1:
        return 'day'
    if col_data.dt.month.nunique() > 1:
        return 'month'
    
    return 'year'