"""
Atlantic - Detailed Preprocessing Components Example.

Demonstrates individual usage of all preprocessing components:
- Encoders (Label, IFrequency, OneHot)
- Scalers (Standard, MinMax, Robust)
- Imputers (Simple, KNN, Iterative)
- Feature Selectors (H2O, VIF)
- Date Engineering
"""

import numpy as np
from atlantic.data import DatasetGenerator
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning)

# =============================================================================
# DATA PREPARATION
# =============================================================================

print("=" * 70)
print("DETAILED PREPROCESSING COMPONENTS EXAMPLE")
print("=" * 70)

# Generate dataset with mixed types
data, target = DatasetGenerator.generate_with_datetime(
    n_samples=1000,
    n_numeric=8,
    n_categorical=4,
    task_type="classification",
    null_percentage=0.10,
    random_state=42
)

train, test = train_test_split(data, train_size=0.8, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print(f"\nDataset Shape: {data.shape}")
print(f"Target: {target}")
print("\nColumn Types:")
print(f"  Numeric: {train.select_dtypes(include=['int', 'float']).columns.tolist()[:5]}...")
print(f"  Categorical: {train.select_dtypes(include=['object', 'category']).columns.tolist()}")
print(f"  Datetime: {train.select_dtypes(include=['datetime64']).columns.tolist()}")

# =============================================================================
# 1. DATE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("1. DATE ENGINEERING")
print("=" * 70)

from atlantic.utils.datetime import engineer_datetime_features, DATE_COMPONENTS

print(f"\nAvailable Date Components: {DATE_COMPONENTS}")

# Apply date engineering to BOTH train and test
train_dates = engineer_datetime_features(
    train.copy(),
    drop_original=True,
    components=['day_of_week', 'month', 'year', 'is_wknd']
)
test_dates = engineer_datetime_features(
    test.copy(),
    drop_original=True,
    components=['day_of_week', 'month', 'year', 'is_wknd']
)

print(f"\nBefore Date Engineering: {train.shape}")
print(f"After Date Engineering: {train_dates.shape}")

# Show generated columns
date_generated = [col for col in train_dates.columns if any(
    comp in col for comp in ['day_of_week', 'month', 'year', 'is_wknd']
)]
print(f"Generated Date Features: {date_generated}")

# Full date engineering (all components) - BOTH train and test
train_full_dates = engineer_datetime_features(train.copy(), drop_original=True)
test_full_dates = engineer_datetime_features(test.copy(), drop_original=True)
print(f"With All Components - Train: {train_full_dates.shape}, Test: {test_full_dates.shape}")

# =============================================================================
# 2. ENCODERS
# =============================================================================

print("\n" + "=" * 70)
print("2. ENCODERS")
print("=" * 70)

from atlantic.preprocessing import (
    AutoLabelEncoder,
    AutoIFrequencyEncoder,
    AutoOneHotEncoder
)
from atlantic.utils.columns import get_categorical_columns

# Get categorical columns (excluding target)
cat_cols = get_categorical_columns(train_full_dates, exclude=[target])
print(f"\nCategorical Columns: {cat_cols}")

# --- Label Encoder ---
print("\n--- Label Encoder ---")
label_encoder = AutoLabelEncoder()
label_encoder.fit(train_full_dates[cat_cols])

train_label = train_full_dates.copy()
test_label = test_full_dates.copy()
train_label[cat_cols] = label_encoder.transform(train_full_dates[cat_cols])
test_label[cat_cols] = label_encoder.transform(test_full_dates[cat_cols])

print(f"Original Categories (first col): {train_full_dates[cat_cols[0]].unique()[:5]}")
print(f"Encoded Values (first col): {train_label[cat_cols[0]].unique()[:5]}")

# Inverse transform
train_inverse = label_encoder.inverse_transform(train_label[cat_cols])
print(f"Inverse Transform: {train_inverse[cat_cols[0]].unique()[:5]}")

# --- IFrequency Encoder ---
print("\n--- IFrequency Encoder ---")
ifreq_encoder = AutoIFrequencyEncoder()
ifreq_encoder.fit(train_full_dates[cat_cols])

train_ifreq = train_full_dates.copy()
test_ifreq = test_full_dates.copy()
train_ifreq[cat_cols] = ifreq_encoder.transform(train_full_dates[cat_cols])
test_ifreq[cat_cols] = ifreq_encoder.transform(test_full_dates[cat_cols])
print(f"IFrequency Encoded (first col sample): {train_ifreq[cat_cols[0]].head()}")

# --- OneHot Encoder ---
print("\n--- OneHot Encoder ---")
onehot_encoder = AutoOneHotEncoder()
onehot_encoder.fit(train_full_dates[cat_cols])

train_onehot = onehot_encoder.transform(train_full_dates[cat_cols])
test_onehot = onehot_encoder.transform(test_full_dates[cat_cols])
print(f"Before OneHot: {train_full_dates.shape}")
print(f"After OneHot - Train: {train_onehot.shape}, Test: {test_onehot.shape}")

# =============================================================================
# 3. SCALERS
# =============================================================================

print("\n" + "=" * 70)
print("3. SCALERS")
print("=" * 70)

from atlantic.preprocessing import (
    AutoMinMaxScaler,
    AutoStandardScaler,
    AutoRobustScaler
)
from atlantic.utils.columns import get_numeric_columns

# Get numeric columns
num_cols = get_numeric_columns(train_label, exclude=[target])
print(f"\nNumeric Columns: {num_cols[:5]}...")

# --- Standard Scaler ---
print("\n--- Standard Scaler (zero mean, unit variance) ---")
std_scaler = AutoStandardScaler()
std_scaler.fit(train_label[num_cols])

train_std = std_scaler.transform(train_label[num_cols])
test_std = std_scaler.transform(test_label[num_cols])
print(f"Original Mean: {train_label[num_cols[0]].mean():.4f}")
print(f"Scaled Mean: {train_std[num_cols[0]].mean():.4f}")
print(f"Original Std: {train_label[num_cols[0]].std():.4f}")
print(f"Scaled Std: {train_std[num_cols[0]].std():.4f}")

# Inverse transform
train_std_inverse = std_scaler.inverse_transform(train_std)
print(f"Inverse Mean: {train_std_inverse[num_cols[0]].mean():.4f}")

# --- MinMax Scaler ---
print("\n--- MinMax Scaler (0 to 1 range) ---")
minmax_scaler = AutoMinMaxScaler()
minmax_scaler.fit(train_label[num_cols])

train_minmax = minmax_scaler.transform(train_label[num_cols])
test_minmax = minmax_scaler.transform(test_label[num_cols])
print(f"Original Range: [{train_label[num_cols[0]].min():.2f}, {train_label[num_cols[0]].max():.2f}]")
print(f"Scaled Range: [{train_minmax[num_cols[0]].min():.2f}, {train_minmax[num_cols[0]].max():.2f}]")

# --- Robust Scaler ---
print("\n--- Robust Scaler (median and IQR, outlier-resistant) ---")
robust_scaler = AutoRobustScaler()
robust_scaler.fit(train_label[num_cols])

train_robust = robust_scaler.transform(train_label[num_cols])
test_robust = robust_scaler.transform(test_label[num_cols])
print(f"Original Median: {train_label[num_cols[0]].median():.4f}")
print(f"Scaled Median: {train_robust[num_cols[0]].median():.4f}")

# =============================================================================
# 4. IMPUTERS
# =============================================================================

print("\n" + "=" * 70)
print("4. IMPUTERS")
print("=" * 70)

from atlantic.preprocessing import (
    AutoSimpleImputer,
    AutoKNNImputer,
    AutoIterativeImputer
)

# Check missing values
print("\nMissing Values Before Imputation:")
print(f"  Train Total: {train_label.select_dtypes(include=['float', 'int']).isna().sum().sum()}")
print(f"  Test Total: {test_label.select_dtypes(include=['float', 'int']).isna().sum().sum()}")

# --- Simple Imputer ---
print("\n--- Simple Imputer (mean strategy) ---")
simple_imputer = AutoSimpleImputer(strategy='mean', target=target)
simple_imputer.fit(train_label)

train_simple = simple_imputer.transform(train_label.copy())
test_simple = simple_imputer.transform(test_label.copy())
print(f"Missing After (Train): {train_simple.select_dtypes(include=['float', 'int']).isna().sum().sum()}")
print(f"Missing After (Test): {test_simple.select_dtypes(include=['float', 'int']).isna().sum().sum()}")

# --- KNN Imputer ---
print("\n--- KNN Imputer (3 neighbors) ---")
knn_imputer = AutoKNNImputer(n_neighbors=3, weights="uniform", target=target)
knn_imputer.fit(train_label)

train_knn = knn_imputer.transform(train_label.copy())
test_knn = knn_imputer.transform(test_label.copy())
print(f"Missing After (Train): {train_knn.select_dtypes(include=['float', 'int']).isna().sum().sum()}")
print(f"Missing After (Test): {test_knn.select_dtypes(include=['float', 'int']).isna().sum().sum()}")

# --- Iterative Imputer ---
print("\n--- Iterative Imputer (multivariate) ---")
iter_imputer = AutoIterativeImputer(
    max_iter=10,
    random_state=42,
    initial_strategy="mean",
    imputation_order="ascending",
    target=target
)
iter_imputer.fit(train_label)

train_iter = iter_imputer.transform(train_label.copy())
test_iter = iter_imputer.transform(test_label.copy())
print(f"Missing After (Train): {train_iter.select_dtypes(include=['float', 'int']).isna().sum().sum()}")
print(f"Missing After (Test): {test_iter.select_dtypes(include=['float', 'int']).isna().sum().sum()}")

# =============================================================================
# 5. FEATURE SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("5. FEATURE SELECTION")
print("=" * 70)

from atlantic.feature_selection import VIFFeatureSelector

# Prepare data for VIF (needs numeric, no nulls)
train_for_vif = train_simple.copy()
test_for_vif = test_simple.copy()

# --- VIF Feature Selection ---
print("\n--- VIF Feature Selection ---")
vif_selector = VIFFeatureSelector(target=target, vif_threshold=10.0)

try:
    vif_selector.fit(train_for_vif)
    print(f"Original Features: {len(num_cols)}")
    print(f"Selected Features: {vif_selector.n_selected}")
    print(f"Features Removed: {vif_selector.n_removed}")
    print(f"Selected: {vif_selector.selected_features[:5]}...")
    
    # Transform both train and test
    train_vif = vif_selector.transform(train_for_vif)
    test_vif = vif_selector.transform(test_for_vif)
    print(f"After VIF - Train: {train_vif.shape}, Test: {test_vif.shape}")
    
    if vif_selector.vif_dataframe is not None:
        print("\nVIF Values (top 5):")
        print(vif_selector.vif_dataframe.head())
except Exception as e:
    print(f"VIF Selection Error: {e}")

# =============================================================================
# 6. USING REGISTRIES
# =============================================================================

print("\n" + "=" * 70)
print("6. USING COMPONENT REGISTRIES")
print("=" * 70)

from atlantic.preprocessing.registry import (
    EncoderRegistry,
    ScalerRegistry,
    ImputerRegistry
)

print("\nAvailable Components:")
print(f"  Encoders: {EncoderRegistry.list_available()}")
print(f"  Scalers: {ScalerRegistry.list_available()}")
print(f"  Imputers: {ImputerRegistry.list_available()}")

# Get components from registry
encoder = EncoderRegistry.get("ifrequency")
scaler = ScalerRegistry.get("standard")
imputer = ImputerRegistry.get("knn", n_neighbors=5, target=target)

print("\nCreated from Registry:")
print(f"  Encoder: {type(encoder).__name__}")
print(f"  Scaler: {type(scaler).__name__}")
print(f"  Imputer: {type(imputer).__name__}")

# =============================================================================
# 7. ENCODING VERSION FACTORY
# =============================================================================

print("\n" + "=" * 70)
print("7. ENCODING VERSIONS")
print("=" * 70)

from atlantic.encoding import EncodingVersionFactory, EncodingVersion

print("\nAvailable Encoding Versions:")
for version in EncodingVersionFactory.list_versions():
    desc = EncodingVersionFactory.describe_version(version)
    print(f"  {version.upper()}: {desc}")

# Apply specific version
ev = EncodingVersion(train=train_simple.copy(), test=test_simple.copy(), target=target)

print("\nApplying Encoding Version 1 (Standard + IFrequency):")
train_v1, test_v1 = ev.encoding_v1()
print(f"  Train Shape: {train_v1.shape}, Test Shape: {test_v1.shape}")

print("\nApplying Encoding Version 4 (MinMax + Label):")
train_v4, test_v4 = ev.encoding_v4()
print(f"  Train Shape: {train_v4.shape}, Test Shape: {test_v4.shape}")

# =============================================================================
# 8. METRICS
# =============================================================================

print("\n" + "=" * 70)
print("8. EVALUATION METRICS")
print("=" * 70)

from atlantic.evaluation import MetricRegistry, metrics_classification
from atlantic.core.enums import TaskType

print("\nAvailable Metrics:")
print(f"  Regression: {MetricRegistry.list_available(TaskType.REGRESSION)}")
print(f"  Classification: {MetricRegistry.list_available(TaskType.CLASSIFICATION)}")

# Example metric usage
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1])

cls_metrics = metrics_classification(y_true, y_pred, n_classes=2)
print("\nClassification Metrics Example:")
print(cls_metrics)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPONENTS SUMMARY")
print("=" * 70)
print("""
Components Available:
- Encoders: AutoLabelEncoder, AutoIFrequencyEncoder, AutoOneHotEncoder
- Scalers: AutoStandardScaler, AutoMinMaxScaler, AutoRobustScaler  
- Imputers: AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer
- Feature Selectors: H2OFeatureSelector, VIFFeatureSelector

Utilities:
- Date Engineering: engineer_datetime_features()
- Column Detection: get_numeric_columns(), get_categorical_columns()
- Registries: EncoderRegistry, ScalerRegistry, ImputerRegistry
- Encoding Versions: EncodingVersionFactory (V1-V4)
""")





























