"""
Atlantic - Complete Builder Pattern Example.

Demonstrates all builder configurations and customization options
for different use cases and scenarios.
"""

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning)

from atlantic.data import DatasetGenerator
from atlantic.pipeline import AtlanticBuilder

print("=" * 70)
print("COMPLETE BUILDER PATTERN EXAMPLE")
print("=" * 70)

# =============================================================================
# DATA PREPARATION
# =============================================================================

# Generate various datasets for different scenarios
data_cls, target_cls = DatasetGenerator.generate_classification(
    n_samples=1000, n_features=15, n_classes=2, n_categorical=4, 
    null_percentage=0.08, random_state=42
)

data_reg, target_reg = DatasetGenerator.generate_regression(
    n_samples=1000, n_features=15, n_categorical=4,
    null_percentage=0.08, random_state=42
)

data_dt, target_dt = DatasetGenerator.generate_with_datetime(
    n_samples=1000, n_numeric=10, n_categorical=3, 
    task_type="regression", null_percentage=0.05, random_state=42
)

data_hn, target_hn = DatasetGenerator.generate_high_null(
    n_samples=1000, null_percentage=0.25, task_type="classification", random_state=42
)

data_hc, target_hc = DatasetGenerator.generate_high_cardinality(
    n_samples=1000, n_categorical=5, cardinality_range=(50, 150),
    task_type="classification", random_state=42
)

# =============================================================================
# BUILDER CONFIGURATION 1: DEFAULT (BALANCED)
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 1: DEFAULT BALANCED SETTINGS")
print("=" * 70)

default_builder = AtlanticBuilder()

# Get default configuration
default_config = default_builder.get_config()
print("\nDefault Configuration:")
print(f"  Date Engineering: {default_config.date_engineering}")
print(f"  Null Removal Threshold: {default_config.null_removal_threshold}")
print(f"  Feature Selection: {default_config.feature_selection.method}")
print(f"  H2O Relevance: {default_config.feature_selection.h2o_relevance}")
print(f"  H2O Models: {default_config.feature_selection.h2o_max_models}")
print(f"  Encoding FS: {default_config.feature_selection.encoding_for_fs}")
print(f"  VIF Threshold: {default_config.feature_selection.vif_threshold}")
print(f"  Scaler: {default_config.encoding.scaler}")
print(f"  Encoder: {default_config.encoding.encoder}")
print(f"  Auto-Select Encoding: {default_config.encoding.auto_select}")
print(f"  Imputer: {default_config.imputation.method}")
print(f"  Auto-Select Imputation: {default_config.imputation.auto_select}")
print(f"  optimization {default_config.optimizer.optimization_level}")
print(f"  Random State: {default_config.optimizer.random_state}")

# =============================================================================
# BUILDER CONFIGURATION 2: FAST PROTOTYPING
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 2: FAST PROTOTYPING")
print("=" * 70)

fast_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.90)
    .with_feature_selection(
        method="h2o",
        relevance=0.85,         # Lower threshold = faster, fewer features
        h2o_models=3,           # Minimal models
        encoding_fs=True
    )
    .with_encoding(
        scaler="minmax",
        encoder="label",        # Fastest encoder
        auto_select=False       # Skip auto-selection
    )
    .with_imputation(
        method="simple",        # Fastest imputer
        auto_select=False
    )
    .with_vif_filtering(threshold=15.0)  # Relaxed threshold
    .with_optimization(
        optimization_level="fast",
        random_state=42
    )
    .build()
)

print("Fast Prototyping Configuration:")
print(f"  Feature Selection: {fast_pipeline.config.feature_selection.h2o_max_models} models")
print(f"  Encoding: {fast_pipeline.config.encoding.scaler} + {fast_pipeline.config.encoding.encoder}")
print(f"  Imputation: {fast_pipeline.config.imputation.method}")
print(f"  optimization {fast_pipeline.config.optimizer.optimization_level}")

# =============================================================================
# BUILDER CONFIGURATION 3: THOROUGH OPTIMIZATION
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 3: THOROUGH OPTIMIZATION (Best Results)")
print("=" * 70)

thorough_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.99)
    .with_feature_selection(
        method="h2o",
        relevance=0.98,         # High threshold = more features
        h2o_models=15,          # Many models for better importance
        encoding_fs=True
    )
    .with_encoding(
        scaler="standard",
        encoder="ifrequency",   # Best encoder for most cases
        auto_select=True        # Auto-select best version
    )
    .with_imputation(
        method="iterative",     # Best imputer
        auto_select=True        # Auto-select best method
    )
    .with_vif_filtering(threshold=8.0)  # Strict multicollinearity
    .with_optimization(
        optimization_level="thorough",
        random_state=42
    )
    .build()
)

print("Thorough Optimization Configuration:")
print(f"  Feature Selection: {thorough_pipeline.config.feature_selection.h2o_max_models} models")
print(f"  Relevance: {thorough_pipeline.config.feature_selection.h2o_relevance}")
print(f"  Auto-Select Encoding: {thorough_pipeline.config.encoding.auto_select}")
print(f"  Auto-Select Imputation: {thorough_pipeline.config.imputation.auto_select}")
print(f"  VIF Threshold: {thorough_pipeline.config.feature_selection.vif_threshold}")

# =============================================================================
# BUILDER CONFIGURATION 4: HIGH-NULL DATA
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 4: HIGH-NULL DATA HANDLING")
print("=" * 70)

high_null_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.80)  # More aggressive null column removal
    .with_feature_selection(
        method="h2o",
        relevance=0.90,
        h2o_models=5,
        encoding_fs=True
    )
    .with_encoding(
        scaler="robust",        # Robust to outliers from imputation
        encoder="ifrequency",
        auto_select=True
    )
    .with_imputation(
        method="iterative",     # Best for high-null scenarios
        auto_select=True        # Let system choose best
    )
    .with_vif_filtering(threshold=12.0)
    .with_optimization(
        optimization_level="balanced",
        random_state=42
    )
    .build()
)

print("High-Null Data Configuration:")
print(f"  Null Removal: {high_null_pipeline.config.null_removal_threshold}")
print(f"  Scaler: {high_null_pipeline.config.encoding.scaler}")
print(f"  Imputation: {high_null_pipeline.config.imputation.method}")

# =============================================================================
# BUILDER CONFIGURATION 5: HIGH-CARDINALITY CATEGORICAL
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 5: HIGH-CARDINALITY CATEGORICAL DATA")
print("=" * 70)

high_card_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.99)
    .with_feature_selection(
        method="h2o",
        relevance=0.95,
        h2o_models=7,
        encoding_fs=True        # Important: encode before selection
    )
    .with_encoding(
        scaler="standard",
        encoder="ifrequency",   # Best for high-cardinality
        auto_select=True
    )
    .with_imputation(
        method="knn",
        auto_select=True
    )
    .with_vif_filtering(threshold=10.0)
    .with_optimization(
        optimization_level="balanced",
        random_state=42
    )
    .build()
)

print("High-Cardinality Configuration:")
print(f"  Encoder: {high_card_pipeline.config.encoding.encoder}")
print(f"  Encoding FS: {high_card_pipeline.config.feature_selection.encoding_for_fs}")

# =============================================================================
# BUILDER CONFIGURATION 6: NO H2O FEATURE SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 6: NO H2O FEATURE SELECTION (VIF Only)")
print("=" * 70)

no_h2o_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.99)
    .with_feature_selection(method="none")  # Skip H2O
    .with_encoding(
        scaler="standard",
        encoder="label",
        auto_select=True
    )
    .with_imputation(
        method="simple",
        auto_select=True
    )
    .with_vif_filtering(threshold=10.0)     # Only VIF filtering
    .with_optimization(
        optimization_level="fast",
        random_state=42
    )
    .build()
)

print("No H2O Configuration:")
print(f"  Feature Selection: {no_h2o_pipeline.config.feature_selection.method}")
print(f"  VIF Filtering: {no_h2o_pipeline.config.feature_selection.vif_threshold}")

# =============================================================================
# BUILDER CONFIGURATION 7: REGRESSION OPTIMIZED
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 7: REGRESSION OPTIMIZED")
print("=" * 70)

regression_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.98)
    .with_feature_selection(
        method="h2o",
        relevance=0.97,         # Keep more features for regression
        h2o_models=10,
        encoding_fs=True
    )
    .with_encoding(
        scaler="robust",        # Robust scaling for regression
        encoder="ifrequency",
        auto_select=True
    )
    .with_imputation(
        method="iterative",     # Best for continuous data
        auto_select=True
    )
    .with_vif_filtering(threshold=7.0)  # Strict VIF for regression
    .with_optimization(
        optimization_level="thorough",
        random_state=42
    )
    .build()
)

print("Regression Optimized Configuration:")
print(f"  Scaler: {regression_pipeline.config.encoding.scaler}")
print(f"  VIF Threshold: {regression_pipeline.config.feature_selection.vif_threshold}")
print(f"  Imputation: {regression_pipeline.config.imputation.method}")

# =============================================================================
# BUILDER CONFIGURATION 8: CLASSIFICATION OPTIMIZED
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 8: CLASSIFICATION OPTIMIZED")
print("=" * 70)

classification_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.99)
    .with_feature_selection(
        method="h2o",
        relevance=0.95,
        h2o_models=8,
        encoding_fs=True
    )
    .with_encoding(
        scaler="standard",      # Standard scaling for classification
        encoder="ifrequency",
        auto_select=True
    )
    .with_imputation(
        method="knn",           # KNN often good for classification
        auto_select=True
    )
    .with_vif_filtering(threshold=10.0)
    .with_optimization(
        optimization_level="balanced",
        random_state=42
    )
    .build()
)

print("Classification Optimized Configuration:")
print(f"  Scaler: {classification_pipeline.config.encoding.scaler}")
print(f"  Imputation: {classification_pipeline.config.imputation.method}")

# =============================================================================
# BUILDER CONFIGURATION 9: MINIMAL PREPROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 9: MINIMAL PREPROCESSING")
print("=" * 70)

minimal_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=False, drop=False)   # Skip date engineering
    .with_null_removal(threshold=1.0)                   # Keep all columns
    .with_feature_selection(method="none")              # No feature selection
    .with_encoding(
        scaler="minmax",
        encoder="label",
        auto_select=False
    )
    .with_imputation(
        method="simple",
        auto_select=False
    )
    .with_vif_filtering(threshold=30.0)  # Very relaxed (almost no filtering)
    .with_optimization(
        optimization_level="fast",
        random_state=42
    )
    .build()
)

print("Minimal Preprocessing Configuration:")
print(f"  Date Engineering: {minimal_pipeline.config.date_engineering}")
print(f"  Null Removal: {minimal_pipeline.config.null_removal_threshold}")
print(f"  Feature Selection: {minimal_pipeline.config.feature_selection.method}")
print(f"  VIF Threshold: {minimal_pipeline.config.feature_selection.vif_threshold}")

# =============================================================================
# BUILDER CONFIGURATION 10: MAXIMUM PREPROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("CONFIGURATION 10: MAXIMUM PREPROCESSING")
print("=" * 70)

maximum_pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.70)      # Aggressive null removal
    .with_feature_selection(
        method="h2o",
        relevance=0.99,                     # Maximum relevance
        h2o_models=20,                      # Maximum models
        encoding_fs=True
    )
    .with_encoding(
        scaler="robust",
        encoder="ifrequency",
        auto_select=True
    )
    .with_imputation(
        method="iterative",
        auto_select=True
    )
    .with_vif_filtering(threshold=5.0)      # Very strict VIF
    .with_optimization(
        optimization_level="thorough",
        random_state=42
    )
    .build()
)

print("Maximum Preprocessing Configuration:")
print(f"  Null Removal: {maximum_pipeline.config.null_removal_threshold}")
print(f"  H2O Models: {maximum_pipeline.config.feature_selection.h2o_max_models}")
print(f"  VIF Threshold: {maximum_pipeline.config.feature_selection.vif_threshold}")
print(f"  optimization {maximum_pipeline.config.optimizer.optimization_level}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("BUILDER CONFIGURATIONS SUMMARY")
print("=" * 70)

configs = [
    ("Default", default_config),
    ("Fast", fast_pipeline.config),
    ("Thorough", thorough_pipeline.config),
    ("High-Null", high_null_pipeline.config),
    ("High-Card", high_card_pipeline.config),
    ("No-H2O", no_h2o_pipeline.config),
    ("Regression", regression_pipeline.config),
    ("Classification", classification_pipeline.config),
    ("Minimal", minimal_pipeline.config),
    ("Maximum", maximum_pipeline.config),
]

print(f"\n{'Config':<15} {'H2O Models':<12} {'Optimization Level':<15} {'VIF':<8} {'Scaler':<10} {'Imputer':<12}")
print("-" * 80)
for name, cfg in configs:
    h2o_models = cfg.feature_selection.h2o_max_models if cfg.feature_selection.method == "h2o" else "N/A"
    print(f"{name:<15} {str(h2o_models):<12} {cfg.feature_selection.vif_threshold:<8} {cfg.encoding.scaler:<10} {cfg.imputation.method:<12}")

print("\n" + "=" * 70)
print("USE CASE RECOMMENDATIONS")
print("=" * 70)
print("""
- Quick Prototyping:     Fast configuration
- Production Pipeline:   Thorough configuration  
- High Missing Data:     High-Null configuration
- Many Categories:       High-Cardinality configuration
- No H2O Available:      No-H2O configuration
- Regression Tasks:      Regression Optimized
- Classification Tasks:  Classification Optimized
- Minimal Processing:    Minimal configuration
- Best Results:          Maximum configuration
""")








































