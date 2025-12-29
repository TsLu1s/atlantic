from enum import Enum


class TaskType(Enum):
    """Machine learning task type."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class DimensionalityTier(Enum):
    """Data dimensionality tier for optimization configuration."""
    SMALL = "small"       # < 5000 rows, < 30 features
    MEDIUM = "medium"     # < 20000 rows, < 100 features
    LARGE = "large"       # < 100000 rows, < 500 features
    XLARGE = "xlarge"     # >= 100000 rows or >= 500 features


class OptimizationLevel(Enum):
    """Optimization optimization level."""
    FAST = "fast"           # Minimal trials, quick results
    BALANCED = "balanced"   # Default, good trade-off
    THOROUGH = "thorough"   # Maximum trials, best results


class EncodingVersion(Enum):
    """Encoding version combinations."""
    V1 = "v1"  # StandardScaler + IFrequencyEncoder
    V2 = "v2"  # MinMaxScaler + IFrequencyEncoder
    V3 = "v3"  # StandardScaler + LabelEncoder
    V4 = "v4"  # MinMaxScaler + LabelEncoder


class ScalerType(Enum):
    """Scaler types available."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class EncoderType(Enum):
    """Encoder types available."""
    LABEL = "label"
    IFREQUENCY = "ifrequency"
    ONEHOT = "onehot"


class ImputerType(Enum):
    """Imputer types available."""
    SIMPLE = "simple"
    KNN = "knn"
    ITERATIVE = "iterative"


class FeatureSelectorType(Enum):
    """Feature selector types available."""
    H2O = "h2o"
    VIF = "vif"


class SimpleImputerStrategy(Enum):
    """Strategy for simple imputation."""
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class RegressionMetric(Enum):
    """Regression evaluation metrics."""
    MAE = "mae"
    MAPE = "mape"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    EVS = "evs"
    MAX_ERROR = "max_error"


class ClassificationMetric(Enum):
    """Classification evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"