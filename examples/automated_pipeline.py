"""
Atlantic - Usage Examples.

Examples for regression, binary classification, and multi-class classification.

Note: Set relevance=1.0 to skip H2O feature selection.
      Set relevance<1.0 (e.g., 0.99) to enable H2O feature selection.
"""

from atlantic import Atlantic
from atlantic.data import DatasetGenerator
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning)


# #############################################################################
#                            BINARY CLASSIFICATION
# #############################################################################

data, target_col = DatasetGenerator.generate_classification(
    n_samples=10000,
    n_features=15,
    n_classes=2,
    n_categorical=4,
    null_percentage=0.08,
    random_state=42
)

train, test = train_test_split(data, train_size=0.8, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

atl = Atlantic(X=train, target=target_col)

atl.fit_processing(
    split_ratio=0.75,
    relevance=1.0,
    h2o_fs_models=7,
    vif_ratio=10.0,
    optimization_level="balanced"
)

train_copy = atl.data_processing(X=train.copy())
test_copy = atl.data_processing(X=test.copy())

#atl.save('atlantic_binary.pkl')
#loaded_atl = Atlantic.load('atlantic_binary.pkl')

# #############################################################################
#                         MULTI-CLASS CLASSIFICATION
# #############################################################################

data, target_col = DatasetGenerator.generate_classification(
    n_samples=1000,
    n_features=15,
    n_classes=5,
    n_categorical=4,
    null_percentage=0.08,
    random_state=42
)

train, test = train_test_split(data, train_size=0.8, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

atl = Atlantic(X=train, target=target_col)

atl.fit_processing(
    split_ratio=0.75,
    relevance=1.0,
    h2o_fs_models=7,
    vif_ratio=10.0,
    optimization_level="balanced"
)

train = atl.data_processing(X=train)
test = atl.data_processing(X=test)

#atl.save('atlantic_multiclass.pkl')
#loaded_atl = Atlantic.load('atlantic_multiclass.pkl')

# #############################################################################
#                               REGRESSION
# #############################################################################

data, target_col = DatasetGenerator.generate_regression(
    n_samples=1000,
    n_features=15,
    n_categorical=4,
    null_percentage=0.08,
    random_state=42
)

train, test = train_test_split(data, train_size=0.8, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

atl = Atlantic(X=train, target=target_col)

atl.fit_processing(
    split_ratio=0.75,
    relevance=1.0,
    h2o_fs_models=7,
    vif_ratio=10.0,
    optimization_level="balanced"
)

train = atl.data_processing(X=train)
test = atl.data_processing(X=test)

#atl.save('atlantic_regression.pkl')

# #############################################################################
#                      LOADING SAVED FITTED PIPELINE
# #############################################################################

# Load previously fitted Atlantic pipeline
loaded_atl = Atlantic.load('atlantic_regression.pkl')

# Apply preprocessing to new data using loaded pipeline
new_data, _ = DatasetGenerator.generate_regression(
    n_samples=200,
    n_features=15,
    n_categorical=4,
    null_percentage=0.05,
    random_state=99
)

new_data = new_data.reset_index(drop=True)
new_data_processed = loaded_atl.data_processing(X=new_data)

# Access fitted components metadata
print(f"Encoding Method: {loaded_atl.enc_method}")
print(f"Imputation Method: {loaded_atl.imp_method}")
print(f"Selected Columns: {loaded_atl.cols}")
print(f"Numerical Columns: {loaded_atl.n_cols}")
print(f"Categorical Columns: {loaded_atl.c_cols}")
