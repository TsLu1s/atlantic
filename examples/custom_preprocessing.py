from atlantic.processing.analysis import Analysis 
from atlantic.processing.scalers import (AutoMinMaxScaler, 
                                         AutoStandardScaler,
                                         AutoRobustScaler)
from atlantic.processing.encoders import (AutoLabelEncoder, 
                                          AutoIFrequencyEncoder,
                                          AutoOneHotEncoder)
from atlantic.imputers.imputation import (AutoSimpleImputer, 
                                          AutoKNNImputer,
                                          AutoIterativeImputer)
from atlantic.feature_selection.selector import Selector  
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

url="https://raw.githubusercontent.com/TsLu1s/Atlantic/main/data/Fraudulent_Claim_Cars_class.csv"

data= pd.read_csv(url, encoding='latin', delimiter=',')

target="fraud"

data["claim_date"]=pd.to_datetime(data["claim_date"])
data=data[data[target].isnull()==False]
data=data.reset_index(drop=True)
data[target]=data[target].astype('category')

data.dtypes
data.isna().sum()

train,test = train_test_split(data, train_size=0.8)
train,test=train.reset_index(drop=True), test.reset_index(drop=True)

#################################### ATLANTIC DATA PREPROCESSING #####################################

########################################################################
################# Date Time Feature Engineering #################

an = Analysis(target=target)
train = an.engin_date(X=train,drop=True)
test = an.engin_date(X=test,drop=True)

########################################################################
################# Encoders #################

# LabelEncoder, OneHotEncoder and InverseFrequency (IDF based) methods with an automatic multicolumn application.

train_df,test_df=train.copy(),test.copy()

cat_cols=list(Analysis(target).cat_cols(X=train_df))

## Create Label Encoder
encoder = AutoLabelEncoder()
## Create IFrequency Encoder
encoder = AutoIFrequencyEncoder()
## Create One-hot Encoder
encoder = AutoOneHotEncoder()

## Fit
encoder.fit(train_df[cat_cols])

# Transform the DataFrame using Label\IDF\One-hot Encoding
train_df=encoder.transform(X=train_df)
test_df=encoder.transform(X=test_df)

# Perform an inverse transform to convert it back the original categorical columns values
train_df = encoder.inverse_transform(X = train_df)
test_df = encoder.inverse_transform(X = test_df)

########################################################################
################# Automated Null Imputation [Only numeric features]

# Simplified and automated multivariate null imputation methods based from Sklearn are also provided and applicable, as following:

# Example usage of AutoSimpleImputer
simple_imputer = AutoSimpleImputer(strategy='mean')
simple_imputer.fit(train)  # Fit on the Train DataFrame
df_imputed = simple_imputer.transform(train.copy())  # Transform the Train DataFrame
df_imputed_test = simple_imputer.transform(test.copy()) # Transform the Test DataFrame

# Example usage of AutoKNNImputer
knn_imputer = AutoKNNImputer(n_neighbors=3,
                             weights="uniform")
knn_imputer.fit(train)  # Fit on the Train DataFrame
df_imputed = knn_imputer.transform(train.copy())  # Transform the Train DataFrame
df_imputed_test = knn_imputer.transform(test.copy()) # Transform the Test DataFrame

# Example usage of AutoIterativeImputer
iterative_imputer = AutoIterativeImputer(max_iter=10, 
                                         random_state=0, 
                                         initial_strategy="mean", 
                                         imputation_order="ascending")
iterative_imputer.fit(train)  # Fit on the Train DataFrame
df_imputed = iterative_imputer.transform(train.copy())  # Transform the Train DataFrame
df_imputed_test = iterative_imputer.transform(test.copy()) # Transform the Test DataFrame

train.select_dtypes(include=["float","int"]).isnull().sum()
df_imputed.select_dtypes(include=["float","int"]).isnull().sum()
test.select_dtypes(include=["float","int"]).isnull().sum()
df_imputed_test.select_dtypes(include=["float","int"]).isnull().sum()

########################################################################
################# Feature Selection #################

# You can get filter your most valuable features from the dataset via this 2 feature selection methods:

# * H2O AutoML Feature Selection - This method is based of variable importance evaluation and 
#   calculation for tree-based models in H2Os AutoML and it can be customized by use of the 
#   following parameters:
#   * relevance: Minimal value of the total sum of relative variable\feature importance 
#     percentage selected.
#   * h2o_fs_models: Quantity of models generated for competition to evaluate the relative 
#     importance of each feature (only leaderboard model will be selected for evaluation).
#   * encoding_fs: You can choose if you want to encond your features in order to reduce loading 
#     time. If in `True` mode label encoding is applied to categorical features.

# * VIF Feature Selection (Variance Inflation Factor) - Variance inflation factor aims at 
#   measuring the amount of multicollinearity in a set of multiple regression variables or 
#   features, therefore for this filtering method to be applied all input variables need to be 
#   of numeric type. It can be customized by changing the column filtering treshold 
#   `vif_threshold` designated with a default value of 10.

fs = Selector(X = train, target = target)

selected_cols, selected_importance = fs.feature_selection_h2o(relevance = 0.98,     # total_vi:float [0.5,1], h2o_fs_models:int [1,100]
                                                              h2o_fs_models = 7,    # encoding_fs:bool=True/False
                                                              encoding_fs = True)

cols_vif = fs.feature_selection_vif(vif_threshold = 10.0)                           # X: Only numerical values allowed & No nans allowed
