[![LinkedIn][linkedin-shield]][linkedin-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Downloads][downloads-shield]][downloads-url]
[![Month Downloads][downloads-month-shield]][downloads-month-url]

[contributors-shield]: https://img.shields.io/github/contributors/TsLu1s/Atlantic.svg?style=for-the-badge&logo=github&logoColor=white
[contributors-url]: https://github.com/TsLu1s/Atlantic/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/TsLu1s/Atlantic.svg?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/TsLu1s/Atlantic/stargazers
[license-shield]: https://img.shields.io/github/license/TsLu1s/Atlantic.svg?style=for-the-badge&logo=opensource&logoColor=white
[license-url]: https://github.com/TsLu1s/Atlantic/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luisfssantos98/
[downloads-shield]: https://static.pepy.tech/personalized-badge/atlantic?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-url]: https://pepy.tech/project/atlantic
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/atlantic?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-month-url]: https://pepy.tech/project/atlantic

<br>
<p align="center">
  <h2 align="center"> Atlantic - Automated Data Preprocessing Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `Atlantic` project constitutes a comprehensive and objective approach to simplify and automate data processing through the integration and validated application of various preprocessing mechanisms, ranging from feature engineering, automated feature selection, multiple encoding versions and null imputation methods. The optimization methodology of this framework follows an evaluation structured in tree-based model ensembles.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed preprocessing procedures are applicable on multiple domains associated with Supervised Machine Learning, regardless of the properties or specifications of the data.

* Automated treatment of tabular data associated with predictive analysis: It implements a global and carefully validated data processing based on the characteristics of the data input columns.

* Robustness and improvement of predictive results: The implementation of the `atlantic` automated data preprocessing pipeline aims at improving predictive performance directly associated with the processing methods implemented based on the data properties.  
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to build this project: 
   
* [H2O.ai](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* [Optuna](https://optuna.org/)
* [Pandas](https://pandas.pydata.org/)

    
## Framework Architecture <a name = "ta"></a>

<p align="center">
  <img src="https://i.ibb.co/C9dWJmk/ATL-Architecture-Final.png" align="center" width="700" height="680" />
</p>    

## Where to get it <a name = "ta"></a>

Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/atlantic/).  

## Installation  

To install this package from Pypi repository run the following command:
```
pip install atlantic
```

# Usage Examples
    
## 1. Atlantic - Automated Data Preprocessing Pipeline

Import the package, load a dataset, split it, and define your target column name. Customize the `fit_processing` method with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `split_ratio` | Train/Validation split ratio for preprocessing evaluation | 0.75 |
| `relevance` | Minimum feature importance percentage for H2O AutoML selection | 0.99 |
| `h2o_fs_models` | Number of models for H2O AutoML feature selection | 7 |
| `encoding_fs` | Encode categorical features before H2O selection | True |
| `vif_ratio` | Variance Inflation Factor threshold | 10.0 |
| `optimization_level` | Optimization intensity: `"fast"`, `"balanced"`, `"thorough"` | "balanced" |

Once fitted, use `data_processing` to transform any future dataframes with the same structure.

```py
import pandas as pd
from sklearn.model_selection import train_test_split
from atlantic.pipeline import Atlantic
import warnings
warnings.filterwarnings("ignore", category=Warning)
    
data = pd.read_csv('csv_directory_path', encoding='latin', delimiter=',')

train, test = train_test_split(data, train_size=0.8)
test, future_data = train_test_split(test, train_size=0.6)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)
future_data.drop(columns=["Target_Column"], inplace=True)

### Fit Data Processing
atl = Atlantic(X=train, target="Target_Column")    

atl.fit_processing(
    split_ratio=0.75,
    relevance=0.99,
    h2o_fs_models=7,
    vif_ratio=10.0,
    optimization_level="balanced"
)

### Transform Data Processing
train = atl.data_processing(X=train)
test = atl.data_processing(X=test)
future_data = atl.data_processing(X=future_data)

### Export Preprocessing Metadata
import dill as pickle
with open('fit_atl.pkl', 'wb') as output:
    pickle.dump(atl, output)
```  

## 2. Atlantic - Builder Pattern (Granular Control)

For fine-grained control over preprocessing steps, use the `AtlanticBuilder` fluent interface:
```py
from sklearn.model_selection import train_test_split
from atlantic.pipeline import AtlanticBuilder

train, test = train_test_split(data, train_size=0.8)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

### Build Custom Pipeline
pipeline = (AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.90)
    .with_feature_selection(
        method="h2o",
        relevance=0.95,
        h2o_models=10,
        encoding_fs=True
    )
    .with_encoding(
        scaler="standard",
        encoder="ifrequency",
        auto_select=True
    )
    .with_imputation(
        method="knn",
        auto_select=True
    )
    .with_vif_filtering(threshold=10.0)
    .with_optimization(optimization_level="balanced")
    .build()
)

### Fit and Transform
train_processed = pipeline.fit_transform(train, target="Target_Column")
test_processed = pipeline.transform(test)
```

### Builder Configuration Presets

| Configuration | Use Case | Key Settings |
|--------------|----------|--------------|
| Fast | Quick prototyping | `h2o_models=3`, `method="simple"`, `optimization_level="fast"` |
| Balanced | General purpose | Default settings |
| Thorough | Best results | `h2o_models=15`, `method="iterative"`, `optimization_level="thorough"` |
| High-Null | Missing data >20% | `threshold=0.80`, `scaler="robust"`, `method="iterative"` |
| No-H2O | Skip H2O selection | `method="none"`, VIF filtering only |
```py
# Fast Prototyping
fast_pipeline = (AtlanticBuilder()
    .with_feature_selection(method="h2o", relevance=0.85, h2o_models=3)
    .with_encoding(scaler="minmax", encoder="label", auto_select=False)
    .with_imputation(method="simple", auto_select=False)
    .with_optimization(optimization_level="fast")
    .build()
)

# Thorough Optimization  
thorough_pipeline = (AtlanticBuilder()
    .with_feature_selection(method="h2o", relevance=0.98, h2o_models=15)
    .with_encoding(auto_select=True)
    .with_imputation(method="iterative", auto_select=True)
    .with_vif_filtering(threshold=8.0)
    .with_optimization(optimization_level="thorough")
    .build()
)

# High-Null Data
high_null_pipeline = (AtlanticBuilder()
    .with_null_removal(threshold=0.80)
    .with_encoding(scaler="robust")
    .with_imputation(method="iterative", auto_select=True)
    .build()
)
```

## 3. Atlantic - Preprocessing Components
    
### 3.1 Encoding Methods
 
Encode categorical variables into numerical format. Choose from label encoding (ordinal mapping), one-hot encoding (binary columns), or inverse frequency encoding (IDF-based weights).

```py
import pandas as pd
from sklearn.model_selection import train_test_split 
from atlantic.preprocessing import AutoLabelEncoder, AutoIFrequencyEncoder, AutoOneHotEncoder

train, test = train_test_split(data, train_size=0.8)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

target = "Target_Column"
cat_cols = [col for col in data.select_dtypes(include=['object']).columns if col != target]

### Create Encoder (choose one)
encoder = AutoLabelEncoder()
# encoder = AutoIFrequencyEncoder()
# encoder = AutoOneHotEncoder()

### Fit and Transform
encoder.fit(train[cat_cols])
train[cat_cols] = encoder.transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

### Inverse Transform (if needed)
train[cat_cols] = encoder.inverse_transform(train[cat_cols])
```    

### 3.2 Scalers

Normalize numerical features to improve model convergence. Standard scaler for normal distributions, MinMax for bounded ranges, or Robust scaler for data with outliers.

```py
from atlantic.preprocessing import AutoStandardScaler, AutoMinMaxScaler, AutoRobustScaler

num_cols = train.select_dtypes(include=['int', 'float']).columns.tolist()

### Create Scaler (choose one)
scaler = AutoStandardScaler()  # Zero mean, unit variance
# scaler = AutoMinMaxScaler()  # Scale to [0, 1]
# scaler = AutoRobustScaler()  # Median/IQR based, outlier-resistant

### Fit and Transform
scaler.fit(train[num_cols])
train[num_cols] = scaler.transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])
```

### 3.3 Imputers

Handle missing values using statistical methods. Simple imputation for speed, KNN for local patterns, or iterative modeling for complex dependencies.

```py
from atlantic.preprocessing import AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer

### Create Imputer (choose one)
imputer = AutoSimpleImputer(strategy='mean', target=target)
# imputer = AutoKNNImputer(n_neighbors=5, target=target)
# imputer = AutoIterativeImputer(max_iter=10, target=target)

### Fit and Transform
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)
```

### 3.4 Feature Selection

Reduce dimensionality by removing redundant or low-importance features. VIF removes multicollinear features, while H2O AutoML identifies the most predictive ones.


```py
from atlantic.feature_selection import VIFFeatureSelector, H2OFeatureSelector

### VIF-based Selection
vif_selector = VIFFeatureSelector(target=target, vif_threshold=10.0)
vif_selector.fit(train)
train = vif_selector.transform(train)
test = vif_selector.transform(test)

### H2O AutoML Selection
h2o_selector = H2OFeatureSelector(target=target, relevance=0.95, max_models=7)
h2o_selector.fit(train)
train = h2o_selector.transform(train)
test = h2o_selector.transform(test)
```

Check out the <a href="https://github.com/TsLu1s/atlantic/blob/main/examples/custom_preprocessing.py" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Custom%20Preprocessing-blue?style=for-the-badge&logo=readme&logoColor=white" alt="Custom Preprocessing">
</a> for detailed implementations of all preprocessing methods.


## Citation
```bibtex
@article{SANTOS2023100532,
  author = {Luis Fernando Santos and Luis Ferreira},
  title = {Atlantic - Automated data preprocessing framework for supervised machine learning},
  journal = {Software Impacts},
  volume = {17},
  year = {2023},
  issn = {2665-9638},
  doi = {http://dx.doi.org/10.1016/j.simpa.2023.100532},
  url = {https://www.sciencedirect.com/science/article/pii/S2665963823000696}
}
```
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/Atlantic/blob/main/LICENSE) for more information.

## Contact 
 
[Luis Santos - LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
