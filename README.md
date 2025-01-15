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
[linkedin-url]: https://www.linkedin.com/in/luísfssantos/
[downloads-shield]: https://static.pepy.tech/personalized-badge/atlantic?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-url]: https://pepy.tech/project/atlantic
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/atlantic?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-month-url]: https://pepy.tech/project/atlantic

<br>
<p align="center">
  <h2 align="center"> Atlantic - Automated Data Preprocessing Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `Atlantic` project constitutes an comprehensive and objective approach to simplify and automate data processing through the integration and objectively validated application of various preprocessing mechanisms, ranging from feature engineering, automated feature selection, multiple encoding versions and null imputation methods. The optimization methodology of this framework follows a evaluation structured in tree based models ensembles.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed preprocessing procedures are applicable on multiple domains associated with Supervised Machine Learning, regardless of the properties or specifications of the data.

* Automated treatment of tabular data associated with predictive analysis: It implements a global and carefully validated data processing based on the characteristics of the data input columns.

* Robustness and improvement of predictive results: The implementation of the `atlantic` automated data preprocessing pipeline aims at improving predictive performance directly associated with the processing methods implemented based on the data properties.  
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 
   
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

In order to be able to apply the automated preprocessing `atlantic` pipeline you need first to import the package. 
The following needed step is to load a dataset, split it and define your to be predicted target column name into the variable `target`.
You can customize the `fit_processing` method by altering the following running pipeline parameters:
* split_ratio: Division ratio (Train\Validation) in which the preprocessing methods will be evaluated within the loaded Dataset.
* relevance: Minimal value of the total sum of relative feature importance percentage selected in the `H2O AutoML feature selection` step.
* h2o_fs_models: Quantity of models generated for competition in step `H2O AutoML feature selection` to evaluate the relative importance of each feature (only leaderboard model is selected for evaluation).
* encoding_fs: You can choose if you want to enconde categorical features in order to reduce loading time in `H2O AutoML feature selection` step.
* vif_ratio: This value defines the minimal `threshold` for Variance Inflation Factor filtering (default value=10).

Once the data fitting process is complete, you can automaticaly optimize preprocessing transformations on all future dataframes with the same properties using the `data_processing` method.
    
```py
import pandas as pd
from sklearn.model_selection import train_test_split
from atlantic.pipeline import Atlantic
import warnings
warnings.filterwarnings("ignore", category=Warning) # -> For a clean console
    
data = pd.read_csv('csv_directory_path', encoding='latin', delimiter=',') # Dataframe Loading Example
# data["Target Column"] = data["Target Column"].astype(str) # -> If Classification Task

train,test = train_test_split(data, train_size = 0.8)
test,future_data = train_test_split(test, train_size = 0.6)

# Resetting Index is Required
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

future_data.drop(columns=["Target_Column"], inplace=True) # Drop Target

### Fit Data Processing

atl = Atlantic(X = train,                # X:pd.DataFrame, target:str="Target_Column"
               target = "Target Column")    

atl.fit_processing(split_ratio = 0.75,   # split_ratio:float=0.75, relevance:float=0.99 [0.5,1]
                   relevance = 0.99,     # h2o_fs_models:int [1,100], encoding_fs:bool=True\False
                   h2o_fs_models = 7,    # vif_ratio:float=10.0 [3,30]
                   encoding_fs = True,
                   vif_ratio = 10.0)

### Transform Data Processing

train = atl.data_processing(X = train)
test = atl.data_processing(X = test)
future_data = atl.data_processing(X = future_data)

### Export Atlantic Preprocessing Metadata

import dill as pickle
with open('fit_atl.pkl', 'wb') as output:
    pickle.dump(atl, output)
    
```  

## 2. Atlantic - Preprocessing Data
    
### 2.1 Encoding Versions
 
There are multiple preprocessing methods available to direct use. This package provides upgrated encoding `LabelEncoder`, `OneHotEncoder` and `InverseFrequency` ([IDF](https://pypi.org/project/cane/) based) methods with an automatic multicolumn application. 
 
```py
import pandas as pd
from sklearn.model_selection import train_test_split 
from atlantic.processing.encoders import AutoLabelEncoder, AutoIFrequencyEncoder, AutoOneHotEncoder

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # Required

target = "Target_Column" # -> target feature name
    
cat_cols = [col for col in data.select_dtypes(include=['object']).columns if col != target]

### Encoders
## Create Label Encoder
encoder = AutoLabelEncoder()
## Create InverseFrequency Encoder
encoder = AutoIFrequencyEncoder()
## Create One-hot Encoder
encoder = AutoOneHotEncoder()

## Fit
encoder.fit(train[cat_cols])

# Transform the DataFrame using Label\IF\One-hot Encoding
train = encoder.transform(X = train)
test = encoder.transform(X = test)

# Perform an inverse transform to convert it back the original categorical columns values
train = encoder.inverse_transform(X = train)
test = encoder.inverse_transform(X = test)
            
```    
   
### 2.2 Feature Selection and Null Imputation Methods

Atlantic provides automated feature selection methods (H2O AutoML and VIF-based) and null imputation techniques (Simple, KNN, and Iterative). Check out the <a href="https://github.com/TsLu1s/atlantic/edit/main/examples/custom_preprocessing.py" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Custom%20Preprocessing-blue?style=for-the-badge&logo=readme&logoColor=white" alt="Custom Preprocessing">
</a> for detailed implementations of all preprocessing methods integrated in `Atlantic`.

## Citation

Feel free to cite Atlantic as following:

```

@article{SANTOS2023100532,
  author = {Luis Santos and Luis Ferreira}
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
