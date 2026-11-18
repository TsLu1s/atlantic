<div align="center">

<h1>Atlantic - Automated Data Preprocessing for Supervised Machine Learning</h1>

<p>Feature engineering, feature selection, encoding and imputation, each chosen for your data by validation score.</p>

[![PyPI][pypi-shield]][pypi-url]
[![Python][python-shield]][pypi-url]
[![MIT License][license-shield]][license-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<br>
[![Downloads][downloads-shield]][downloads-url]
[![Month Downloads][downloads-month-shield]][downloads-url]

<p>
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#usage-guide">Usage guide</a> &nbsp;·&nbsp;
  <a href="#examples-and-notebooks">Examples</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

---

## Overview

The Atlantic project constitutes a comprehensive and objective approach to simplify and automate data processing through the integration and validated application of various preprocessing mechanisms, ranging from feature engineering, automated feature selection, multiple encoding versions and null imputation methods. The optimization methodology of this framework follows an evaluation structured in tree-based model ensembles.

- **Works on any tabular dataset.** Regression, binary and multiclass classification, across domains, with numeric, categorical, boolean and datetime columns.
- **Decisions backed by evidence.** Encoding, imputation and multicollinearity filtering are selected by validation score, not by rule of thumb.
- **Automated feature selection.** H2O AutoML variable importance keeps the features that carry the share of signal you ask for.
- **Leak-free by design.** Final transformers are fitted on training rows only, so validation rows never influence them.
- **Built for reuse.** A fitted pipeline replays the exact same steps on new data and can be saved and reloaded.
- **Transparent.** Every decision, including every dropped column and the reason it was dropped, is exposed for inspection.

---

## How it works

<p align="center">
  <a href="https://raw.githubusercontent.com/TsLu1s/atlantic/main/docs/Atlantic_Pipeline_Diagram.png">
    <img src="https://raw.githubusercontent.com/TsLu1s/atlantic/main/docs/Atlantic_Pipeline_Diagram.png" alt="Atlantic pipeline: validation and preparation, train/validation split with optional H2O feature selection, selection of encoding, imputation and VIF filtering by validation score, final fit on the training split, transformation of new data, and persistence" width="100%" />
  </a>
</p>

The pipeline runs in six stages:

| # | Stage | What happens |
|:-:|-------|--------------|
| 1 | **Validate & prepare** | Infers the task, drops rows without a target, encodes booleans as 0/1, expands datetimes and removes unusable columns |
| 2 | **Split & select features** | Splits into train and validation (stratified for classification) and, when `relevance < 1.0`, keeps the most important features found by H2O AutoML |
| 3 | **Select by validation score** | Chooses the encoding, imputation and VIF filtering steps one at a time, keeping each only if it scores best |
| 4 | **Fit final transformers** | Refits the chosen scaler, encoder and imputer on the training rows only |
| 5 | **Transform new data** | `data_processing` replays the same steps on any future frame |
| 6 | **Inspect & persist** | Exposes every decision and saves the fitted pipeline for reuse |

<details>
<summary><strong>Stage-by-stage details</strong></summary>

<br>

**1. Validate & prepare**

The task is inferred from the target column:

| Target type | Task | Evaluation metric |
|-------------|------|-------------------|
| Numeric | Regression | MAE |
| Other, with 2 classes | Binary classification | Precision |
| Other, with more than 2 classes | Multiclass classification | F1 |

- Rows with no target value are dropped.
- Boolean columns become 0/1 features.
- Each datetime column is expanded into nine calendar features.
- High-null, constant and identifier-like columns are removed. Every removal is recorded in `dropped_columns`.

**2. Split & select features**

- The data is divided into a train and a validation split. Classification splits are stratified.
- When `relevance < 1.0`, only the features carrying the requested share of H2O AutoML variable importance are kept.

**3. Select by validation score**

| Step | Candidates | Applied when |
|------|------------|--------------|
| Encoding | 4 combinations: Standard or MinMax scaler × IFrequency or Label encoder | Always |
| Imputation | Simple (mean), KNN, Iterative | The data has missing values |
| VIF filtering | Iterative removal of features above `vif_ratio` | It improves the validation score |

- Each candidate is fitted on the train split and scored on the validation split, using tuned RandomForest, ExtraTrees and XGBoost models.
- The number of tuning trials adapts to the data size and to `optimization_level`.

**4. Fit final transformers**

The chosen scaler, encoder and imputer are refitted on the training rows only, so validation rows never influence them.

**5. Transform new data**

- Frames missing a fitted column are rejected with a clear error.
- Unseen categories and missing values are handled.

**6. Inspect & persist**

- The pipeline's decisions are exposed as `enc_method`, `imp_method`, `cols` and `dropped_columns`.
- The fitted pipeline can be saved with `save` and reloaded with `Atlantic.load`.

</details>

---

## Installation

Atlantic is published on [PyPI](https://pypi.org/project/atlantic/). One command installs the full framework, H2O AutoML included:

```bash
pip install atlantic
```

---

## Quick start

```python
from sklearn.model_selection import train_test_split
from atlantic import Atlantic
from atlantic.data import DatasetGenerator

# Any pandas DataFrame with a target column works; here we generate one.
data, target = DatasetGenerator.generate_classification(
    n_samples=2000, n_features=15, n_classes=2, n_categorical=4, null_percentage=0.05
)
train, test = train_test_split(data, train_size=0.8, random_state=42)

atl = Atlantic(X=train, target=target)
atl.fit_processing()                      # H2O feature selection + balanced search

train_ready = atl.data_processing(X=train)
test_ready = atl.data_processing(X=test)

print(atl.enc_method, atl.imp_method)     # what Atlantic decided
```

---

## Usage guide

### 1. Automated pipeline

The `Atlantic` class runs the full pipeline. Fit it once on training data, then use `data_processing` to transform any future frame with the same structure.

**`fit_processing` parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `split_ratio` | Train/validation split ratio used to evaluate preprocessing candidates (0.5 – 0.98) | `0.75` |
| `relevance` | Share of H2O AutoML feature importance to keep (0.4 – 1.0). `1.0` skips H2O feature selection | `0.99` |
| `h2o_fs_models` | Number of models trained by H2O AutoML for feature selection | `7` |
| `encoding_fs` | Encode categorical features before H2O selection | `True` |
| `vif_ratio` | Variance Inflation Factor threshold (3 – 30) | `10.0` |
| `optimization_level` | Search intensity: `"fast"`, `"balanced"` or `"thorough"` | `"balanced"` |

**`Atlantic` constructor parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `random_state` | Seed for the internal train/validation split and every model evaluation | `42` |
| `n_jobs` | Parallelism of the evaluation models. `None` keeps the tree ensembles single-threaded; `-1` uses all cores (same results, faster on mid-sized data) | `None` |
| `verbosity` | `0` silent, `1` selection summary, `2` detailed | `1` |

> **Task type.** A numeric target is treated as regression; anything else (strings, categories, booleans) as classification. Integer-coded class labels such as `0`/`1` are therefore treated as regression and a warning is raised. Cast them first: `data["Target_Column"] = data["Target_Column"].astype(str)`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from atlantic.pipeline import Atlantic

data = pd.read_csv("csv_directory_path", encoding="latin", delimiter=",")

train, test = train_test_split(data, train_size=0.8)
test, future_data = train_test_split(test, train_size=0.6)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True).drop(columns=["Target_Column"])

# Fit
atl = Atlantic(X=train, target="Target_Column")
atl.fit_processing(
    split_ratio=0.75,
    relevance=0.99,              # H2O AutoML feature selection; 1.0 skips it
    h2o_fs_models=7,
    vif_ratio=10.0,
    optimization_level="balanced",
)

# Transform
train = atl.data_processing(X=train)
test = atl.data_processing(X=test)
future_data = atl.data_processing(X=future_data)

# Inspect what the pipeline decided
print(atl.enc_method, atl.imp_method)   # selected encoding version / imputation method
print(atl.cols)                         # feature columns kept
print(atl.dropped_columns)              # columns removed, grouped by reason

# Persist (pickle-based; the training data is not stored in the file)
atl.save("fit_atl.pkl")
loaded_atl = Atlantic.load("fit_atl.pkl")
```

> **Security note.** Pipeline files are pickles. Only load files you trust.

### 2. Builder pattern

For fine-grained control over each step, compose a pipeline with the fluent `AtlanticBuilder` interface:

```python
from sklearn.model_selection import train_test_split
from atlantic.pipeline import AtlanticBuilder

train, test = train_test_split(data, train_size=0.8)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

pipeline = (
    AtlanticBuilder()
    .with_date_engineering(enabled=True, drop=True)
    .with_null_removal(threshold=0.90)
    .with_feature_selection(method="h2o", relevance=0.95, h2o_models=10, encoding_fs=True)
    .with_encoding(scaler="standard", encoder="ifrequency", auto_select=True)
    .with_imputation(method="knn", auto_select=True)
    .with_vif_filtering(threshold=10.0)
    .with_optimization(optimization_level="balanced")
    .build()
)

train_processed = pipeline.fit_transform(train, target="Target_Column")
test_processed = pipeline.transform(test)

print(pipeline.atlantic.enc_method, pipeline.atlantic.dropped_columns)
pipeline.save("fit_pipeline.pkl")
# from atlantic.pipeline import AtlanticPipeline
# pipeline = AtlanticPipeline.load("fit_pipeline.pkl")
```

| Builder method | Behaviour |
|----------------|-----------|
| `with_date_engineering(enabled, drop)` | `enabled=False` leaves datetime columns out of the features; `drop=False` keeps the original datetime columns in the output next to the generated features |
| `with_encoding(..., auto_select=True)` | Selects the best scaler and encoder by validation score and ignores the explicit `scaler` / `encoder`. With `auto_select=False` they are applied as given |
| `with_imputation(..., auto_select=True)` | Selects the best imputer by validation score and ignores the explicit `method`. With `auto_select=False` it is applied as given |
| `with_optimization(optimization_level, random_state, split_ratio, n_jobs, verbosity)` | Controls the evaluation budget, reproducibility and console output |

#### Configuration presets

| Preset | Use case | Key settings |
|--------|----------|--------------|
| **Fast** | Quick prototyping | `h2o_models=3`, `method="simple"`, `optimization_level="fast"` |
| **Balanced** | General purpose | Default settings |
| **Thorough** | Best results | `h2o_models=15`, `method="iterative"`, `optimization_level="thorough"` |
| **High-null** | More than 20% missing data | `threshold=0.80`, `scaler="robust"`, `method="iterative"` |
| **No-H2O** | No Java runtime available | `method="none"`, VIF filtering only |

```python
# Fast prototyping
fast_pipeline = (
    AtlanticBuilder()
    .with_feature_selection(method="h2o", relevance=0.85, h2o_models=3)
    .with_encoding(scaler="minmax", encoder="label", auto_select=False)
    .with_imputation(method="simple", auto_select=False)
    .with_optimization(optimization_level="fast")
    .build()
)

# Thorough optimization
thorough_pipeline = (
    AtlanticBuilder()
    .with_feature_selection(method="h2o", relevance=0.98, h2o_models=15)
    .with_encoding(auto_select=True)
    .with_imputation(method="iterative", auto_select=True)
    .with_vif_filtering(threshold=8.0)
    .with_optimization(optimization_level="thorough")
    .build()
)

# High-null data
high_null_pipeline = (
    AtlanticBuilder()
    .with_null_removal(threshold=0.80)
    .with_encoding(scaler="robust")
    .with_imputation(method="iterative", auto_select=True)
    .build()
)
```

### 3. Preprocessing components

Every building block of the pipeline is also available on its own, with a scikit-learn style `fit` / `transform` interface.

| Component | Classes |
|-----------|---------|
| [Encoders](#31-encoders) | `AutoLabelEncoder`, `AutoIFrequencyEncoder`, `AutoOneHotEncoder` |
| [Scalers](#32-scalers) | `AutoStandardScaler`, `AutoMinMaxScaler`, `AutoRobustScaler` |
| [Imputers](#33-imputers) | `AutoSimpleImputer`, `AutoKNNImputer`, `AutoIterativeImputer` |
| [Feature selectors](#34-feature-selectors) | `VIFFeatureSelector`, `H2OFeatureSelector` |

#### 3.1 Encoders

Label encoding (ordinal mapping) and inverse-frequency encoding (IDF-based weights) both preserve the column layout, so the result can be assigned back in place. `AutoOneHotEncoder` expands each categorical column into several binary columns and returns a wider frame.

```python
from sklearn.model_selection import train_test_split
from atlantic.preprocessing import AutoLabelEncoder, AutoIFrequencyEncoder, AutoOneHotEncoder
from atlantic.utils import get_categorical_columns

train, test = train_test_split(data, train_size=0.8)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

target = "Target_Column"
cat_cols = get_categorical_columns(data, exclude=[target])  # object, category and string dtypes

# Same-shape output: label or inverse-frequency
encoder = AutoLabelEncoder()
# encoder = AutoIFrequencyEncoder()

encoder.fit(train[cat_cols])
train[cat_cols] = encoder.transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

train[cat_cols] = encoder.inverse_transform(train[cat_cols])  # if needed

# One-hot returns a WIDER frame: do not assign it back into cat_cols
# ohe = AutoOneHotEncoder()
# ohe.fit(train[cat_cols])
# train = ohe.transform(train)   # original cat_cols are replaced by binary columns
# test = ohe.transform(test)
```

#### 3.2 Scalers

Standard scaling for roughly normal distributions, MinMax for bounded ranges, and Robust scaling for data with outliers.

```python
from atlantic.preprocessing import AutoStandardScaler, AutoMinMaxScaler, AutoRobustScaler
from atlantic.utils import get_numeric_columns

target = "Target_Column"
num_cols = get_numeric_columns(train, exclude=[target])  # never scale the target

scaler = AutoStandardScaler()    # zero mean, unit variance
# scaler = AutoMinMaxScaler()    # scale to [0, 1]
# scaler = AutoRobustScaler()    # median/IQR based, outlier-resistant

scaler.fit(train[num_cols])
train[num_cols] = scaler.transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])
```

#### 3.3 Imputers

Simple imputation for speed, KNN for local patterns, or iterative modelling for complex dependencies between features.

```python
from atlantic.preprocessing import AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer

imputer = AutoSimpleImputer(strategy="mean", target=target)
# imputer = AutoKNNImputer(n_neighbors=5, target=target)
# imputer = AutoIterativeImputer(max_iter=10, target=target)

imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)
```

#### 3.4 Feature selectors

VIF filtering removes multicollinear features; H2O AutoML keeps the most predictive ones.

```python
from atlantic.feature_selection import VIFFeatureSelector, H2OFeatureSelector

# VIF: expects encoded, imputed, numeric features
vif_selector = VIFFeatureSelector(target=target, vif_threshold=10.0)
vif_selector.fit(train)
train = vif_selector.transform(train)
test = vif_selector.transform(test)

# H2O AutoML: starts a local H2O cluster (needs Java)
h2o_selector = H2OFeatureSelector(target=target, relevance=0.95, max_models=7)
h2o_selector.fit(train)
train = h2o_selector.transform(train)
test = h2o_selector.transform(test)
```

> **Note.** `VIFFeatureSelector` evaluates numeric, null-free columns, so impute nulls first (it raises `ValidationError` on nulls). Non-numeric columns are passed through untouched, and `vif_threshold` must be within `[3.0, 30.0]`.

---

## Examples and notebooks

| Resource | What it covers |
|----------|----------------|
| [`automated_pipeline.py`](https://github.com/TsLu1s/atlantic/blob/main/examples/automated_pipeline.py) | The automated pipeline on regression, binary and multiclass tasks |
| [`builder_complete.py`](https://github.com/TsLu1s/atlantic/blob/main/examples/builder_complete.py) | Every builder configuration, plus an end-to-end fit |
| [`custom_preprocessing.py`](https://github.com/TsLu1s/atlantic/blob/main/examples/custom_preprocessing.py) | Each preprocessing component used on its own |
| [Jupyter notebooks](https://github.com/TsLu1s/atlantic/tree/main/examples/notebooks) | Interactive, step-by-step walkthroughs of all of the above |

---

## Built with

[H2O.ai AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) · [scikit-learn](https://scikit-learn.org/stable/) · [XGBoost](https://xgboost.readthedocs.io/en/stable/) · [Optuna](https://optuna.org/) · [pandas](https://pandas.pydata.org/)

---

## Citation

If Atlantic supports your work, please cite the accompanying paper:

```bibtex
@article{SANTOS2023100532,
  author  = {Luis Fernando Santos and Luis Ferreira},
  title   = {Atlantic - Automated data preprocessing framework for supervised machine learning},
  journal = {Software Impacts},
  volume  = {17},
  year    = {2023},
  issn    = {2665-9638},
  doi     = {http://dx.doi.org/10.1016/j.simpa.2023.100532},
  url     = {https://www.sciencedirect.com/science/article/pii/S2665963823000696}
}
```

## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/atlantic/blob/main/LICENSE) for details.

## Contact

**Luís Santos** · [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)

<!-- Badge references -->
[pypi-shield]: https://img.shields.io/pypi/v/atlantic.svg?style=for-the-badge&logo=pypi&logoColor=white
[pypi-url]: https://pypi.org/project/atlantic/
[python-shield]: https://img.shields.io/pypi/pyversions/atlantic.svg?style=for-the-badge&logo=python&logoColor=white
[downloads-shield]: https://static.pepy.tech/personalized-badge/atlantic?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/atlantic?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-url]: https://pepy.tech/project/atlantic
[license-shield]: https://img.shields.io/github/license/TsLu1s/Atlantic.svg?style=for-the-badge&logo=opensource&logoColor=white
[license-url]: https://github.com/TsLu1s/Atlantic/blob/main/LICENSE
[contributors-shield]: https://img.shields.io/github/contributors/TsLu1s/Atlantic.svg?style=for-the-badge&logo=github&logoColor=white
[contributors-url]: https://github.com/TsLu1s/Atlantic/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/TsLu1s/Atlantic.svg?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/TsLu1s/Atlantic/stargazers
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luisfssantos98/
