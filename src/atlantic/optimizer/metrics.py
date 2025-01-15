import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error,
                             explained_variance_score,
                             max_error,
                             r2_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def metrics_regression(y_true, y_pred):
    
    # Calculate various regression model evaluation metrics.
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    maximo_error = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {'Mean Absolute Error': mae,
               'Mean Absolute Percentage Error': mape,
               'Mean Squared Error': mse,
               'Explained Variance Score': evs,
               'Max Error': maximo_error,
               'R2 Score':r2}

    return pd.DataFrame(metrics, index=[0])


def metrics_classification(y_true, y_pred):
    
    n_classes = len(np.unique(y_true))
    average = 'weighted' if n_classes > 2 else 'binary'
    
    # Calculate various classification model evaluation metrics.
    precision_metric = precision_score(y_true, y_pred, average=average)
    f1_metric = f1_score(y_true, y_pred, average=average)
    recall_score_metric = recall_score(y_true, y_pred, average=average)
    
    metrics = {'Precision': precision_metric,
               'F1':f1_metric,
               'Recall':recall_score_metric}
    if n_classes > 2 :
        accuracy_metric = accuracy_score(y_true, y_pred)
        metrics.update({'Accuracy': accuracy_metric})
    
    
    return pd.DataFrame(metrics, index=[0])