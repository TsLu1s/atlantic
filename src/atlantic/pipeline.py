import pandas as pd
from atlantic.processing.analysis import Analysis 
from atlantic.processing.scalers import (AutoMinMaxScaler,
                                         AutoStandardScaler,
                                         AutoRobustScaler) 
from atlantic.processing.encoders import (AutoLabelEncoder,
                                          AutoIFrequencyEncoder,
                                          AutoOneHotEncoder) 
from atlantic.processing.versions import Encoding_Version
from atlantic.imputers.imputation import (AutoSimpleImputer,
                                          AutoKNNImputer,
                                          AutoIterativeImputer) 
from atlantic.feature_selection.selector import Selector 
from atlantic.scheme.pattern import Pattern 

class Atlantic(Selector):
    """
    The Atlantic class  provides comprehensive data preprocessing capabilities including feature engineering, feature selection, 
    encoding, imputation, and transformation. It is designed to handle both classification and regression tasks
    by automating the selection of optimal preprocessing strategies.

    Attributes
    ----------
    X : pd.DataFrame
        The input dataset to be processed.
    target : str
        The name of the target variable in the dataset.
    enc_method : str
        Selected encoding method.
    imp_method : str
        Selected imputation method.
    encoder : object
        Fitted encoder object.
    scaler : object
        Fitted scaler object.
    imputer : object
        Fitted imputer object.
    n_cols : list
        List of numerical columns.
    c_cols : list
        List of categorical columns.
    cols : list
        List of selected columns after preprocessing.
    h2o_feature_importance : list
        List of feature importances obtained from H2O feature selection.

    Methods
    -------
    fit_processing(split_ratio, relevance=0.99, h2o_fs_models=7, encoding_fs=True, vif_ratio=10.0):
        Fits the preprocessing steps on the input dataset.
    data_processing(X):
        Applies the fitted preprocessing steps to a new dataset.
    """
    def __init__(self,
                 X : pd.DataFrame = None,
                 target : str = None):
        super().__init__(X, target)
        self.enc_method = None
        self.imp_method = None
        self.encoder = None
        self.scaler = None
        self.imputer = None
        self.n_cols = None
        self.c_cols = None
        self.cols = None
        self.h2o_feature_importance = None
        """
        Initialize the Atlantic class with the input dataset and target variable.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The input dataset to be processed, by default None.
        target : str, optional
            The name of the target variable, by default None.
        """

    def fit_processing(self,
                       split_ratio : float,
                       relevance : float = 0.99,
                       h2o_fs_models : int = 7,
                       encoding_fs : bool = True,
                       vif_ratio : float = 10.0):
        """
        Fit the preprocessing steps on the input dataset.

        This method performs a series of preprocessing steps including feature engineering, feature selection,
        encoding method selection, imputation method selection, and VIF-based feature selection. The method
        updates the attributes of the class with the fitted preprocessing objects and selected columns.

        Parameters
        ----------
        split_ratio : float
            The ratio for splitting the dataset into training and testing sets.
        relevance : float, optional
            The relevance threshold for H2O feature selection, by default 0.99.
        h2o_fs_models : int, optional
            The number of models to use for H2O feature selection, by default 7.
        encoding_fs : bool, optional
            Whether to use encoding during feature selection, by default True.
        vif_ratio : float, optional
            The VIF threshold for feature selection, by default 10.0.

        Returns
        -------
        self
            The fitted Atlantic object.
        """
        ### Data Treatment
        if self.pred_type == 'Class':
            self.X[self.target] = self.X[self.target].astype(str)
            
        X_ = self.X.copy()
        
        X_ = super().engin_date(X = X_, drop = True)
        X_ = super().remove_columns_by_nulls(X = X_, percentage=99.9)
        
        X_ = X_.drop(columns=[col for col in X_.columns 
                              if (X_[col].nunique() == len(X_) or X_[col].nunique() == 1) 
                                                               and col != self.target])  
        X_ = X_[[col for col in X_.columns if col != self.target] + [self.target]]       
        
        data = X_.copy()
        
        train, test = super().split_dataset(X = X_, split_ratio = split_ratio)
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
        
        ### Feature Selection 
        if relevance!=1:
            fs = Selector(X = train, 
                          target = self.target)
            sel_cols, self.h2o_feature_importance = fs.feature_selection_h2o(relevance = relevance,
                                                                             h2o_fs_models = h2o_fs_models,
                                                                             encoding_fs = encoding_fs)
            
            train, test = train[sel_cols], test[sel_cols]
            
        #### Encoding Method Selection    
        ptn = Pattern(train = train,
                      test = test,
                      target = self.target)          
        self.enc_method, perf_ = ptn.encoding_selection(), ptn.perf
        
        ### Null Imputation Selection 
        if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
            train, test = ptn.imputation_selection()
            self.imp_method, perf_ = ptn.imp_method, ptn.perf
        else:
            self.imp_method='Undefined'
            ev = Encoding_Version(train = train,
                                  test = test,
                                  target = self.target)
            if self.enc_method == 'Encoding Version 1':
                train, test = ev.encoding_v1()
            elif self.enc_method == 'Encoding Version 2':
                train, test = ev.encoding_v2()
            elif self.enc_method == 'Encoding Version 3':
                train, test = ev.encoding_v3()
            elif self.enc_method == 'Encoding Version 4':
                train, test = ev.encoding_v4()
            print('There are no missing values in the Dataset')  

        ### Variance Inflation Factor (VIF) Application 
        train, test = Pattern(train = train,
                              test = test,
                              target = self.target).vif_performance(vif_threshold = vif_ratio,
                                                                    perf_ = perf_)
                                                                    
        self.cols = [col for col in train.columns if col != self.target]
    
        ### Fit Processors 
        data = data[self.cols]
        self.n_cols = list(super().num_cols(X=data))
        self.c_cols = list(super().cat_cols(X=data))
        
        ## Fit Encoding Version
        if len(self.n_cols) > 0:
            
            if self.enc_method == 'Encoding Version 1' or self.enc_method == 'Encoding Version 3':
                    self.scaler = AutoStandardScaler()
                    self.scaler.fit(X = data[self.n_cols])
                    data[self.n_cols] = self.scaler.transform(X = data[self.n_cols])
                    
            if self.enc_method == 'Encoding Version 2' or self.enc_method == 'Encoding Version 4':
                    self.scaler = AutoMinMaxScaler()
                    self.scaler.fit(X=data[self.n_cols])
                    data[self.n_cols] = self.scaler.transform(X = data[self.n_cols])
                    
        if len(self.c_cols) > 0:
            
            if self.enc_method == 'Encoding Version 1' or self.enc_method == 'Encoding Version 2':
                    self.encoder = AutoIFrequencyEncoder()
                    self.encoder.fit(data[self.c_cols])
                    data = self.encoder.transform(X = data)
                    
            if self.enc_method == 'Encoding Version 3' or self.enc_method == 'Encoding Version 4':
                    self.encoder = AutoLabelEncoder()
                    self.encoder.fit(data[self.c_cols])
                    data=self.encoder.transform(X = data)
            
        ## Fit Null Imputation
        if self.imp_method == 'Simple':
            self.imputer = AutoSimpleImputer(strategy='mean')
            self.imputer.fit(data)  # Fit on DataFrame
        
        elif self.imp_method == 'KNN':
            self.imputer = AutoKNNImputer(n_neighbors=3,
                                          weights="uniform")
            self.imputer.fit(data)  # Fit on DataFrame
        
        elif self.imp_method == 'Iterative':
            self.imputer = AutoIterativeImputer(max_iter=10,
                                                random_state=42,
                                                initial_strategy="mean",
                                                imputation_order="ascending")
            self.imputer.fit(data)  # Fit on DataFrame
        
        return self
        
    def data_processing(self, X : pd.DataFrame):
        """
        Apply the fitted preprocessing steps to the loaded dataset.

        This method transforms a new dataset using the preprocessing steps fitted in the `fit_processing` method.
        It handles date feature engineering, encoding, scaling, and imputation based on the fitted objects and 
        selected columns.

        Parameters
        ----------
        X : pd.DataFrame
            The new dataset to be processed.

        Returns
        -------
        pd.DataFrame
            The transformed dataset.
        """
        ### Transformation Proceedment
        data = X.copy()
        data = Analysis.engin_date(X = data,
                                   drop = True)
        
        if self.target in list(data.columns):
            data = data[self.cols + [self.target]]
        else: 
            data = data[self.cols]
            
        if len(self.n_cols) > 0:
            data[self.n_cols] = self.scaler.transform(data[self.n_cols])
        if len(self.c_cols) > 0:
            data = self.encoder.transform(X = data)
        if self.imp_method != "Undefined" and data[self.c_cols + self.n_cols].isnull().sum().sum() != 0:
            data = self.imputer.transform(data.copy())
    

    
        return data


__all__ = [
    'Analysis',
    'AutoMinMaxScaler',
    'AutoStandardScaler',
    'AutoRobustScaler',
    'AutoLabelEncoder',
    'AutoIFrequencyEncoder',
    'AutoOneHotEncoder',
    'Encoding_Version',
    'AutoSimpleImputer',
    'AutoKNNImputer',
    'AutoIterativeImputer',
    'Selector',
    'Pattern'
]

    