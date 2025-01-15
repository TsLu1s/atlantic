import pandas as pd
from atlantic.processing.versions import Encoding_Version 
from atlantic.imputers.imputation import AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer 
from atlantic.optimizer.evaluation import Evaluation 
from atlantic.feature_selection.selector import Selector 
from tqdm import tqdm

class Pattern(Evaluation):
    def __init__(self, 
                 train : pd.DataFrame, 
                 test : pd.DataFrame, 
                 target : str):
        super().__init__(train,test,target)
        self.enc_method = None
        self.imp_method = None
        self._imputer = None
        self.perf = None
        """
        Provides a sequentially optimized application of methods for selecting optimal encoding and imputation strategies, 
        as well as performing feature selection based on Variance Inflation Factor (VIF).
    
        Methods
        -------
        encoding_selection():
            Selects the best encoding method based on predictive performance.
        imputation_selection():
            Selects the best imputation method based on predictive performance.
        vif_performance(vif_threshold=10.0, perf_=None):
            Evaluates the Variance Inflation Factor (VIF) and applies VIF-based feature selection.
        """
        
    def encoding_selection(self):
        """
        Select the optimal encoding method based on predictive performance.

        This method generates different encoding versions of the training and testing datasets,
        imputes missing values if necessary, and evaluates the performance of each encoding version.
        The best-performing encoding method is selected and stored in the `enc_method` attribute.

        """
        # Create an Encoding_Version instance and a SimpleImputer instance.
        ev = Encoding_Version(train = self.train.copy(),
                              test = self.test.copy(),
                              target = self.target)
        simple_imputer = AutoSimpleImputer(strategy = 'mean', target = self.target)
        
        # Generate different encoding versions and separate train and test data for each encoding version.
        # Initialize the progress bar
        print('  ')        
        encoding_versions = [ev.encoding_v1, ev.encoding_v2, ev.encoding_v3, ev.encoding_v4]
        with tqdm(total=1, desc="Fitting Encoding Versions", ncols=75) as pbar:
            (train_v1, test_v1), (train_v2, test_v2), (train_v3, test_v3), (train_v4, test_v4) = [method() for method in encoding_versions]
            pbar.update(1)
            
        # Impute missing values if necessary.
        if (train_v1.isnull().sum().sum() or test_v1.isnull().sum().sum()) != 0:
            train_v1, test_v1 = simple_imputer.impute(train = train_v1,
                                                      test = test_v1)
        
        if (train_v2.isnull().sum().sum() or test_v2.isnull().sum().sum()) != 0:
            train_v2, test_v2 = simple_imputer.impute(train = train_v2,
                                                      test = test_v2)
            
        if (train_v3.isnull().sum().sum() or test_v3.isnull().sum().sum()) != 0:
            train_v3, test_v3 = simple_imputer.impute(train = train_v3,
                                                      test = test_v3)
            
        if (train_v4.isnull().sum().sum() or test_v4.isnull().sum().sum()) != 0:
             train_v4, test_v4 = simple_imputer.impute(train = train_v4,
                                                       test = test_v4)
            
        # Perform model evaluation for each encoding version.
        print('    ')
        encoding_versions = [
            (train_v1, test_v1),
            (train_v2, test_v2),
            (train_v3, test_v3),
            (train_v4, test_v4)
        ]
        
        performances = []
        for i, (train, test) in enumerate(encoding_versions, 1):
            print(f'Encoding Version {i} Loading')
            performance = Evaluation(train=train, test=test, target=self.target).auto_evaluate()[self.eval_metric][0]
            performances.append(performance)
        
        p_v1, p_v2, p_v3, p_v4 = performances
        
        metric = 'MAE' if self.pred_type == 'Reg' else 'F1 Score' if self.n_classes > 2 else 'Precision'
        
        print('\nPredictive Performance Encoding Versions:')
        print(f'\n Version 1 [IFrequency + StandardScaler] : {round(p_v1, 4)}',
              f'\n Version 2 [IFrequency + MinMaxScaler] : {round(p_v2, 4)}',
              f'\n Version 3 [Label + StandardScaler] : {round(p_v3, 4)}',
              f'\n Version 4 [Label + MinMaxScaler] : {round(p_v4, 4)}')
        
        list_encoding=[p_v1,p_v2,p_v3,p_v4]
        
        if self.pred_type == 'Class':
            list_encoding.sort(reverse=True)
        else:
            list_encoding.sort()
        
        # Select the encoding method with the best performance.
        if list_encoding[0] == p_v1:
            self.enc_method = 'Encoding Version 1'
        elif list_encoding[0] == p_v2:
            self.enc_method = 'Encoding Version 2'
        elif list_encoding[0] == p_v3:
            self.enc_method = 'Encoding Version 3'
        elif list_encoding[0] == p_v4:
            self.enc_method = 'Encoding Version 4'  
        self.perf = list_encoding[0]
        
        print(f'{self.enc_method} was selected with a', metric, 'of: ', round(self.perf, 4))
        
        return self.enc_method
        
    def imputation_selection(self):
        """
        Select the optimal imputation method based on predictive performance.

        This method applies the selected encoding method to the training and testing datasets,
        performs imputation using different imputation strategies, and evaluates their performance.
        The best-performing imputation method is selected and stored in the `imp_method` attribute.

        """
        # Select the imputation method and assess its performance.
        metric = 'MAE' if self.pred_type == 'Reg' else 'F1 Score' if self.n_classes > 2 else 'Precision'
        
        ev = Encoding_Version(train=self.train.copy(), test=self.test.copy(), target=self.target)
        # Depending on the selected encoding method, apply the corresponding encoding function.
        if self.enc_method == 'Encoding Version 1':
            self.train, self.test = ev.encoding_v1()
        elif self.enc_method == 'Encoding Version 2':
            self.train, self.test=ev.encoding_v2()
        elif self.enc_method == 'Encoding Version 3':
            self.train, self.test = ev.encoding_v3()
        elif self.enc_method == 'Encoding Version 4':
            self.train, self.test = ev.encoding_v4()
        elif self.enc_method == None:
            print(" No Encoding Version Selected ")
            self.train, self.test = ev.encoding_v4()
            
        print('    ')
        print('Simple Imputation Loading')
        simple_imputer = AutoSimpleImputer(strategy='mean')
        simple_imputer.fit(self.train)  # Fit on the DataFrame
        train_s = simple_imputer.transform(self.train.copy())  # Transform the Train DataFrame
        test_s = simple_imputer.transform(self.test.copy()) # Transform the Test DataFrame
        
        p_s = Evaluation(train = train_s,
                         test = test_s,
                         target = self.target).auto_evaluate()[self.eval_metric][0]
        
        print('KNN Imputation Loading')
        knn_imputer = AutoKNNImputer(n_neighbors=3)
        knn_imputer.fit(self.train)  # Fit on the DataFrame
        train_knn = knn_imputer.transform(self.train.copy())  # Transform the Train DataFrame
        test_knn = knn_imputer.transform(self.test.copy()) # Transform the Test DataFrame
        
        p_knn = Evaluation(train = train_knn,
                           test = test_knn,
                           target = self.target).auto_evaluate()[self.eval_metric][0]
        
        print('Iterative Imputation Loading')
        iter_imputer = AutoIterativeImputer(max_iter=10, random_state=42)
        iter_imputer.fit(self.train)  # Fit on the DataFrame
        train_iter = iter_imputer.transform(self.train.copy())  # Transform the Train DataFrame
        test_iter = iter_imputer.transform(self.test.copy()) # Transform the Test DataFrame
        
        p_iter = Evaluation(train = train_iter,
                            test = test_iter,
                            target = self.target).auto_evaluate()[self.eval_metric][0]
            
        print('    ')
        print('Predictive Performance Null Imputation Versions:')
        print('    ')
        print(' KNN Performance: ', round(p_knn, 4), 
              '\n Iterative Performance: ', round(p_iter, 4),
              '\n Simple Performance: ', round(p_s, 4))
        
        list_imp=[p_iter,p_knn,p_s]
        list_imp.sort()
        if self.pred_type == 'Class':
            list_imp.sort(reverse=True)
        
        # Select the imputation method with the best performance.
        if list_imp[0] == p_s:
            self.imp_method = 'Simple'
            self.train, self.test, self._imputer = train_s.copy(), test_s.copy(), simple_imputer
            
        elif list_imp[0] == p_knn:
            self.imp_method = 'KNN'
            self.train, self.test, self._imputer = train_knn.copy(), test_knn.copy(), knn_imputer

        elif list_imp[0] == p_iter:
            self.imp_method = 'Iterative'
            self.train, self.test,self._imputer=train_iter.copy(),test_iter.copy(),iter_imputer

        self.perf = list_imp[0]
        print(f'{self.imp_method} Imputation Algorithm was selected with a', metric, 'of:', round(self.perf, 4))
        
        return self.train, self.test
    
    def vif_performance(self,
                        vif_threshold : float = 10.0,
                        perf_ : float = None):
        """
        Evaluate and apply Variance Inflation Factor (VIF) based feature selection.

        This method evaluates the performance before and after applying VIF-based feature selection.
        It selects features with VIF below the specified threshold and compares performance.
        If performance improves or remains stable, VIF-based feature selection is applied.

        """

        train_vif, test_vif = self.train.copy(), self.test.copy()

        cols_vif = Selector(X = train_vif,
                            target = self.target).feature_selection_vif(vif_threshold = vif_threshold)
        print('    ')
        print('Selected VIF Columns: ', len(cols_vif),
              '\nRemoved Columns by VIF :', train_vif.shape[1] - len(cols_vif))
        
        train_vif, test_vif = train_vif[cols_vif], test_vif[cols_vif]
        apply_vif = False
        
        if self.train.shape[1] - len(cols_vif) == 0:
            self.train = self.train[cols_vif]
            self.test = self.test[cols_vif]
            
        else:
            if perf_ != None: 
                self.perf = perf_
            if self.perf == None:
                self.perf = Evaluation(train = self.train.copy(),
                                       test = self.test.copy(),
                                       target = self.target).auto_evaluate()[self.eval_metric][0]
                
            perf_vif = Evaluation(train = train_vif,
                                  test = test_vif,
                                  target = self.target).auto_evaluate()[self.eval_metric][0]
            
            print('Default Performance:', round(self.perf, 4))
            print('VIF Performance:', round(perf_vif, 4))
    
            if self.pred_type == 'Reg' and perf_vif < self.perf:
                apply_vif = True
            elif self.pred_type == 'Class' and perf_vif > self.perf:
                apply_vif = True
            
            if apply_vif:
                print('')
                print('The VIF filtering method was applied')
                self.train = self.train[cols_vif]
                self.test = self.test[cols_vif]
            else:
                print('')
                print('The VIF filtering method was not applied')
    
        return self.train, self.test
    

    