import pandas as pd
from atlantic.processing.encoders import AutoLabelEncoder, AutoIFrequencyEncoder #, AutoOneHotEncoder
from atlantic.processing.scalers import AutoMinMaxScaler, AutoStandardScaler #, AutoRobustScaler

class Encoding_Version:
    def __init__(self, train : pd.DataFrame, test : pd.DataFrame, target : str):
        self.train = train
        self.test = test
        self.target = target
        self.cat_cols = [col for col in self.train.select_dtypes(include=['object','category']).columns if col != self.target]
        self.num_cols = [col for col in self.train.select_dtypes(include=['int','int32', 'int64','float','float32', 'float64']).columns if col != self.target]
        """
        Initialize the Encoding_Version class with training and testing datasets and the target column.
        Is responsible for setting up the initial state of the Encoding_Version object. 
        It accepts the training and testing datasets, along with the target column name. During initialization,
        it also identifies and segregates the categorical and numerical columns within the datasets, excluding
        the target column. This segregation facilitates the application of appropriate encoding strategies to 
        different types of data.
        """
        
    @staticmethod
    def apply_encoder(train, test, encoder_, cols):
        """
        Apply a specified encoder to the given columns of the training and testing datasets.
        
        This method is responsible for applying a chosen encoding or scaling technique to a subset of columns 
        within the training and testing datasets. It does so by instantiating the encoder or scaler class, 
        fitting it on the training data, and then transforming both the training and testing datasets using 
        the fitted encoder or scaler. This process ensures that the transformations applied to the datasets are
        consistent and based on the characteristics of the training data.
        """
        if len(cols) > 0:
            encoder_instance = encoder_()  # Instantiate the selected encoder
            encoder_instance.fit(X=train[cols])
            train[cols] = encoder_instance.transform(X=train[cols].copy())
            test[cols] = encoder_instance.transform(X=test[cols].copy())

    def encoding_v1(self):
        """
        Apply Standard Scaling to numerical columns and Inverse Frequency Encoding to categorical columns.
        
        This method generates a version of the training and testing datasets where numerical features are scaled
        using the standard scaling technique and categorical features are encoded using the Inverse Frequency
        encoding method. The standard scaling transforms numerical features to have zero mean nd unit variance,
        a which is crucial for many machine learning algorithms. The IDF encoding transforms categorical features
        based on the frequency of terms, providing a more informative encoding than simple one-hot encoding.
        """
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoStandardScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoIFrequencyEncoder, self.cat_cols)
        return train_enc, test_enc

    def encoding_v2(self):
        """
        Apply MinMax Scaling to numerical columns and Inverse Frequency Encoding to categorical columns.
        
        This method generates a version of the training and testing datasets where numerical features are scaled
        using the MinMax scaling technique and categorical features are encoded using the Inverse Document 
        Frequency (IDF) encoding method. The MinMax scaling transforms numerical features to a specified range 
        (usually between 0 and 1), which can be useful for algorithms that require normalized input. The IDF 
        encoding provides a more informative representation of categorical features.
        """
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoMinMaxScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoIFrequencyEncoder, self.cat_cols)
        return train_enc, test_enc

    def encoding_v3(self):
        """
        Apply Standard Scaling to numerical columns and Label Encoding to categorical columns.
        
        This method generates a version of the training and testing datasets where numerical features are scaled
        using the standard scaling technique and categorical features are encoded using the label encoding 
        method. The standard scaling ensures that numerical features have zero mean and unit variance, which is
        crucial for many machine learning algorithms. Label encoding converts categorical features into 
        numerical values by assigning a unique integer to each category, which can be useful for certain types 
        of models.
        """
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoStandardScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoLabelEncoder, self.cat_cols)
        return train_enc, test_enc

    def encoding_v4(self):
        """
        Apply MinMax Scaling to numerical columns and Label Encoding to categorical columns.
        
        This method generates a version of the training and testing datasets where numerical features are scaled
        using the MinMax scaling technique and categorical features are encoded using the label encoding method. 
        The MinMax scaling transforms numerical features to a specified range (usually between 0 and 1), which 
        can be useful for algorithms that require normalized input. Label encoding converts categorical features 
        into numerical values by assigning a unique integer to each category.
        """
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoMinMaxScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoLabelEncoder, self.cat_cols)
        return train_enc, test_enc
