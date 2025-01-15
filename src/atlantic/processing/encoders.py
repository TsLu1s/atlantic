import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
       
class AutoLabelEncoder(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoLabelEncoder class. This class provides functionality for automatically
        encoding categorical features using Label Encoding, where each unique category is converted to a numerical value.
        
        Attributes:
        label_encoders (dict): A dictionary to store the LabelEncoder instance for each column.
        columns (Index): Stores the column names of the DataFrame being transformed.
        """
        self.label_encoders={}
        self.columns=None
    
    def fit(self, X, y=None):
        """
        Fit the LabelEncoders to each column in the DataFrame. This method prepares the encoder for transforming the
        categorical data into numerical data.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be encoded.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoLabelEncoder.
        """
        self.columns=X.columns
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            #Store the original classes and add a new label for unseen values
            le.classes_ = np.append(le.classes_, "Unknown")
            self.label_encoders[col] = le
            
        return self
    
    def transform(self, X, y=None):
        """
        Transform the categorical features in the DataFrame into numerical values using the fitted LabelEncoders. This
        method also handles unseen values by encoding them as "Unknown".
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be transformed.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the categorical features encoded as numerical values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_encoded = X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            # Use the "UNKNOWN" label for previously unseen values
            X_encoded[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["Unknown"])[0])

        return X_encoded
    
    def inverse_transform(self, X):
        """
        Inverse transform the numerical values back to the original categorical values using the fitted LabelEncoders.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be inverse transformed.
        
        Returns:
        pd.DataFrame: A DataFrame with the numerical features decoded back to their original categorical values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_decoded=X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            X_decoded[col]=le.inverse_transform(X[col])
            
        return X_decoded
        
class AutoOneHotEncoder(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoOneHotEncoder class. This class provides functionality for automatically
        encoding categorical features using One-Hot Encoding, where each unique category is converted to a binary vector.
        
        Attributes:
        one_hot_encoders (dict): A dictionary to store the OneHotEncoder instance for each column.
        columns (Index): Stores the column names of the DataFrame being transformed.
        decoded (pd.DataFrame): A copy of the original DataFrame before transformation, used for inverse transformation.
        """
        self.one_hot_encoders = {}
        self.columns = None
        self.decoded = None

    def fit(self, X, y=None):
        """
        Fit the OneHotEncoders to each column in the DataFrame. This method prepares the encoder for transforming the
        categorical data into binary vectors.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be encoded.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoOneHotEncoder.
        """
        self.columns = X.columns
        
        for col in X.columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(X[col].values.reshape(-1, 1))
            self.one_hot_encoders[col] = encoder
            
        return self

    def transform(self, X, y=None):
        """
        Transform the categorical features in the DataFrame into binary vectors using the fitted OneHotEncoders. This
        method also handles unseen values by adding columns of zeros for unknown categories.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be transformed.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the categorical features encoded as binary vectors.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")

        X_encoded = X.copy()
        self.decoded = X.copy()

        for col in self.columns:
            if col in X_encoded.columns:
                encoder = self.one_hot_encoders[col]
                encoded_col = encoder.transform(X_encoded[col].values.reshape(-1, 1))

                # Handle unknown categories by adding a column of zeros
                if len(encoder.get_feature_names_out([col])) > 1:
                    unknown_cols = [col for col in encoder.get_feature_names_out([col]) if
                                    col.startswith(f"{col}_Unknown")]
                    for unknown_col in unknown_cols:
                        X_encoded[unknown_col] = 0

                encoded_columns = encoder.get_feature_names_out([col])
                X_encoded[encoded_columns] = encoded_col

                # Drop the original column
                X_encoded.drop(columns=[col], inplace=True)

        return X_encoded
    
    def inverse_transform(self, X):
        """
        Inverse transform the binary vectors back to the original categorical values using the fitted OneHotEncoders.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the binary vectors to be inverse transformed.
        
        Returns:
        pd.DataFrame: The DataFrame with the original categorical values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")

        X_decoded = pd.DataFrame()

        for col in self.columns:
            encoder = self.one_hot_encoders[col]
            encoded_columns = encoder.get_feature_names_out([col])
            encoded_data = X[encoded_columns].values

            # Get the original categories back from the encoded data
            decoded_col = encoder.inverse_transform(encoded_data)
            X_decoded[col] = decoded_col.flatten()

        return X_decoded
        
class AutoIFrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoFrequencyEncoder class. This class provides functionality for encoding categorical features
        using a frequency-based approach, and then applying a logarithmic transformation to set importance based on incidence.
        
        Attributes:
        freq_fit (dict): A dictionary to store the frequency mapping for each column.
        columns (Index): Stores the column names of the DataFrame being transformed.
        ratio (float): Scaling ratio to adjust the log transformation.
        """
        self.freq_fit = None
        self.columns = None
        self._ratio = 1

    def fit(self, X, y=None):
        """
        Fit the encoder to the input DataFrame X to compute the frequency of each category.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing categorical features to be encoded.
        y: Ignored.
        
        Returns:
        self: Returns the instance itself.
        """
        self.columns = X.columns
        self.X_size = X.shape[0]  # Store the size of the training data

        if self.columns is None:
            raise ValueError("You must specify categorical columns when creating the AutoFrequencyEncoder instance.")
        
        # Calculate frequency of each category
        self.freq_fit = {}
        for col in self.columns:
            frequency = X[col].value_counts().to_dict()
            self.freq_fit[col] = frequency
        
        return self

    def transform(self, X):
        """
        Transform the input DataFrame X using the fitted frequency mapping and apply a logarithmic transformation.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing categorical features to be encoded.
        
        Returns:
        pd.DataFrame: The encoded DataFrame with frequency values transformed by a logarithmic function.
        """
        if self.freq_fit is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_encoded = X.copy()
        self.decoded = X.copy()
        
        for col in self.columns:
            # Map frequencies to the column values
            frequency_map = self.freq_fit[col]
            
            # Apply log transformation to the frequencies
            def transform_value(x):
                if pd.isna(x):
                    return x
                if x in frequency_map:
 
                    return np.log1p(int(self.X_size * self._ratio) / frequency_map[x])
                else:
                    return np.log1p(int(self.X_size * self._ratio) / min(frequency_map.values()))
            
            X_encoded[col] = X[col].map(transform_value)
            
        return X_encoded
    
    def inverse_transform(self, X):
        """
        Inverse transform the frequency-encoded DataFrame back to the original categorical values using the fitted frequency mappings.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the frequency-encoded values to be inverse transformed.
        
        Returns:
        pd.DataFrame: The DataFrame with the original categorical values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.freq_fit is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_decoded = pd.DataFrame()

        for col in self.columns:
            # Invert the log transformation and map back to original values
            frequency_map = self.freq_fit[col]
            inverse_frequency_map = {np.log1p(int(self.X_size * self._ratio) / v): k for k, v in frequency_map.items()}
            sorted_freq_keys = sorted(inverse_frequency_map.keys(), reverse=True)
            
            def inverse_transform_value(x):
                if pd.isna(x):
                    return x
                closest_log_val = min(sorted_freq_keys, key=lambda log_val: abs(log_val - x))
                if abs(closest_log_val - x) > 1e-6:  
                    return 'unknown'
                return inverse_frequency_map[closest_log_val]
            
            X_decoded[col] = X[col].map(inverse_transform_value)
            
        return X_decoded