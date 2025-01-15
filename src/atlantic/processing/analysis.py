import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field
from typing import Tuple, List, Literal

@dataclass
class AnalysisConfig:
    """Configuration for Analysis class containing common parameters and constants."""
    
    NUMERIC_TYPES: List[str] = field(default_factory=lambda: [
        'int', 'int32', 'int64', 'float', 'float32', 'float64'
    ])
    DATE_COMPONENTS: List[str] = field(default_factory=lambda: [
        'day_of_month', 'day_of_week', 'is_wknd', 'month',
        'day_of_year', 'year', 'hour', 'minute', 'second'
    ])
    MIN_SPLIT_RATIO: float = 0.5
    MAX_SPLIT_RATIO: float = 0.98

class Analysis:
    def __init__(self, target: str = None):
        """
        Initialize an instance of the Analysis class. The Analysis class is designed to facilitate the preprocessing
        and preparation of datasets for machine learning tasks. It includes methods for splitting datasets,
        encoding target variables, and engineering features.
        
        Parameters:
        target (str): The name of the target variable column. This is the dependent variable that the model will learn to predict.
        
        Attributes:
        target (str): Stores the name of the target variable provided during initialization. This attribute is used by other methods
                      in the class to identify and manipulate the target variable.
        _label_encoder (LabelEncoder): A private attribute that is used to encode categorical target variables. It is initialized as None
                                       and is instantiated when needed.
        n_dtypes (list): A list of numerical data types. This attribute helps in identifying numerical columns in the dataset.
                         The list includes common integer and floating-point data types used in pandas DataFrames.
        """
        self.target = target
        self._label_encoder = None
        self.config = AnalysisConfig()

    def _validate_dataframe(self, X: pd.DataFrame) -> None:
        """
        Validate input DataFrame for required properties.
        
        Args:
            X: Input DataFrame to validate
            
        Raises:
            ValueError: If DataFrame validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if self.target is not None:
            if self.target not in X.columns:
                raise ValueError(f"Target column '{self.target}' not found in DataFrame")

    def split_dataset(
            self, 
            X: pd.DataFrame,
            split_ratio: float = 0.75
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and testing sets based on the specified split ratio. The method ensures that the split ratio
        is within a sensible range and handles any missing values in the target column by dropping those rows.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the features and the target variable. It is expected that the DataFrame
                          includes a column with the name matching the target attribute.
        split_ratio (float): The proportion of the dataset to be used for the training set. The value must be between 0.5 and 0.98
                             to ensure that both training and testing sets are sufficiently large for meaningful analysis.
        """
        # Splits the dataset into train and test sets based on the split_ratio.
        if not self.config.MIN_SPLIT_RATIO <= split_ratio <= self.config.MAX_SPLIT_RATIO:
            raise ValueError(
                f"split_ratio must be between {self.config.MIN_SPLIT_RATIO} and "
                f"{self.config.MAX_SPLIT_RATIO}"
            )
            
        self._validate_dataframe(X)
        
        if self.target:
            X = X.dropna(subset=[self.target])
            
        if self.pred_type=='Class':
            return train_test_split(X,
                                    train_size=split_ratio,
                                    stratify=X[self.target])
        else:
            return train_test_split(X, 
                                    train_size = split_ratio)

    def divide_dfs(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Separate features and target variables from training and testing sets.
        
        Args:
            train: Training DataFrame containing features and target
            test: Testing DataFrame containing features and target
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If target is not set or DataFrames are invalid
        """
            
        self._validate_dataframe(train)
        self._validate_dataframe(test)
        
        X_train, X_test = train.drop(self.target, axis=1), test.drop(self.target, axis=1)
        y_train, y_test = train[self.target], test[self.target]
        
        if self.target_type(X = train)[0] == "Class":
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            y_test = self._label_encoder.transform(y_test)
        
        return X_train, X_test, y_train, y_test

    def target_type(self, X: pd.DataFrame) -> Tuple[Literal["Reg", "Class"], str]:
        """
        Determines the type of the target variable and the appropriate evaluation metric based on the target's data type.
        If the target variable is numerical, it is considered a regression problem. If it is categorical, it is considered a
        classification problem.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the target variable.
        
        """
        self._validate_dataframe(X)
        
        target_dtype = X[self.target].dtype
        pred_type, eval_metric = 'Reg', 'Mean Absolute Error'
        
        if target_dtype not in self.config.NUMERIC_TYPES:
            pred_type, eval_metric = 'Class', 'Precision'
            self.n_classes = len(pd.unique(X[self.target])) if pred_type == "Class" else None
            if self.n_classes > 2:
                eval_metric = "F1"
            
        return pred_type, eval_metric

    def num_cols(self, X: pd.DataFrame):
        """
        Get list of numerical columns, excluding target variable.
        
        Args:
            X: Input DataFrame
            
        Returns:
            List of numerical column names
        """
        return [col for col in X.select_dtypes(include=self.config.NUMERIC_TYPES).columns if col != self.target]

    def cat_cols(self, X: pd.DataFrame):
        """
        Get list of categorical columns, excluding target variable.
        
        Args:
            X: Input DataFrame
            
        Returns:
            List of categorical column names
        """
        return [col for col in X.select_dtypes(include=['object','category']).columns if col != self.target]

    @staticmethod
    def remove_columns_by_nulls(X: pd.DataFrame, percentage: float):
        """
        Remove columns with null values exceeding specified percentage.
        
        Args:
            X: Input DataFrame
            percentage: Maximum allowed percentage of null values
            
        Returns:
            DataFrame with high-null columns removed
            
        Raises:
            ValueError: If percentage is outside valid range
        """
        min_count = int((1 - percentage / 100) * X.shape[0])
        return X.dropna(axis=1, thresh=min_count)
    
    @staticmethod
    def engin_date(X: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.
        
        Args:
            X: Input DataFrame
            drop: Whether to drop original datetime columns
            
        Returns:
            DataFrame with engineered date features
        """
        X = X.copy()
        # Identify datetime columns in the DataFrame.
        datetime_columns = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        for col in datetime_columns:
            # Convert datetime values to a standardized format (Year-Month-Day Hour:Minute:Second).
            X[col] = pd.to_datetime(X[col].dt.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Create additional date-related features.
            X[col + '_day_of_month'] = X[col].dt.day
            X[col + '_day_of_week'] = X[col].dt.dayofweek + 1
            X[col + '_is_wknd'] = X[col + '_day_of_week'].isin([6, 7]).astype(int)
            X[col + '_month'] = X[col].dt.month
            X[col + '_day_of_year'] = X[col].dt.dayofyear
            X[col + '_year'] = X[col].dt.year
            X[col + '_hour'] = X[col].dt.hour
            X[col + '_minute'] = X[col].dt.minute
            X[col + '_second'] = X[col].dt.second
            
            # Drop the original datetime column if 'drop' is set to True.
            if drop:
                X = X.drop(columns=col)
        
        return X
    
    
