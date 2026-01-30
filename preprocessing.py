"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.processed_df = None
        self.feature_matrix = None
        
    def clean_data(self, df):
        """
        Clean and preprocess raw data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw laptop data
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataframe
        """
        df = df.copy()
        
        # Fill missing values in numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing values in categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Handle Graphics_GB (some are missing)
        df['Graphics_GB'] = pd.to_numeric(df['Graphics_GB'], errors='coerce').fillna(0)
        
        self.processed_df = df
        return df
    
    def create_feature_matrix(self, df):
        """
        Create feature matrix for similarity calculation
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed dataframe
            
        Returns:
        --------
        numpy.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        # Numerical features to normalize
        numerical_features = [
            'Price', 'Rating', 'Processor_gen', 'Core_per_processor',
            'Total_processor', 'Execution_units', 'Threads', 'RAM_GB',
            'Storage_capacity_GB', 'Graphics_GB', 'Display_size_inches',
            'Horizontal_pixel', 'Vertical_pixel', 'ppi'
        ]
        
        # Categorical features to encode
        categorical_features = [
            'Brand', 'Processor_brand', 'RAM_type', 'Storage_type',
            'Graphics_brand', 'Operating_system'
        ]
        
        feature_list = []
        
        # Process numerical features (normalize using StandardScaler)
        for feat in numerical_features:
            if feat in df.columns:
                values = df[feat].values.reshape(-1, 1)
                normalized = self.scaler.fit_transform(values)
                feature_list.append(normalized.flatten())
        
        # Process categorical features (encode and normalize)
        for feat in categorical_features:
            if feat in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[feat].astype(str))
                # Normalize encoded values
                encoded_normalized = (encoded - encoded.mean()) / (encoded.std() + 1e-8)
                feature_list.append(encoded_normalized)
        
        # Process binary features
        if 'Touch_screen' in df.columns:
            touch = pd.to_numeric(
                df['Touch_screen'].replace({True: 1, False: 0, 'True': 1, 'False': 0}), 
                errors='coerce'
            ).fillna(0).astype(int).values
            feature_list.append(touch)
        
        if 'Graphics_integreted' in df.columns:
            integrated = pd.to_numeric(
                df['Graphics_integreted'].replace({True: 1, False: 0, 'True': 1, 'False': 0}), 
                errors='coerce'
            ).fillna(0).astype(int).values
            feature_list.append(integrated)
        
        # Combine all features into a single matrix
        self.feature_matrix = np.column_stack(feature_list)
        
        return self.feature_matrix
    
    def get_processed_data(self):
        """Get the processed dataframe"""
        return self.processed_df
    
    def get_feature_matrix(self):
        """Get the feature matrix"""
        return self.feature_matrix

