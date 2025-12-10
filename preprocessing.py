"""
Preprocessing Utilities for Machine Learning
Tugas Besar 2 IF3070 – Dasar Inteligensi Artifisial

This module contains all preprocessing utilities implemented from scratch
using only numpy and pandas.

Author: AbyuDAIya-Ganbatte Team
"""

import numpy as np
import pandas as pd
import json


# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Manually split data into training and testing sets.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    y : np.ndarray or pd.Series
        Target vector
    test_size : float
        Proportion of data for testing (0.0 to 1.0)
    random_state : int or None
        Seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple of arrays
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.flatten()
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate shuffled indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing_values(df, numerical_cols=None, categorical_cols=None):
    """
    Handle missing values using mean imputation for numerical columns
    and mode imputation for categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with potential missing values
    numerical_cols : list or None
        List of numerical column names (auto-detected if None)
    categorical_cols : list or None
        List of categorical column names (auto-detected if None)
    
    Returns:
    --------
    df_filled : pd.DataFrame
        DataFrame with imputed values
    imputation_values : dict
        Dictionary storing the imputation values for each column
    """
    df_filled = df.copy()
    imputation_values = {}
    
    # Auto-detect column types if not provided
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Mean imputation for numerical columns
    for col in numerical_cols:
        if col in df_filled.columns and df_filled[col].isnull().any():
            mean_value = df_filled[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_value)
            imputation_values[col] = {'type': 'mean', 'value': mean_value}
    
    # Mode imputation for categorical columns
    for col in categorical_cols:
        if col in df_filled.columns and df_filled[col].isnull().any():
            # Get the mode (most frequent value)
            mode_value = df_filled[col].mode()
            if len(mode_value) > 0:
                mode_value = mode_value[0]
                df_filled[col] = df_filled[col].fillna(mode_value)
                imputation_values[col] = {'type': 'mode', 'value': mode_value}
    
    return df_filled, imputation_values


def apply_imputation(df, imputation_values):
    """
    Apply previously computed imputation values to new data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to impute
    imputation_values : dict
        Dictionary of imputation values from handle_missing_values
    
    Returns:
    --------
    df_filled : pd.DataFrame
    """
    df_filled = df.copy()
    
    for col, info in imputation_values.items():
        if col in df_filled.columns and df_filled[col].isnull().any():
            df_filled[col] = df_filled[col].fillna(info['value'])
    
    return df_filled


# =============================================================================
# ONE-HOT ENCODING
# =============================================================================

def one_hot_encode(df, categorical_cols, drop_first=False):
    """
    Perform One-Hot Encoding on categorical columns using pandas only.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical columns
    categorical_cols : list
        List of column names to encode
    drop_first : bool
        Whether to drop the first category to avoid multicollinearity
    
    Returns:
    --------
    df_encoded : pd.DataFrame
        DataFrame with encoded columns
    encoding_info : dict
        Information about the encoding for each column
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Get unique categories
            categories = df_encoded[col].unique()
            categories = [c for c in categories if pd.notna(c)]  # Remove NaN
            categories = sorted(categories)  # Sort for consistency
            
            encoding_info[col] = categories
            
            # Create dummy columns manually
            for i, category in enumerate(categories):
                if drop_first and i == 0:
                    continue
                new_col_name = f"{col}_{category}"
                df_encoded[new_col_name] = (df_encoded[col] == category).astype(int)
            
            # Drop original column
            df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded, encoding_info


def apply_one_hot_encode(df, encoding_info, drop_first=False):
    """
    Apply previously fitted one-hot encoding to new data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to encode
    encoding_info : dict
        Encoding information from one_hot_encode
    drop_first : bool
        Whether to drop the first category
    
    Returns:
    --------
    df_encoded : pd.DataFrame
    """
    df_encoded = df.copy()
    
    for col, categories in encoding_info.items():
        if col in df_encoded.columns:
            for i, category in enumerate(categories):
                if drop_first and i == 0:
                    continue
                new_col_name = f"{col}_{category}"
                df_encoded[new_col_name] = (df_encoded[col] == category).astype(int)
            
            df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded


# =============================================================================
# FEATURE SCALING
# =============================================================================

class StandardScaler:
    """
    Standardization (Z-score normalization) scaler.
    Transforms features to have zero mean and unit variance.
    
    Formula: X_scaled = (X - mean) / std
    
    This is implemented from scratch without using sklearn.
    """
    
    def __init__(self):
        self.means = None
        self.stds = None
        self.feature_names = None
    
    def fit(self, X):
        """
        Compute the mean and standard deviation for each feature.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature matrix to fit
        
        Returns:
        --------
        self : StandardScaler
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Compute mean and std for each feature (column)
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        
        # Avoid division by zero: replace zero stds with 1
        self.stds[self.stds == 0] = 1.0
        
        return self
    
    def transform(self, X):
        """
        Apply standardization to features.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature matrix to transform
        
        Returns:
        --------
        X_scaled : np.ndarray
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = (X - self.means) / self.stds
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        X_scaled : np.ndarray
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the standardization transformation.
        
        Parameters:
        -----------
        X_scaled : np.ndarray
            Scaled feature matrix
        
        Returns:
        --------
        X_original : np.ndarray
        """
        return X_scaled * self.stds + self.means
    
    def save(self, filename):
        """Save scaler parameters to file."""
        params = {
            'means': self.means.tolist() if self.means is not None else None,
            'stds': self.stds.tolist() if self.stds is not None else None,
            'feature_names': self.feature_names
        }
        with open(filename, 'w') as f:
            json.dump(params, f)
    
    def load(self, filename):
        """Load scaler parameters from file."""
        with open(filename, 'r') as f:
            params = json.load(f)
        self.means = np.array(params['means']) if params['means'] else None
        self.stds = np.array(params['stds']) if params['stds'] else None
        self.feature_names = params['feature_names']
        return self


# =============================================================================
# FRAUD DATA PREPROCESSING PIPELINE
# =============================================================================

def preprocess_fraud_data(train_df, test_df):
    """
    Preprocess the fraud detection dataset.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with 'is_fraud' column
    test_df : pd.DataFrame
        Test data without 'is_fraud' column
    
    Returns:
    --------
    X_train, y_train, X_test, test_ids, scaler, encoding_info, imputation_values
    """
    # Store test IDs for submission
    test_ids = test_df['ID'].values
    train_ids = train_df['ID'].values
    
    # Separate target
    y_train = train_df['is_fraud'].values
    
    # Drop ID and target columns, also drop transaction_id and user_id (not useful as features)
    cols_to_drop = ['ID', 'transaction_id', 'user_id']
    
    train_features = train_df.drop(columns=['is_fraud'] + cols_to_drop, errors='ignore')
    test_features = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Identify column types
    numerical_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Handle missing values in training data
    train_filled, imputation_values = handle_missing_values(
        train_features,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols
    )
    
    # Apply same imputation to test data
    test_filled = apply_imputation(test_features, imputation_values)
    
    # Handle any remaining missing values in test (categories not in train)
    for col in numerical_cols:
        if col in test_filled.columns and test_filled[col].isnull().any():
            test_filled[col] = test_filled[col].fillna(imputation_values.get(col, {}).get('value', 0))
    for col in categorical_cols:
        if col in test_filled.columns and test_filled[col].isnull().any():
            mode_val = train_filled[col].mode()[0] if len(train_filled[col].mode()) > 0 else 'unknown'
            test_filled[col] = test_filled[col].fillna(mode_val)
    
    # One-hot encode categorical features
    train_encoded, encoding_info = one_hot_encode(train_filled, categorical_cols)
    test_encoded = apply_one_hot_encode(test_filled, encoding_info)
    
    # Ensure both have same columns (add missing columns with 0)
    train_cols = set(train_encoded.columns)
    test_cols = set(test_encoded.columns)
    
    # Add missing columns to test
    for col in train_cols - test_cols:
        test_encoded[col] = 0
    
    # Add missing columns to train (shouldn't happen often)
    for col in test_cols - train_cols:
        train_encoded[col] = 0
    
    # Ensure same column order
    all_cols = sorted(train_encoded.columns.tolist())
    train_encoded = train_encoded[all_cols]
    test_encoded = test_encoded[all_cols]
    
    print(f"\nFeatures after encoding: {train_encoded.shape[1]}")
    print(f"Training samples: {len(train_encoded)}")
    print(f"Test samples: {len(test_encoded)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_encoded)
    X_test = scaler.transform(test_encoded)
    
    return X_train, y_train, X_test, test_ids, scaler, encoding_info, imputation_values
