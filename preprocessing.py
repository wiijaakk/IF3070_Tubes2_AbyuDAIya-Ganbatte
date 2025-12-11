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

def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """
    Manually split data into training and testing sets with optional stratification.
    
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
    stratify : np.ndarray or None
        If provided, split will maintain class proportions
    
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
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    if stratify is not None:
        # Stratified split - maintain class proportions
        if isinstance(stratify, (pd.Series, pd.DataFrame)):
            stratify = stratify.values.flatten()
        
        train_indices = []
        test_indices = []
        
        # Get unique classes
        classes = np.unique(stratify)
        
        for cls in classes:
            # Get indices for this class
            cls_indices = np.where(stratify == cls)[0]
            np.random.shuffle(cls_indices)
            
            # Calculate split point for this class
            n_test_cls = int(len(cls_indices) * test_size)
            
            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Shuffle the indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    else:
        # Regular random split
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def k_fold_split(X, y, n_folds=5, random_state=None, stratify=True):
    """
    Generate K-Fold cross-validation splits.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_folds : int
        Number of folds
    random_state : int or None
        Seed for reproducibility
    stratify : bool
        Whether to maintain class proportions in each fold
    
    Yields:
    -------
    X_train, X_val, y_train, y_val : tuple for each fold
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.flatten()
    
    n_samples = len(X)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if stratify:
        # Stratified K-Fold
        classes = np.unique(y)
        fold_indices = [[] for _ in range(n_folds)]
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            # Distribute class samples across folds
            for i, idx in enumerate(cls_indices):
                fold_indices[i % n_folds].append(idx)
        
        # Convert to arrays
        fold_indices = [np.array(fold) for fold in fold_indices]
    else:
        # Regular K-Fold
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds
        fold_indices = []
        
        for i in range(n_folds):
            start = i * fold_size
            if i == n_folds - 1:
                fold_indices.append(indices[start:])
            else:
                fold_indices.append(indices[start:start + fold_size])
    
    # Generate train/val splits for each fold
    for i in range(n_folds):
        val_indices = fold_indices[i]
        train_indices = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
        
        yield X[train_indices], X[val_indices], y[train_indices], y[val_indices]


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


class MinMaxScaler:
    """
    Min-Max normalization scaler.
    Transforms features to a specified range (default 0 to 1).
    
    Formula: X_scaled = (X - min) / (max - min)
    
    This is implemented from scratch without using sklearn.
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.mins = None
        self.maxs = None
        self.feature_names = None
    
    def fit(self, X):
        """Compute min and max for each feature."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.mins = np.min(X, axis=0)
        self.maxs = np.max(X, axis=0)
        
        # Avoid division by zero
        diff = self.maxs - self.mins
        diff[diff == 0] = 1.0
        self.maxs = self.mins + diff
        
        return self
    
    def transform(self, X):
        """Apply min-max normalization."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_std = (X - self.mins) / (self.maxs - self.mins)
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """Reverse the normalization."""
        X_std = (X_scaled - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return X_std * (self.maxs - self.mins) + self.mins


# =============================================================================
# OUTLIER HANDLING
# =============================================================================

def clip_outliers(X, lower_percentile=1, upper_percentile=99):
    """
    Clip outliers using percentile-based winsorization.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    lower_percentile : float
        Lower percentile for clipping (default: 1)
    upper_percentile : float
        Upper percentile for clipping (default: 99)
    
    Returns:
    --------
    X_clipped : np.ndarray
    clip_bounds : dict
        Dictionary with lower and upper bounds for each feature
    """
    if isinstance(X, pd.DataFrame):
        X = X.values.copy()
    else:
        X = X.copy()
    
    n_features = X.shape[1]
    clip_bounds = {'lower': [], 'upper': []}
    
    for i in range(n_features):
        lower = np.percentile(X[:, i], lower_percentile)
        upper = np.percentile(X[:, i], upper_percentile)
        
        clip_bounds['lower'].append(lower)
        clip_bounds['upper'].append(upper)
        
        X[:, i] = np.clip(X[:, i], lower, upper)
    
    return X, clip_bounds


def apply_clip_outliers(X, clip_bounds):
    """Apply previously computed clip bounds to new data."""
    if isinstance(X, pd.DataFrame):
        X = X.values.copy()
    else:
        X = X.copy()
    
    for i in range(X.shape[1]):
        X[:, i] = np.clip(X[:, i], clip_bounds['lower'][i], clip_bounds['upper'][i])
    
    return X


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_polynomial_features(X, degree=2, interaction_only=False, include_bias=False):
    """
    Generate polynomial and interaction features.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    degree : int
        Maximum degree of polynomial features (default: 2)
    interaction_only : bool
        If True, only interaction features are produced (no x^2, x^3, etc.)
    include_bias : bool
        If True, include a bias column (all ones)
    
    Returns:
    --------
    X_poly : np.ndarray
        Feature matrix with polynomial features
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    n_samples, n_features = X.shape
    
    # Start with original features
    features = [X]
    
    if include_bias:
        features.insert(0, np.ones((n_samples, 1)))
    
    if degree >= 2:
        # Add interaction terms (x_i * x_j for i < j)
        for i in range(n_features):
            for j in range(i, n_features):
                if interaction_only and i == j:
                    continue
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    
    if degree >= 3 and not interaction_only:
        # Add cubic terms for individual features
        for i in range(n_features):
            features.append((X[:, i] ** 3).reshape(-1, 1))
    
    return np.hstack(features)


def create_ratio_features(df, numerical_cols):
    """
    Create ratio features from numerical columns.
    Select meaningful ratios based on domain knowledge.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numerical columns
    numerical_cols : list
        List of numerical column names
    
    Returns:
    --------
    df_with_ratios : pd.DataFrame
    """
    df = df.copy()
    
    # Define meaningful ratio pairs for fraud detection
    ratio_pairs = [
        ('transaction_amount', 'avg_transaction_amount', 'amount_vs_avg_ratio'),
        ('transaction_amount', 'std_transaction_amount', 'amount_vs_std_ratio'),
        ('transactions_last_1h', 'transactions_last_24h', 'hourly_vs_daily_ratio'),
        ('failed_login_attempts', 'num_prev_transactions', 'failed_vs_total_ratio'),
        ('shared_ip_users', 'shared_device_users', 'ip_vs_device_shared_ratio'),
    ]
    
    for num, denom, name in ratio_pairs:
        if num in df.columns and denom in df.columns:
            # Avoid division by zero
            denom_safe = df[denom].replace(0, 1e-10)
            df[name] = df[num] / denom_safe
    
    return df


def create_log_features(df, numerical_cols, epsilon=1e-10):
    """
    Create log-transformed features for skewed distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numerical columns
    numerical_cols : list
        Columns to log-transform
    epsilon : float
        Small value to avoid log(0)
    
    Returns:
    --------
    df_with_logs : pd.DataFrame
    """
    df = df.copy()
    
    # Columns that typically benefit from log transform
    log_candidates = [
        'transaction_amount', 'avg_transaction_amount', 'std_transaction_amount',
        'account_age_days', 'distance_from_home', 'num_prev_transactions'
    ]
    
    for col in log_candidates:
        if col in df.columns:
            # Only log transform non-negative columns
            min_val = df[col].min()
            if min_val >= 0:
                df[f'{col}_log'] = np.log1p(df[col])  # log(1 + x) for stability
    
    return df


def create_interaction_features(df):
    """
    Create meaningful interaction features for fraud detection.
    
    These are domain-specific interaction terms that can capture
    complex patterns in fraud behavior.
    """
    df = df.copy()
    
    # Risk Score Interactions
    if 'ip_risk_score' in df.columns and 'device_trust_score' in df.columns:
        # Combined risk indicator
        df['risk_interaction'] = df['ip_risk_score'] * (1 - df['device_trust_score'] / 100)
    
    if 'merchant_risk' in df.columns and 'country_risk' in df.columns:
        # Combined merchant-country risk
        df['merchant_country_risk'] = df['merchant_risk'] * df['country_risk']
    
    # Transaction amount anomaly indicators
    if 'transaction_amount' in df.columns and 'avg_transaction_amount' in df.columns:
        # Z-score like feature
        std_col = 'std_transaction_amount' if 'std_transaction_amount' in df.columns else None
        if std_col and df[std_col].mean() > 0:
            df['amount_zscore'] = (df['transaction_amount'] - df['avg_transaction_amount']) / (df[std_col] + 1e-6)
    
    # Velocity-based features
    if 'transactions_last_24h' in df.columns and 'transactions_last_1h' in df.columns:
        # Recent activity concentration
        df['hourly_concentration'] = df['transactions_last_1h'] / (df['transactions_last_24h'] + 1)
    
    # Account trust features
    if 'account_age_days' in df.columns and 'num_prev_transactions' in df.columns:
        # Transaction frequency per day of account age
        df['tx_per_day_age'] = df['num_prev_transactions'] / (df['account_age_days'] + 1)
    
    # New user risk
    if 'is_new_country' in df.columns and 'distance_from_home' in df.columns:
        df['new_location_distance'] = df['is_new_country'] * df['distance_from_home']
    
    # Failed login impact
    if 'failed_login_attempts' in df.columns and 'transaction_amount' in df.columns:
        df['failed_login_amount'] = df['failed_login_attempts'] * df['transaction_amount']
    
    # Shared resource risk
    if 'shared_ip_users' in df.columns and 'shared_device_users' in df.columns:
        df['total_shared_users'] = df['shared_ip_users'] + df['shared_device_users']
        df['shared_resource_product'] = df['shared_ip_users'] * df['shared_device_users']
    
    # Time-based risk
    if 'time_of_day' in df.columns:
        # Night time risk (0-6 AM and 22-24)
        df['is_night_time'] = ((df['time_of_day'] >= 0) & (df['time_of_day'] <= 6) | 
                               (df['time_of_day'] >= 22)).astype(float)
    
    if 'day_of_week' in df.columns:
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
    
    # Chargeback history interaction
    if 'has_chargeback_history' in df.columns and 'transaction_amount' in df.columns:
        df['chargeback_high_amount'] = df['has_chargeback_history'] * (df['transaction_amount'] > df['transaction_amount'].median()).astype(float)
    
    # Additional high-value interaction features
    if 'ip_risk_score' in df.columns and 'transaction_amount' in df.columns:
        df['high_risk_high_amount'] = df['ip_risk_score'] * df['transaction_amount']
    
    if 'failed_login_attempts' in df.columns and 'ip_risk_score' in df.columns:
        df['failed_login_risk'] = df['failed_login_attempts'] * df['ip_risk_score']
    
    if 'is_new_country' in df.columns and 'transaction_amount' in df.columns:
        df['new_country_amount'] = df['is_new_country'] * df['transaction_amount']
    
    # Squared features for important risk indicators
    if 'ip_risk_score' in df.columns:
        df['ip_risk_squared'] = df['ip_risk_score'] ** 2
    
    if 'merchant_risk' in df.columns:
        df['merchant_risk_squared'] = df['merchant_risk'] ** 2
    
    if 'country_risk' in df.columns:
        df['country_risk_squared'] = df['country_risk'] ** 2
    
    # Combined risk score
    risk_cols = []
    if 'ip_risk_score' in df.columns:
        risk_cols.append('ip_risk_score')
    if 'merchant_risk' in df.columns:
        risk_cols.append('merchant_risk')
    if 'country_risk' in df.columns:
        risk_cols.append('country_risk')
    if len(risk_cols) > 0:
        df['combined_risk'] = df[risk_cols].mean(axis=1)
    
    return df


def create_binned_features(df, col, n_bins=5, strategy='quantile'):
    """
    Create binned (discretized) features from continuous variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    col : str
        Column to bin
    n_bins : int
        Number of bins
    strategy : str
        'quantile' for equal-frequency bins, 'uniform' for equal-width bins
    
    Returns:
    --------
    df_with_bins : pd.DataFrame
    bin_edges : np.ndarray
    """
    df = df.copy()
    
    if strategy == 'quantile':
        bin_edges = np.percentile(df[col].dropna(), np.linspace(0, 100, n_bins + 1))
    else:
        bin_edges = np.linspace(df[col].min(), df[col].max(), n_bins + 1)
    
    # Make edges unique
    bin_edges = np.unique(bin_edges)
    
    # Assign bins
    df[f'{col}_bin'] = np.digitize(df[col], bin_edges[1:-1])
    
    return df, bin_edges


# =============================================================================
# FRAUD DATA PREPROCESSING PIPELINE
# =============================================================================

def preprocess_fraud_data(train_df, test_df, use_feature_engineering=True):
    """
    Preprocess the fraud detection dataset with advanced feature engineering.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with 'is_fraud' column
    test_df : pd.DataFrame
        Test data without 'is_fraud' column
    use_feature_engineering : bool
        Whether to apply advanced feature engineering (default: True)
    
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
    
    # =========================================================================
    # FEATURE ENGINEERING (NEW)
    # =========================================================================
    if use_feature_engineering:
        print("\nApplying feature engineering...")
        
        # Create ratio features
        train_filled = create_ratio_features(train_filled, numerical_cols)
        test_filled = create_ratio_features(test_filled, numerical_cols)
        
        # Create log features for skewed distributions
        train_filled = create_log_features(train_filled, numerical_cols)
        test_filled = create_log_features(test_filled, numerical_cols)
        
        # Create interaction features (domain-specific combinations)
        train_filled = create_interaction_features(train_filled)
        test_filled = create_interaction_features(test_filled)
        
        # Update numerical columns list
        numerical_cols = train_filled.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Features after engineering: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
    
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
    
    # Clip outliers before scaling
    train_values = train_encoded.values
    test_values = test_encoded.values
    
    train_clipped, clip_bounds = clip_outliers(train_values, lower_percentile=1, upper_percentile=99)
    test_clipped = apply_clip_outliers(test_values, clip_bounds)
    
    print(f"\nFeatures after encoding: {train_clipped.shape[1]}")
    print(f"Training samples: {len(train_clipped)}")
    print(f"Test samples: {len(test_clipped)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clipped)
    X_test = scaler.transform(test_clipped)
    
    return X_train, y_train, X_test, test_ids, scaler, encoding_info, imputation_values
