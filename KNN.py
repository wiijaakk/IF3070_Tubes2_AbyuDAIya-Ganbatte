import numpy as np
import pandas as pd
from collections import Counter


class KNN:
    def __init__(self, n_neighbors=5, weighted=False, metric='euclidean', p=2):
        self.n_neighbors = n_neighbors
        self.weighted = weighted
        self.metric = metric  # 'euclidean', 'manhattan', 'minkowski'
        self.p = p  # power parameter for minkowski
        self.X_train = None
        self.y_train = None
        
    def _calculate_distances(self, x):
        """Vectorized distance calculation for one test point to all training points"""
        diff = self.X_train - x
        if self.metric == 'manhattan':
            return np.sum(np.abs(diff), axis=1)
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(diff) ** self.p, axis=1), 1/self.p)
        else:  # euclidean (default)
            return np.sqrt(np.sum(diff ** 2, axis=1))

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X, show_progress=False):
        X = np.array(X)
        predictions = []
        n = len(X)
        for i, x in enumerate(X):
            if show_progress and (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{n}")
            predictions.append(self._predict(x))
        return np.array(predictions)

    def _predict(self, x):
        distances = self._calculate_distances(x)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_labels = self.y_train[k_indices]
        
        if self.weighted:
            weights = 1 / (distances[k_indices] + 1e-8)
            class_weights = {}
            for label, weight in zip(k_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            return max(class_weights, key=class_weights.get)
        else:
            return Counter(k_labels).most_common(1)[0][0]
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def load_and_preprocess_data(train_path='train.csv', test_path='test.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = 'is_fraud'
    drop_cols = ['ID', 'transaction_id', 'user_id', target_col]
    
    X = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
    y = train_df[target_col].values
    X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])
    test_ids = test_df['ID'].values
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values
    for col in numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # One-hot encode
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    X = X.values.astype(np.float64)
    X_test = X_test.values.astype(np.float64)
    
    # Min-Max normalization
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X = (X - X_min) / X_range
    X_test = (X_test - X_min) / X_range
    
    # Train-val split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]
    
    return X_train, X_val, y_train, y_val, X_test, test_ids


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}}


def find_best_k(X_train, y_train, X_val, y_val, k_values=[3, 5, 7, 9, 11], sample_size=5000):
    # Sample for speed
    if len(X_train) > sample_size:
        np.random.seed(42)
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_s, y_train_s = X_train[idx], y_train[idx]
    else:
        X_train_s, y_train_s = X_train, y_train
    
    val_size = min(1000, len(X_val))
    val_idx = np.random.choice(len(X_val), val_size, replace=False)
    X_val_s, y_val_s = X_val[val_idx], y_val[val_idx]
    
    best_k, best_f1 = k_values[0], 0
    for k in k_values:
        knn = KNN(n_neighbors=k, weighted=True)
        knn.fit(X_train_s, y_train_s)
        metrics = calculate_metrics(y_val_s, knn.predict(X_val_s))
        if metrics['f1_score'] > best_f1:
            best_f1, best_k = metrics['f1_score'], k
    
    return best_k


def create_submission(test_ids, predictions, filename='knn_submission.csv'):
    submission = pd.DataFrame({'ID': test_ids, 'is_fraud': predictions})
    submission.to_csv(filename, index=False)


if __name__ == "__main__":
    # Load data
    X_train, X_val, y_train, y_val, X_test, test_ids = load_and_preprocess_data()
    
    # Find best k
    best_k = find_best_k(X_train, y_train, X_val, y_val)
    print(f"Best k: {best_k}")
    
    # Train with sampled data (KNN is slow on large datasets)
    train_sample_size = 3000
    np.random.seed(42)
    if len(X_train) > train_sample_size:
        idx = np.random.choice(len(X_train), train_sample_size, replace=False)
        X_train_final, y_train_final = X_train[idx], y_train[idx]
    else:
        X_train_final, y_train_final = X_train, y_train
    
    knn = KNN(n_neighbors=best_k, weighted=True)
    knn.fit(X_train_final, y_train_final)
    
    # Evaluate on sampled validation (for speed)
    val_sample_size = 1000
    val_idx = np.random.choice(len(X_val), min(val_sample_size, len(X_val)), replace=False)
    val_metrics = calculate_metrics(y_val[val_idx], knn.predict(X_val[val_idx]))
    print(f"Validation - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    
    # Create submission
    print("Predicting test set...")
    y_test_pred = knn.predict(X_test, show_progress=True)
    create_submission(test_ids, y_test_pred)
    print("Done! Submission saved to knn_submission.csv")