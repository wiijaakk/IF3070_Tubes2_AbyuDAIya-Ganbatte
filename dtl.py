import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

class DecisionTreeCART:

    def __init__(self, max_depth=100, min_samples=2, ccp_alpha=0.0, verbose=True):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.ccp_alpha = ccp_alpha
        self.tree = None
        self._y_dtype = None
        self._num_all_samples = None
        self.verbose = verbose
        self._node_count = 0
        self._max_depth_reached = 0

    def _set_df_type(self, X, y, dtype):
        X = X.astype(dtype)
        self._y_dtype = y.dtype
        return X, y

    @staticmethod
    def _purity(y):
        unique_classes = np.unique(y)

        return unique_classes.size == 1

    @staticmethod
    def _is_leaf_node(node):
        return not isinstance(node, dict) 

    def _leaf_node(self, y):
        class_index = 0
        return y.mode()[class_index]

    def _split_df(self, X, y, feature, threshold):
        feature_values = X[feature]
        left_indexes = X[feature_values <= threshold].index
        right_indexes = X[feature_values > threshold].index
        sizes = np.array([left_indexes.size, right_indexes.size])

        return self._leaf_node(y) if any(sizes == 0) else (left_indexes, right_indexes)

    @staticmethod
    def _gini_impurity(y):
        _, counts_classes = np.unique(y, return_counts=True)
        squared_probabilities = np.square(counts_classes / y.size)
        gini_impurity = 1 - sum(squared_probabilities)

        return gini_impurity

    def _cost_function(self, left_df, right_df):
        total_df_size = left_df.size + right_df.size
        p_left_df = left_df.size / total_df_size
        p_right_df = right_df.size / total_df_size
        J_left = self._gini_impurity(left_df)
        J_right = self._gini_impurity(right_df)
        J = p_left_df*J_left + p_right_df*J_right

        return J

    def _best_split(self, X, y):
        features = X.columns
        min_cost_function = np.inf
        best_feature, best_threshold = None, None

        for feature in features:
            unique_feature_values = np.unique(X[feature])
            
            # OPTIMASI: Limit jumlah threshold yang dicoba jika terlalu banyak
            if len(unique_feature_values) > 100:
                # Sample 100 values saja
                indices = np.linspace(0, len(unique_feature_values)-1, 100, dtype=int)
                unique_feature_values = unique_feature_values[indices]

            for i in range(1, len(unique_feature_values)):
                current_value = unique_feature_values[i]
                previous_value = unique_feature_values[i-1]
                threshold = (current_value + previous_value) / 2
                left_indexes, right_indexes = self._split_df(X, y, feature, threshold)
                left_labels, right_labels = y.loc[left_indexes], y.loc[right_indexes]
                current_J = self._cost_function(left_labels, right_labels)

                if current_J <= min_cost_function:
                    min_cost_function = current_J
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _stopping_conditions(self, X, y, depth):
        if depth >= self.max_depth:
            return True
        if len(y) < self.min_samples:
            return True
        if self._purity(y):
            return True
        return False

    def _grow_tree(self, X, y, depth=0):
        X, y = self._set_df_type(X, y, np.float64)

        if depth > self._max_depth_reached:
            self._max_depth_reached = depth

        if self._stopping_conditions(X, y, depth):
            return str(self._leaf_node(y))

        self._node_count += 1
        if self.verbose and self._node_count % 50 == 0:
            print(f"   Progress: {self._node_count} nodes | depth: {depth}/{self.max_depth} | samples: {len(y)} | max depth reached: {self._max_depth_reached}")

        best_feature, best_threshold = self._best_split(X, y)
        decision_node = f'{best_feature} <= {best_threshold} | '

        left_indexes, right_indexes = self._split_df(X, y, best_feature, best_threshold)
        left_X, right_X = X.loc[left_indexes], X.loc[right_indexes]
        left_labels, right_labels = y.loc[left_indexes], y.loc[right_indexes]

        tree = {decision_node: []}
        left_subtree = self._grow_tree(left_X, left_labels, depth+1)
        right_subtree = self._grow_tree(right_X, right_labels, depth+1)

        if left_subtree == right_subtree:
            tree = left_subtree
        else:
            tree[decision_node].extend([left_subtree, right_subtree])

        return tree

    def fit(self, X, y):
        self._num_all_samples = len(y)
        self._node_count = 0
        self._max_depth_reached = 0
        if self.verbose:
            print(f"   Starting tree construction with {len(y)} samples...")
            print(f"   Parameters: max_depth={self.max_depth}, min_samples={self.min_samples}")
        self.tree = self._grow_tree(X, y)
        if self.verbose:
            print(f"   Tree complete! Nodes: {self._node_count} | Max depth reached: {self._max_depth_reached}")
        return self

    def _traverse_tree(self, sample, tree):
        if self._is_leaf_node(tree):
            leaf, *_ = tree.split()
            return leaf

        decision_node = next(iter(tree))
        left_node, right_node = tree[decision_node]
        feature, other = decision_node.split(' <=')
        threshold, *_ = other.split()
        feature_value = sample[feature]

        if np.float64(feature_value) <= np.float64(threshold):
            next_node = self._traverse_tree(sample, left_node)
        else:
            next_node = self._traverse_tree(sample, right_node)

        return next_node

    def predict(self, samples: pd.DataFrame):
        if self.verbose:
            print(f"   Predicting {len(samples)} samples...")
        results = samples.apply(self._traverse_tree, args=(self.tree,), axis=1)
        if self.verbose:
            print(f"   Prediction complete!")
        return np.array(results.astype(self._y_dtype))

# ============= PREPROCESSING =============
print("Loading data...")
df = pd.read_csv('train.csv')

print("\nOriginal data shape:", df.shape)
print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 1. Drop kolom yang tidak perlu
print("\n1. Dropping unnecessary columns...")
X = df.drop(['is_fraud', 'ID', 'transaction_id', 'user_id'], axis=1)
y = df['is_fraud']

# 2. Handle missing values (isi dengan median)
print("2. Handling missing values...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if X[col].dtype in ['float64', 'int64']:
            X[col].fillna(X[col].median(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)

# 3. Encode kolom kategorikal
print("3. Encoding categorical columns...")
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"   Categorical columns: {list(categorical_cols)}")

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("\nAfter preprocessing:")
print("X shape:", X.shape)
print("X dtypes:")
print(X.dtypes)
print("\nAll numeric:", X.select_dtypes(include=['object']).shape[1] == 0)

# 4. Split training dan testing set
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# 5. Inisialisasi model
print("\n5. Training Decision Tree CART...")
# OPTIMASI: Kurangi max_depth dan naikkan min_samples untuk training lebih cepat
model = DecisionTreeCART(max_depth=10, min_samples=100, ccp_alpha=0.0)

# 6. Training model
model.fit(X_train, y_train)
print("Training complete!")

# 7. Prediksi
print("\n7. Making predictions...")
predictions = model.predict(X_test)

# 8. Evaluasi
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
accuracy = accuracy_score(y_test, predictions)
print(f'\nAccuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, predictions))

# 9. Save predictions to file
print("\n9. Saving predictions to file...")
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': predictions
})
predictions_df.to_csv('predictions.csv', index=False)
print("   Predictions saved to 'predictions.csv'")

# Untuk submission (kalau mau predict test.csv untuk kaggle)
print("\n10. Creating submission file (optional)...")
try:
    test_df = pd.read_csv('test.csv')
    print(f"    Test data loaded: {test_df.shape}")
    
    # Preprocessing test data (sama seperti training)
    X_test_submission = test_df.drop(['ID'], axis=1, errors='ignore')
    
    # Handle missing values
    for col in X_test_submission.columns:
        if X_test_submission[col].isnull().sum() > 0:
            if X_test_submission[col].dtype in ['float64', 'int64']:
                X_test_submission[col].fillna(X_test_submission[col].median(), inplace=True)
            else:
                X_test_submission[col].fillna(X_test_submission[col].mode()[0], inplace=True)
    
    # Encode categorical
    categorical_cols = X_test_submission.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_test_submission[col] = le.fit_transform(X_test_submission[col].astype(str))
    
    # Predict
    test_predictions = model.predict(X_test_submission)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'is_fraud': test_predictions
    })
    submission.to_csv('submissions.csv', index=False)
    print("    Submission file saved to 'submissions.csv'")
except FileNotFoundError:
    print("    No test.csv found, skipping submission file creation.")