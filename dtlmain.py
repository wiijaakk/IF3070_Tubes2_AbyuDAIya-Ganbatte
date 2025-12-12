# Buat file main_dtl.py
from preprocessing import preprocess_fraud_data
from dtl import DecisionTreeCART
import pandas as pd

# Load & preprocess
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
X_train, y_train, X_test, test_ids, *_ = preprocess_fraud_data(
    train_df, test_df, use_feature_engineering=False  # Tree ga perlu FE
)

# Convert numpy arrays to DataFrame (DecisionTreeCART expects DataFrame)
X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
y_train_series = pd.Series(y_train, name='target')
X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])

print(f"Training Decision Tree with {X_train_df.shape[0]} samples, {X_train_df.shape[1]} features")
print(f"Fraud rate: {y_train_series.mean():.2%}")

# Train tree with better parameters for imbalanced data
tree = DecisionTreeCART(
    max_depth=15,      # Deeper tree to capture fraud patterns
    min_samples=50,    # Lower threshold to allow smaller fraud clusters
    verbose=True
)
tree.fit(X_train_df, y_train_series)

# Predict
predictions = tree.predict(X_test_df)

# Save submission
pd.DataFrame({'ID': test_ids, 'is_fraud': predictions}).to_csv("submission_tree.csv", index=False)
print(f"\nSubmission saved to submission_tree.csv")
print(f"Total predictions: {len(predictions)}")
print(f"Fraud predictions: {predictions.sum()} ({predictions.mean():.2%})")