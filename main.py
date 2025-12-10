"""
Main Pipeline for Fraud Detection
Tugas Besar 2 IF3070 – Dasar Inteligensi Artifisial

This script orchestrates the fraud detection pipeline using:
- preprocessing.py: Data preprocessing utilities
- logistic_regression.py: Logistic Regression model and metrics

Author: AbyuDAIya-Ganbatte Team
"""

import numpy as np
import pandas as pd

# Import from our modules
from preprocessing import (
    train_test_split,
    preprocess_fraud_data,
    StandardScaler
)
from logistic_regression import (
    LogisticRegression,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def create_submission(model, X_test, test_ids, filename="submission.csv", threshold=0.5):
    """
    Create submission file for Kaggle.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X_test : np.ndarray
        Test features
    test_ids : np.ndarray
        Test IDs
    filename : str
        Output filename
    threshold : float
        Decision threshold
    """
    predictions = model.predict(X_test, threshold=threshold)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'is_fraud': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Fraud predictions: {np.sum(predictions)} ({100*np.mean(predictions):.2f}%)")
    
    return submission


def find_best_threshold(model, X_val, y_val, thresholds=None):
    """
    Find the best threshold for classification based on F1 score.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    thresholds : list or None
        List of thresholds to try
    
    Returns:
    --------
    best_threshold : float
    best_f1 : float
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_f1 = 0
    best_threshold = 0.5
    
    print("\nThreshold tuning:")
    for thresh in thresholds:
        y_pred = model.predict(X_val, threshold=thresh)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
        print(f"  Threshold {thresh:.2f}: F1 = {f1:.4f}")
    
    print(f"\nBest threshold: {best_threshold:.2f} with F1 = {best_f1:.4f}")
    return best_threshold, best_f1


def evaluate_model(model, X, y, threshold=0.5, dataset_name="Dataset"):
    """
    Evaluate model performance on a dataset.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X : np.ndarray
        Features
    y : np.ndarray
        True labels
    threshold : float
        Decision threshold
    dataset_name : str
        Name for display
    """
    y_pred = model.predict(X, threshold=threshold)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n{dataset_name} Results (threshold={threshold:.2f}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}


def main():
    """
    Main function to run the fraud detection pipeline.
    """
    print("=" * 60)
    print("Fraud Detection with Logistic Regression")
    print("Tugas Besar 2 IF3070 – Dasar Inteligensi Artifisial")
    print("=" * 60)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1] Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Check fraud rate
    fraud_rate = train_df['is_fraud'].mean()
    print(f"Fraud rate in training data: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
    
    # =========================================================================
    # 2. Preprocess Data
    # =========================================================================
    print("\n[2] Preprocessing data...")
    X_train_processed, y_train, X_test_processed, test_ids, scaler, encoding_info, imputation_values = preprocess_fraud_data(
        train_df, test_df
    )
    
    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")
    
    # =========================================================================
    # 3. Train/Validation Split
    # =========================================================================
    print("\n[3] Creating train/validation split...")
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # =========================================================================
    # 4. Train Model
    # =========================================================================
    print("\n[4] Training Logistic Regression model...")
    
    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=300,
        optimizer="mini-batch",
        batch_size=1024,
        regularization=0.001,
        class_weight="balanced",
        early_stopping=False,
        verbose=True
    )
    
    model.fit(X_train, y_train_split)
    
    # =========================================================================
    # 5. Evaluate on Validation Set
    # =========================================================================
    print("\n[5] Evaluating model...")
    
    # Find best threshold
    best_threshold, best_f1 = find_best_threshold(model, X_val, y_val)
    
    # Evaluate with best threshold
    val_results = evaluate_model(model, X_val, y_val, threshold=best_threshold, dataset_name="Validation")
    
    # Also evaluate training set
    train_results = evaluate_model(model, X_train, y_train_split, threshold=best_threshold, dataset_name="Training")
    
    # =========================================================================
    # 6. Retrain on Full Data (Optional)
    # =========================================================================
    print("\n[6] Retraining on full training data...")
    
    model_full = LogisticRegression(
        learning_rate=0.1,
        n_iterations=300,
        optimizer="mini-batch",
        batch_size=1024,
        regularization=0.001,
        class_weight="balanced",
        early_stopping=False,
        verbose=True
    )
    
    model_full.fit(X_train_processed, y_train)
    
    # =========================================================================
    # 7. Generate Submission
    # =========================================================================
    print("\n[7] Generating submission...")
    
    submission = create_submission(
        model_full, 
        X_test_processed, 
        test_ids, 
        filename="submission.csv",
        threshold=best_threshold
    )
    
    # =========================================================================
    # 8. Save Model
    # =========================================================================
    print("\n[8] Saving model...")
    model_full.save_model("logistic_model.json")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return model_full, submission


if __name__ == "__main__":
    model, submission = main()
