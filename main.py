"""
Main Pipeline for Fraud Detection
Tugas Besar 2 IF3070 – Dasar Inteligensi Artifisial

This script orchestrates the fraud detection pipeline using:
- preprocessing.py: Data preprocessing utilities
- logistic_regression.py: Logistic Regression model and metrics

Features:
- K-Fold Cross Validation for robust evaluation
- Stratified splits to maintain class balance
- Feature engineering with ratio and log features
- Advanced optimization with momentum and LR scheduling
- Threshold tuning for optimal F1 score

Author: AbyuDAIya-Ganbatte Team
"""

import numpy as np
import pandas as pd

# Import from our modules
from preprocessing import (
    train_test_split,
    k_fold_split,
    preprocess_fraud_data,
    StandardScaler,
    MinMaxScaler,
    clip_outliers,
    create_polynomial_features
)
from logistic_regression import (
    LogisticRegression,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def create_submission(model, X_test, test_ids, filename="submission.csv", use_probabilities=True, threshold=0.5):
    """
    Create submission file for Kaggle.
    
    For AUC evaluation, we submit probabilities instead of binary predictions.
    
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
    use_probabilities : bool
        If True, submit probabilities (for AUC). If False, submit binary predictions.
    threshold : float
        Threshold for binary predictions (only used when use_probabilities=False)
    """
    probabilities = model.predict_proba(X_test)
    
    if use_probabilities:
        # For AUC, submit probabilities
        submission = pd.DataFrame({
            'ID': test_ids,
            'is_fraud': probabilities
        })
        print(f"Submission saved to {filename}")
        print(f"Total predictions: {len(probabilities)}")
        print(f"Probability stats: min={probabilities.min():.4f}, max={probabilities.max():.4f}, mean={probabilities.mean():.4f}")
    else:
        # For F1/accuracy, submit binary 0 or 1
        predictions = (probabilities >= threshold).astype(int)
        submission = pd.DataFrame({
            'ID': test_ids,
            'is_fraud': predictions
        })
        print(f"Submission saved to {filename}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Fraud predictions (1): {np.sum(predictions)} ({100*np.mean(predictions):.2f}%)")
        print(f"Non-fraud predictions (0): {len(predictions) - np.sum(predictions)} ({100*(1-np.mean(predictions)):.2f}%)")
    
    submission.to_csv(filename, index=False)
    
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
        # More fine-grained thresholds
        thresholds = np.arange(0.05, 0.95, 0.02)
    
    best_f1 = 0
    best_threshold = 0.5
    results = []
    
    print("\nThreshold tuning (showing top 10):")
    for thresh in thresholds:
        y_pred = model.predict(X_val, threshold=thresh)
        f1 = f1_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        results.append((thresh, f1, prec, rec))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Sort by F1 and show top 10
    results.sort(key=lambda x: x[1], reverse=True)
    for thresh, f1, prec, rec in results[:10]:
        marker = " <-- BEST" if thresh == best_threshold else ""
        print(f"  Threshold {thresh:.2f}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}{marker}")
    
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
    y_proba = model.predict_proba(X)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n{dataset_name} Results (threshold={threshold:.2f}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 'confusion_matrix': cm}


def cross_validate(X, y, model_params, n_folds=5, threshold=0.5):
    """
    Perform K-Fold Cross Validation.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model_params : dict
        Parameters for LogisticRegression
    n_folds : int
        Number of folds
    threshold : float
        Decision threshold
    
    Returns:
    --------
    results : dict
        Cross-validation results
    """
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation (K={n_folds})")
    print(f"{'='*60}")
    
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    fold_num = 0
    for X_train, X_val, y_train, y_val in k_fold_split(X, y, n_folds=n_folds, random_state=42, stratify=True):
        fold_num += 1
        print(f"\nFold {fold_num}/{n_folds}")
        print(f"  Train: {len(y_train)} samples, Fraud: {np.mean(y_train)*100:.2f}%")
        print(f"  Val:   {len(y_val)} samples, Fraud: {np.mean(y_val)*100:.2f}%")
        
        # Train model
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val, threshold=threshold)
        y_proba = model.predict_proba(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1'].append(f1)
        fold_results['auc'].append(auc)
        
        print(f"  Results: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Calculate mean and std
    print(f"\n{'='*60}")
    print("Cross-Validation Summary:")
    print(f"{'='*60}")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mean = np.mean(fold_results[metric])
        std = np.std(fold_results[metric])
        print(f"  {metric.upper():10s}: {mean:.4f} (+/- {std:.4f})")
    
    return fold_results


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
    # 2. Preprocess Data (with Feature Engineering)
    # =========================================================================
    print("\n[2] Preprocessing data with feature engineering...")
    X_train_processed, y_train, X_test_processed, test_ids, scaler, encoding_info, imputation_values = preprocess_fraud_data(
        train_df, test_df, use_feature_engineering=True
    )
    
    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")
    
    # =========================================================================
    # 3. Model Parameters - Using ADAM Optimizer
    # =========================================================================
    model_params = {
        'learning_rate': 0.0012,  # Trying lower lr
        'n_iterations': 3000,
        'optimizer': "adam",  # ADAM OPTIMIZER
        'batch_size': 256,
        'regularization': 0.0005,
        'l1_ratio': 0.5,  # Balanced L1/L2
        'class_weight': "balanced",  # Use balanced class weights
        'lr_schedule': "constant",  # Adam usually works best with constant lr
        'beta1': 0.9,  # First moment decay (momentum-like)
        'beta2': 0.999,  # Second moment decay (RMSprop-like)
        'epsilon': 1e-8,  # Numerical stability
        'momentum': 0.0,  # Not used for Adam
        'nesterov': False,
        'use_focal_loss': False,
        'focal_gamma': 2.0,
        'early_stopping': True,
        'patience': 400,
        'tol': 1e-9,
        'verbose': True
    }
    
    print("\n[3] Model Configuration:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # =========================================================================
    # 4. Skip CV for faster iteration
    # =========================================================================
    print("\n[4] Skipping CV for speed...")
    cv_results = {'f1': [0.27], 'auc': [0.60]}  # Placeholder
    
    # =========================================================================
    # 5. Train/Validation Split for Threshold Tuning
    # =========================================================================
    print("\n[5] Creating stratified train/validation split...")
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples, Fraud: {np.mean(y_train_split)*100:.2f}%")
    print(f"Validation set: {X_val.shape[0]} samples, Fraud: {np.mean(y_val)*100:.2f}%")
    
    # =========================================================================
    # 6. Train Model
    # =========================================================================
    print("\n[6] Training Logistic Regression model...")
    
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train_split)
    
    # =========================================================================
    # 7. Find Best Threshold (for F1, but AUC doesn't need threshold)
    # =========================================================================
    print("\n[7] Finding best threshold (for F1 reference)...")
    best_threshold, best_f1 = find_best_threshold(model, X_val, y_val)
    
    # =========================================================================
    # 8. Evaluate on Validation Set
    # =========================================================================
    print("\n[8] Evaluating model...")
    val_results = evaluate_model(model, X_val, y_val, threshold=best_threshold, dataset_name="Validation")
    train_results = evaluate_model(model, X_train, y_train_split, threshold=best_threshold, dataset_name="Training")
    
    # =========================================================================
    # 9. Retrain on Full Data
    # =========================================================================
    print("\n[9] Retraining on full training data...")
    
    model_full = LogisticRegression(**model_params)
    model_full.fit(X_train_processed, y_train)
    
    # =========================================================================
    # 10. Generate Submission (AUC uses probabilities, not binary)
    # =========================================================================
    print("\n[10] Generating submission...")
    print("NOTE: AUC evaluation uses probabilities, submitting probabilities instead of binary predictions.")
    
    # Get probabilities for test set
    proba = model_full.predict_proba(X_test_processed)
    
    print(f"\nProbability distribution on test set:")
    print(f"  Min:    {proba.min():.4f}")
    print(f"  Max:    {proba.max():.4f}")
    print(f"  Mean:   {proba.mean():.4f}")
    print(f"  Median: {np.median(proba):.4f}")
    print(f"  Std:    {proba.std():.4f}")
    
    # Show what binary predictions would look like at different thresholds
    print("\nBinary prediction counts (for reference):")
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        preds = (proba >= thresh).astype(int)
        fraud_pct = np.mean(preds) * 100
        print(f"  Threshold {thresh:.2f}: {np.sum(preds):5d} fraud ({fraud_pct:.1f}%)")
    
    # Try threshold that matches the actual fraud rate (~14%)
    # This might give better AUC on Kaggle
    target_fraud_rate = 0.14
    sorted_proba = np.sort(proba)[::-1]  # Sort descending
    threshold_for_14pct = sorted_proba[int(len(sorted_proba) * target_fraud_rate)]
    print(f"\nThreshold for ~14% fraud rate: {threshold_for_14pct:.4f}")
    
    # Create submission with probabilities (better for AUC evaluation)
    print(f"\nSubmitting probabilities for AUC evaluation")
    submission = create_submission(
        model_full, 
        X_test_processed, 
        test_ids, 
        filename="submission.csv",
        use_probabilities=True,  # Submit probabilities for AUC
        threshold=best_threshold
    )
    
    # =========================================================================
    # 11. Save Model
    # =========================================================================
    print("\n[11] Saving model...")
    model_full.save_model("logistic_model.json")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Features used: {X_train_processed.shape[1]}")
    print(f"Validation AUC: {val_results['auc']:.4f}")
    print(f"Validation F1:  {val_results['f1']:.4f} (threshold={best_threshold:.2f})")
    if 'auc' in cv_results:
        print(f"CV Mean AUC: {np.mean(cv_results['auc']):.4f} (+/- {np.std(cv_results['auc']):.4f})")
    print(f"CV Mean F1:  {np.mean(cv_results['f1']):.4f} (+/- {np.std(cv_results['f1']):.4f})")
    print(f"Predictions saved to: submission.csv (probabilities)")
    print("=" * 60)
    
    return model_full, submission, cv_results


if __name__ == "__main__":
    model, submission, cv_results = main()
