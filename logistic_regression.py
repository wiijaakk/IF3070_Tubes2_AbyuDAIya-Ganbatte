"""
Logistic Regression Implementation from Scratch
Tugas Besar 2 IF3070 – Dasar Inteligensi Artifisial

This module implements Logistic Regression with multiple gradient descent optimizers
using only numpy and pandas.

Author: AbyuDAIya-Ganbatte Team
"""

import numpy as np
import pandas as pd
import json


# =============================================================================
# EVALUATION METRICS (From Scratch)
# =============================================================================

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    
    Returns:
    --------
    accuracy : float
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    """
    Calculate precision score.
    Precision = TP / (TP + FP)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def recall_score(y_true, y_pred):
    """
    Calculate recall score.
    Recall = TP / (TP + FN)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)


def f1_score(y_true, y_pred):
    """
    Calculate F1 score.
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix.
    
    Returns:
    --------
    cm : np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    return np.array([[tn, fp], [fn, tp]])


# =============================================================================
# LOGISTIC REGRESSION CLASS
# =============================================================================

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    
    This implementation supports three gradient descent optimization methods:
    1. Batch Gradient Descent - Uses all samples for each update
    2. Stochastic Gradient Descent (SGD) - Uses one sample per update
    3. Mini-Batch Gradient Descent - Uses a subset of samples per update
    
    Optimization Insights:
    ----------------------
    - Batch GD: Most stable convergence, but slow for large datasets.
      Computes exact gradient using all samples.
    
    - SGD: Fastest per iteration, but noisy updates can cause oscillation.
      The randomness can help escape local minima.
    
    - Mini-Batch GD: Best of both worlds - more stable than SGD,
      faster than Batch GD. Commonly used in practice.
    
    Parameters:
    -----------
    learning_rate : float
        Step size for gradient descent (default: 0.01)
    n_iterations : int
        Number of training iterations (default: 1000)
    optimizer : str
        Optimization method: "batch", "sgd", or "mini-batch" (default: "batch")
    batch_size : int or None
        Batch size for mini-batch gradient descent (default: 32 if mini-batch)
    regularization : float
        L2 regularization strength (default: 0.0, no regularization)
    class_weight : str or None
        If "balanced", automatically adjust weights inversely proportional to class frequencies
    early_stopping : bool
        Whether to use early stopping (default: True)
    patience : int
        Number of iterations with no improvement before stopping (default: 10)
    tol : float
        Minimum improvement to consider as progress (default: 1e-5)
    verbose : bool
        Whether to print progress during training (default: True)
    
    Attributes:
    -----------
    weights : np.ndarray
        Model weights (coefficients)
    bias : float
        Model bias (intercept)
    loss_history : list
        History of loss values during training
    weight_history : list
        History of weight values during training (for visualization)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, optimizer="batch", 
                 batch_size=None, regularization=0.0, class_weight=None,
                 early_stopping=True, patience=10, tol=1e-5, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimizer = optimizer.lower()
        self.regularization = regularization
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        
        # Set default batch size for mini-batch
        if batch_size is None:
            self.batch_size = 32 if self.optimizer == "mini-batch" else None
        else:
            self.batch_size = batch_size
        
        # Model parameters (to be learned)
        self.weights = None
        self.bias = None
        
        # Class weights (computed during fit if class_weight="balanced")
        self.class_weights_ = None
        
        # History for visualization
        self.loss_history = []
        self.weight_history = []
    
    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.
        
        σ(z) = 1 / (1 + e^(-z))
        
        Numerically stable implementation to avoid overflow.
        
        Parameters:
        -----------
        z : np.ndarray
            Linear combination of inputs and weights
        
        Returns:
        --------
        sigmoid_z : np.ndarray
            Values between 0 and 1
        """
        # Clip values to avoid overflow in exp
        z = np.clip(z, -500, 500)
        
        # Numerically stable sigmoid
        positive_mask = z >= 0
        negative_mask = ~positive_mask
        
        result = np.zeros_like(z, dtype=float)
        
        # For positive z
        result[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        
        # For negative z
        exp_z = np.exp(z[negative_mask])
        result[negative_mask] = exp_z / (1 + exp_z)
        
        return result
    
    def compute_loss(self, y_true, y_pred, sample_weights=None):
        """
        Compute the binary cross-entropy (log) loss with optional sample weights.
        
        Loss = -1/n * Σ[w * (y*log(p) + (1-y)*log(1-p))] + λ/2 * ||w||²
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels (0 or 1)
        y_pred : np.ndarray
            Predicted probabilities
        sample_weights : np.ndarray or None
            Per-sample weights for handling class imbalance
        
        Returns:
        --------
        loss : float
            Binary cross-entropy loss value
        """
        n_samples = len(y_true)
        
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss per sample
        loss_per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss = np.sum(sample_weights * loss_per_sample) / np.sum(sample_weights)
        else:
            loss = np.mean(loss_per_sample)
        
        # Add L2 regularization term
        if self.regularization > 0 and self.weights is not None:
            loss += (self.regularization / 2) * np.sum(self.weights ** 2)
        
        return loss
    
    def _compute_gradients(self, X, y, y_pred, sample_weights=None):
        """
        Compute gradients for weights and bias with optional sample weights.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        sample_weights : np.ndarray or None
            Per-sample weights
        
        Returns:
        --------
        dw, db : tuple
            Gradients for weights and bias
        """
        n_samples = len(y)
        
        # Compute error
        error = y_pred - y
        
        # Apply sample weights if provided
        if sample_weights is not None:
            weighted_error = sample_weights * error
            dw = np.dot(X.T, weighted_error) / np.sum(sample_weights)
            db = np.sum(weighted_error) / np.sum(sample_weights)
        else:
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
        
        # Add L2 regularization gradient
        if self.regularization > 0:
            dw += self.regularization * self.weights
        
        return dw, db
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Training labels of shape (n_samples,)
        
        Returns:
        --------
        self : LogisticRegression
            Trained model
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten()
        
        n_samples, n_features = X.shape
        
        # Initialize weights with small random values for better convergence
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # Compute class weights if specified
        if self.class_weight == "balanced":
            n_classes = 2
            class_counts = np.bincount(y.astype(int))
            self.class_weights_ = n_samples / (n_classes * class_counts)
            print(f"Class weights: {self.class_weights_}")
        else:
            self.class_weights_ = None
        
        # Compute per-sample weights based on class
        if self.class_weights_ is not None:
            sample_weights = np.where(y == 1, self.class_weights_[1], self.class_weights_[0])
        else:
            sample_weights = None
        
        # Clear histories
        self.loss_history = []
        self.weight_history = []
        
        # Store initial weights
        self.weight_history.append(self.weights.copy())
        
        # Choose optimization method
        if self.optimizer == "batch":
            self._fit_batch(X, y, sample_weights)
        elif self.optimizer == "sgd":
            self._fit_sgd(X, y, sample_weights)
        elif self.optimizer == "mini-batch":
            self._fit_mini_batch(X, y, sample_weights)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. "
                           f"Choose from 'batch', 'sgd', or 'mini-batch'.")
        
        return self
    
    def _fit_batch(self, X, y, sample_weights=None):
        """
        Batch Gradient Descent: Use ALL samples for each weight update.
        """
        n_samples = len(y)
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute and store loss
            loss = self.compute_loss(y, y_pred, sample_weights)
            self.loss_history.append(loss)
            
            # Early stopping check
            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration}, loss: {loss:.6f}")
                        break
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred, sample_weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store weights for visualization
            if iteration % 10 == 0:
                self.weight_history.append(self.weights.copy())
            
            # Print progress
            if self.verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: loss = {loss:.6f}")
        
        self.weight_history.append(self.weights.copy())
    
    def _fit_sgd(self, X, y, sample_weights=None):
        """
        Stochastic Gradient Descent: Use ONE sample for each weight update.
        """
        n_samples = len(y)
        
        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                xi = X[i:i+1]
                yi = y[i:i+1]
                sw_i = sample_weights[i:i+1] if sample_weights is not None else None
                
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                
                dw, db = self._compute_gradients(xi, yi, y_pred, sw_i)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss at end of epoch
            z_full = np.dot(X, self.weights) + self.bias
            y_pred_full = self.sigmoid(z_full)
            loss = self.compute_loss(y, y_pred_full, sample_weights)
            self.loss_history.append(loss)
            
            if iteration % 10 == 0:
                self.weight_history.append(self.weights.copy())
        
        self.weight_history.append(self.weights.copy())
    
    def _fit_mini_batch(self, X, y, sample_weights=None):
        """
        Mini-Batch Gradient Descent: Use a SUBSET of samples for each update.
        """
        n_samples = len(y)
        batch_size = min(self.batch_size, n_samples)
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            sw_shuffled = sample_weights[indices] if sample_weights is not None else None
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                sw_batch = sw_shuffled[start_idx:end_idx] if sw_shuffled is not None else None
                
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred, sw_batch)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss at end of epoch
            z_full = np.dot(X, self.weights) + self.bias
            y_pred_full = self.sigmoid(z_full)
            loss = self.compute_loss(y, y_pred_full, sample_weights)
            self.loss_history.append(loss)
            
            # Early stopping check
            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration}, loss: {loss:.6f}")
                        break
            
            if iteration % 10 == 0:
                self.weight_history.append(self.weights.copy())
            
            if self.verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: loss = {loss:.6f}")
        
        self.weight_history.append(self.weights.copy())
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        
        Returns:
        --------
        probabilities : np.ndarray
            Predicted probabilities for the positive class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels for samples.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        threshold : float
            Decision threshold (default: 0.5)
        
        Returns:
        --------
        predictions : np.ndarray
            Predicted binary labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_loss_history(self):
        """Get the history of loss values during training."""
        return self.loss_history
    
    def get_weight_history(self):
        """Get the history of weight values during training."""
        return self.weight_history
    
    def get_params(self):
        """Get the model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size
        }
    
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filename : str
            Path to save the model (JSON format)
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        
        model_data = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'loss_history': self.loss_history,
            'weight_history': [w.tolist() for w in self.weight_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filename : str
            Path to the saved model file
        
        Returns:
        --------
        self : LogisticRegression
            Loaded model
        """
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.weights = np.array(model_data['weights'])
        self.bias = model_data['bias']
        self.learning_rate = model_data['learning_rate']
        self.n_iterations = model_data['n_iterations']
        self.optimizer = model_data['optimizer']
        self.batch_size = model_data['batch_size']
        self.loss_history = model_data['loss_history']
        self.weight_history = [np.array(w) for w in model_data['weight_history']]
        
        print(f"Model loaded from {filename}")
        return self
