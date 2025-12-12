"""
Logistic Regression Implementation from Scratch
Tugas Besar 2 IF3070 – Dasar Inteligensi Artifisial

This module implements Logistic Regression with multiple gradient descent optimizers
using only numpy and pandas.

Features:
- Multiple optimizers: Batch GD, SGD, Mini-batch GD
- Learning rate schedules: constant, step decay, exponential decay, cosine annealing
- Momentum and Nesterov momentum
- L1, L2, and Elastic Net regularization
- Class weighting for imbalanced datasets
- Early stopping with patience
- Focal loss for hard example mining

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


def roc_auc_score(y_true, y_scores):
    """
    Calculate ROC AUC (Area Under the Receiver Operating Characteristic Curve).
    
    This is implemented from scratch using the trapezoidal rule.
    AUC measures the model's ability to distinguish between classes.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_scores : np.ndarray
        Predicted probabilities or scores for the positive class
    
    Returns:
    --------
    auc : float
        AUC score between 0 and 1
        - 1.0 = perfect classifier
        - 0.5 = random classifier
        - < 0.5 = worse than random
    """
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()
    
    # Get unique thresholds (sorted descending)
    # Add boundary values
    thresholds = np.unique(y_scores)
    thresholds = np.concatenate([[thresholds.max() + 1], thresholds, [thresholds.min() - 1]])
    thresholds = np.sort(thresholds)[::-1]  # Sort descending
    
    # Calculate TPR and FPR for each threshold
    tpr_list = []  # True Positive Rate (Sensitivity/Recall)
    fpr_list = []  # False Positive Rate (1 - Specificity)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Cannot compute AUC with only one class
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    tpr_array = np.array(tpr_list)
    fpr_array = np.array(fpr_list)
    
    # Sort by FPR (ascending) for proper integration
    sorted_indices = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sorted_indices]
    tpr_sorted = tpr_array[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    
    return auc


def roc_curve(y_true, y_scores, n_thresholds=100):
    """
    Calculate ROC curve points.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Predicted probabilities
    n_thresholds : int
        Number of threshold points to calculate
    
    Returns:
    --------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    thresholds : np.ndarray
        Thresholds used
    """
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()
    
    # Generate thresholds
    thresholds = np.linspace(0, 1, n_thresholds)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


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
    
    This implementation supports multiple advanced features:
    
    Optimization Methods:
    ---------------------
    1. Batch Gradient Descent - Uses all samples for each update
    2. Stochastic Gradient Descent (SGD) - Uses one sample per update
    3. Mini-Batch Gradient Descent - Uses a subset of samples per update
    
    Learning Rate Schedules:
    ------------------------
    - constant: Fixed learning rate
    - step: Reduce by factor every N epochs
    - exponential: Exponentially decay learning rate
    - cosine: Cosine annealing schedule
    
    Regularization:
    ---------------
    - L2 (Ridge): Penalizes large weights, prevents overfitting
    - L1 (Lasso): Encourages sparsity, feature selection
    - Elastic Net: Combination of L1 and L2
    
    Loss Functions:
    ---------------
    - Binary Cross-Entropy: Standard logistic loss
    - Focal Loss: Down-weights easy examples, focuses on hard ones
    
    Parameters:
    -----------
    learning_rate : float
        Initial step size for gradient descent (default: 0.01)
    n_iterations : int
        Number of training iterations (default: 1000)
    optimizer : str
        Optimization method: "batch", "sgd", or "mini-batch" (default: "batch")
    batch_size : int or None
        Batch size for mini-batch gradient descent (default: 32 if mini-batch)
    regularization : float
        L2 regularization strength (default: 0.0)
    l1_ratio : float
        Ratio of L1 regularization (0 = pure L2, 1 = pure L1) (default: 0.0)
    class_weight : str or None
        If "balanced", automatically adjust weights inversely proportional to class frequencies
    lr_schedule : str
        Learning rate schedule: "constant", "step", "exponential", "cosine" (default: "constant")
    lr_decay : float
        Decay factor for learning rate schedule (default: 0.1)
    lr_decay_steps : int
        Steps between learning rate reductions for step schedule (default: 100)
    momentum : float
        Momentum factor for gradient updates (default: 0.0)
    nesterov : bool
        Whether to use Nesterov momentum (default: False)
    use_focal_loss : bool
        Whether to use focal loss instead of BCE (default: False)
    focal_gamma : float
        Focusing parameter for focal loss (default: 2.0)
    early_stopping : bool
        Whether to use early stopping (default: True)
    patience : int
        Number of iterations with no improvement before stopping (default: 10)
    tol : float
        Minimum improvement to consider as progress (default: 1e-5)
    verbose : bool
        Whether to print progress during training (default: True)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, optimizer="batch", 
                 batch_size=None, regularization=0.0, l1_ratio=0.0, class_weight=None,
                 lr_schedule="constant", lr_decay=0.1, lr_decay_steps=100,
                 momentum=0.0, nesterov=False,
                 # PARAMETER BARU UNTUK ADAM:
                 beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_focal_loss=False, focal_gamma=2.0,
                 early_stopping=True, patience=10, tol=1e-5, verbose=True):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.n_iterations = n_iterations
        self.optimizer = optimizer.lower()
        self.regularization = regularization
        self.l1_ratio = l1_ratio  # 0 = L2, 1 = L1, between = Elastic Net
        self.class_weight = class_weight
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.momentum = momentum
        self.nesterov = nesterov
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
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
        
        # Momentum velocities
        self.velocity_w = None
        self.velocity_b = None
        
        # Adam optimizer moments
        self.m_w = None  # First moment estimate (mean of gradients)
        self.v_w = None  # Second moment estimate (uncentered variance)
        self.m_b = None
        self.v_b = None
        self.t = 0  # Timestep for Adam
        
        # Class weights (computed during fit if class_weight="balanced")
        self.class_weights_ = None
        
        # History for visualization
        self.loss_history = []
        self.weight_history = []
        self.lr_history = []
    
    def _get_learning_rate(self, iteration):
        """
        Get learning rate based on schedule.
        
        Learning Rate Schedules:
        - constant: lr = lr_0
        - step: lr = lr_0 * decay^(iteration // decay_steps)
        - exponential: lr = lr_0 * decay^iteration
        - cosine: lr = lr_0 * (1 + cos(π * iteration / n_iterations)) / 2
        """
        if self.lr_schedule == "constant":
            return self.initial_lr
        
        elif self.lr_schedule == "step":
            # Reduce learning rate every lr_decay_steps iterations
            return self.initial_lr * (self.lr_decay ** (iteration // self.lr_decay_steps))
        
        elif self.lr_schedule == "exponential":
            # Exponential decay
            return self.initial_lr * (self.lr_decay ** (iteration / self.n_iterations))
        
        elif self.lr_schedule == "cosine":
            # Cosine annealing
            return self.initial_lr * (1 + np.cos(np.pi * iteration / self.n_iterations)) / 2
        
        else:
            return self.initial_lr
    
    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.
        
        σ(z) = 1 / (1 + e^(-z))
        
        Numerically stable implementation to avoid overflow.
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
        Compute the loss function.
        
        Supports:
        - Binary Cross-Entropy: -1/n * Σ[w * (y*log(p) + (1-y)*log(1-p))]
        - Focal Loss: -1/n * Σ[w * (1-p_t)^γ * log(p_t)]
          where p_t = p if y=1, else 1-p
        
        Plus regularization term: λ/2 * (α||w||₁ + (1-α)||w||₂²)
        """
        n_samples = len(y_true)
        
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if self.use_focal_loss:
            # Focal Loss: focuses training on hard examples
            # FL(p_t) = -(1 - p_t)^γ * log(p_t)
            p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
            focal_weight = (1 - p_t) ** self.focal_gamma
            loss_per_sample = -focal_weight * np.log(p_t)
        else:
            # Binary Cross-Entropy Loss
            loss_per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss = np.sum(sample_weights * loss_per_sample) / np.sum(sample_weights)
        else:
            loss = np.mean(loss_per_sample)
        
        # Add regularization term (Elastic Net: combination of L1 and L2)
        if self.regularization > 0 and self.weights is not None:
            l1_term = self.l1_ratio * np.sum(np.abs(self.weights))
            l2_term = (1 - self.l1_ratio) * 0.5 * np.sum(self.weights ** 2)
            loss += self.regularization * (l1_term + l2_term)
        
        return loss
    
    def _compute_gradients(self, X, y, y_pred, sample_weights=None):
        """
        Compute gradients for weights and bias.
        
        For focal loss, the gradient is modified to account for the focusing term.
        """
        n_samples = len(y)
        
        if self.use_focal_loss:
            # Focal loss gradients
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            p_t = np.where(y == 1, y_pred_clipped, 1 - y_pred_clipped)
            
            # Gradient of focal loss
            gamma = self.focal_gamma
            focal_weight = (1 - p_t) ** gamma
            
            # Complex gradient for focal loss
            grad_p = np.where(
                y == 1,
                -focal_weight * (gamma * (1 - y_pred_clipped) * np.log(y_pred_clipped + epsilon) + 1) / (y_pred_clipped + epsilon),
                focal_weight * (gamma * y_pred_clipped * np.log(1 - y_pred_clipped + epsilon) + 1) / (1 - y_pred_clipped + epsilon)
            )
            
            error = grad_p * y_pred_clipped * (1 - y_pred_clipped)  # Chain rule with sigmoid derivative
        else:
            # Standard BCE gradient
            error = y_pred - y
        
        # Apply sample weights if provided
        if sample_weights is not None:
            weighted_error = sample_weights * error
            dw = np.dot(X.T, weighted_error) / np.sum(sample_weights)
            db = np.sum(weighted_error) / np.sum(sample_weights)
        else:
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
        
        # Add regularization gradient (Elastic Net)
        if self.regularization > 0:
            # L2 gradient
            l2_grad = (1 - self.l1_ratio) * self.weights
            # L1 gradient (subgradient)
            l1_grad = self.l1_ratio * np.sign(self.weights)
            dw += self.regularization * (l1_grad + l2_grad)
        
        return dw, db
    
    def _update_parameters(self, dw, db, lr):
        """
        Update parameters with optional momentum.
        
        Standard momentum: v = μ*v - lr*grad; w = w + v
        Nesterov momentum: Look ahead before computing gradient
        """
        if self.momentum > 0:
            # Momentum update
            self.velocity_w = self.momentum * self.velocity_w - lr * dw
            self.velocity_b = self.momentum * self.velocity_b - lr * db
            
            self.weights += self.velocity_w
            self.bias += self.velocity_b
        else:
            # Standard gradient descent update
            self.weights -= lr * dw
            self.bias -= lr * db
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten()
        
        n_samples, n_features = X.shape
        
        # Initialize weights with small random values (Xavier initialization)
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        
        # Initialize momentum velocities
        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0
        
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
        self.lr_history = []
        
        # Store initial weights
        self.weight_history.append(self.weights.copy())
        
        # Choose optimization method
        if self.optimizer == "batch":
            self._fit_batch(X, y, sample_weights)
        elif self.optimizer == "sgd":
            self._fit_sgd(X, y, sample_weights)
        elif self.optimizer == "mini-batch":
            self._fit_mini_batch(X, y, sample_weights)
        elif self.optimizer == "adam":
            self._fit_adam(X, y, sample_weights)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. "
                           f"Choose from 'batch', 'sgd', 'mini-batch', or 'adam'.")
        
        return self
    
    def _fit_batch(self, X, y, sample_weights=None):
        """
        Batch Gradient Descent with momentum and learning rate schedule.
        """
        best_loss = float('inf')
        patience_counter = 0
        best_weights = self.weights.copy()
        best_bias = self.bias
        
        for iteration in range(self.n_iterations):
            # Get current learning rate
            lr = self._get_learning_rate(iteration)
            self.lr_history.append(lr)
            
            # For Nesterov momentum, look ahead
            if self.nesterov and self.momentum > 0:
                self.weights += self.momentum * self.velocity_w
                self.bias += self.momentum * self.velocity_b
            
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
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration}, loss: {loss:.6f}")
                        # Restore best weights
                        self.weights = best_weights
                        self.bias = best_bias
                        break
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred, sample_weights)
            
            # Update parameters
            self._update_parameters(dw, db, lr)
            
            # Store weights for visualization
            if iteration % 10 == 0:
                self.weight_history.append(self.weights.copy())
            
            # Print progress
            if self.verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: loss = {loss:.6f}, lr = {lr:.6f}")
        
        self.weight_history.append(self.weights.copy())
    
    def _fit_sgd(self, X, y, sample_weights=None):
        """
        Stochastic Gradient Descent with momentum.
        """
        n_samples = len(y)
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            lr = self._get_learning_rate(iteration)
            self.lr_history.append(lr)
            
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                xi = X[i:i+1]
                yi = y[i:i+1]
                sw_i = sample_weights[i:i+1] if sample_weights is not None else None
                
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                
                dw, db = self._compute_gradients(xi, yi, y_pred, sw_i)
                self._update_parameters(dw, db, lr)
            
            # Compute loss at end of epoch
            z_full = np.dot(X, self.weights) + self.bias
            y_pred_full = self.sigmoid(z_full)
            loss = self.compute_loss(y, y_pred_full, sample_weights)
            self.loss_history.append(loss)
            
            # Early stopping
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
                print(f"Iteration {iteration}: loss = {loss:.6f}, lr = {lr:.6f}")
        
        self.weight_history.append(self.weights.copy())
    
    def _fit_mini_batch(self, X, y, sample_weights=None):
        """
        Mini-Batch Gradient Descent with momentum and learning rate schedule.
        """
        n_samples = len(y)
        batch_size = min(self.batch_size, n_samples)
        best_loss = float('inf')
        patience_counter = 0
        best_weights = self.weights.copy()
        best_bias = self.bias
        
        for iteration in range(self.n_iterations):
            lr = self._get_learning_rate(iteration)
            self.lr_history.append(lr)
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            sw_shuffled = sample_weights[indices] if sample_weights is not None else None
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                sw_batch = sw_shuffled[start_idx:end_idx] if sw_shuffled is not None else None
                
                # For Nesterov momentum
                if self.nesterov and self.momentum > 0:
                    self.weights += self.momentum * self.velocity_w
                    self.bias += self.momentum * self.velocity_b
                
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred, sw_batch)
                self._update_parameters(dw, db, lr)
            
            # Compute loss at end of epoch
            z_full = np.dot(X, self.weights) + self.bias
            y_pred_full = self.sigmoid(z_full)
            loss = self.compute_loss(y, y_pred_full, sample_weights)
            self.loss_history.append(loss)
            
            # Early stopping check
            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration}, loss: {loss:.6f}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break
            
            if iteration % 10 == 0:
                self.weight_history.append(self.weights.copy())
            
            if self.verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: loss = {loss:.6f}, lr = {lr:.6f}")
        
        self.weight_history.append(self.weights.copy())
    
    def _fit_adam(self, X, y, sample_weights=None):
        """
        Adam (Adaptive Moment Estimation) Optimizer.
        
        Adam combines the advantages of:
        1. Momentum (using first moment - mean of gradients)
        2. RMSprop (using second moment - uncentered variance of gradients)
        
        Update rules:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          # First moment estimate
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         # Second moment estimate
        m̂_t = m_t / (1 - β₁^t)                       # Bias-corrected first moment
        v̂_t = v_t / (1 - β₂^t)                       # Bias-corrected second moment
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        # Parameter update
        
        Parameters:
        -----------
        beta1 : float (default=0.9)
            Exponential decay rate for first moment estimates
        beta2 : float (default=0.999)
            Exponential decay rate for second moment estimates
        epsilon : float (default=1e-8)
            Small constant for numerical stability
        """
        n_samples = len(y)
        n_features = X.shape[1]
        batch_size = self.batch_size if self.batch_size else 32
        batch_size = min(batch_size, n_samples)
        
        # Initialize Adam moments
        self.m_w = np.zeros(n_features)  # First moment for weights
        self.v_w = np.zeros(n_features)  # Second moment for weights
        self.m_b = 0.0  # First moment for bias
        self.v_b = 0.0  # Second moment for bias
        self.t = 0  # Timestep
        
        best_loss = float('inf')
        patience_counter = 0
        best_weights = self.weights.copy()
        best_bias = self.bias
        
        for iteration in range(self.n_iterations):
            # For Adam, we typically use constant learning rate (or can use schedule)
            lr = self._get_learning_rate(iteration)
            self.lr_history.append(lr)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            sw_shuffled = sample_weights[indices] if sample_weights is not None else None
            
            # Mini-batch updates with Adam
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                sw_batch = sw_shuffled[start_idx:end_idx] if sw_shuffled is not None else None
                
                # Forward pass
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                
                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred, sw_batch)
                
                # Increment timestep
                self.t += 1
                
                # Update first moment estimate (momentum)
                self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
                self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
                
                # Update second moment estimate (RMSprop-like)
                self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
                self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
                
                # Bias correction
                m_w_corrected = self.m_w / (1 - self.beta1 ** self.t)
                m_b_corrected = self.m_b / (1 - self.beta1 ** self.t)
                v_w_corrected = self.v_w / (1 - self.beta2 ** self.t)
                v_b_corrected = self.v_b / (1 - self.beta2 ** self.t)
                
                # Update parameters
                self.weights -= lr * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
                self.bias -= lr * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
            
            # Compute loss at end of epoch
            z_full = np.dot(X, self.weights) + self.bias
            y_pred_full = self.sigmoid(z_full)
            loss = self.compute_loss(y, y_pred_full, sample_weights)
            self.loss_history.append(loss)
            
            # Early stopping check
            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration}, loss: {loss:.6f}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break
            
            if iteration % 10 == 0:
                self.weight_history.append(self.weights.copy())
            
            if self.verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: loss = {loss:.6f}, lr = {lr:.6f}")
        
        self.weight_history.append(self.weights.copy())
    
    def predict_proba(self, X):
        """Predict class probabilities for samples."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict binary class labels for samples."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_loss_history(self):
        """Get the history of loss values during training."""
        return self.loss_history
    
    def get_weight_history(self):
        """Get the history of weight values during training."""
        return self.weight_history
    
    def get_lr_history(self):
        """Get the history of learning rates during training."""
        return self.lr_history
    
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
        """Save the trained model to a file."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        
        model_data = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'regularization': self.regularization,
            'l1_ratio': self.l1_ratio,
            'momentum': self.momentum,
            'lr_schedule': self.lr_schedule,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
            'loss_history': self.loss_history,
            'weight_history': [w.tolist() for w in self.weight_history],
            'lr_history': self.lr_history
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model from a file."""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.weights = np.array(model_data['weights'])
        self.bias = model_data['bias']
        self.learning_rate = model_data['learning_rate']
        self.n_iterations = model_data['n_iterations']
        self.optimizer = model_data['optimizer']
        self.batch_size = model_data['batch_size']
        self.loss_history = model_data.get('loss_history', [])
        self.weight_history = [np.array(w) for w in model_data.get('weight_history', [])]
        self.lr_history = model_data.get('lr_history', [])
        
        print(f"Model loaded from {filename}")
        return self
