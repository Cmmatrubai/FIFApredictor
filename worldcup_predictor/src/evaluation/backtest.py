import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import mean_squared_error

def accuracy_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute the accuracy of predictions.
    Args:
        y_true: List of true outcomes.
        y_pred: List of predicted outcomes.
    Returns:
        Accuracy as a float.
    """
    return np.mean(np.array(y_true) == np.array(y_pred))

def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    """
    Compute the Brier score for probabilistic predictions (binary case).
    Args:
        y_true: List of true binary outcomes (0 or 1).
        y_prob: List of predicted probabilities for class 1.
    Returns:
        Brier score as a float.
    """
    return np.mean((np.array(y_true) - np.array(y_prob)) ** 2)

def rmse_score(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute the Root Mean Squared Error (RMSE).
    Args:
        y_true: List of true values.
        y_pred: List of predicted values.
    Returns:
        RMSE as a float.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(y_true: List[str], y_pred: List[str], y_prob: List[float]=None) -> Dict[str, float]:
    """
    Evaluate predictions with accuracy, Brier score, and RMSE (if applicable).
    Args:
        y_true: List of true outcomes.
        y_pred: List of predicted outcomes.
        y_prob: List of predicted probabilities for class 1 (optional).
    Returns:
        Dict of metric names to values.
    """
    results = {'accuracy': accuracy_score(y_true, y_pred)}
    if y_prob is not None:
        # For binary classification: convert y_true to 0/1
        y_true_bin = [1 if y == 'win' else 0 for y in y_true]
        results['brier'] = brier_score(y_true_bin, y_prob)
        results['rmse'] = rmse_score(y_true_bin, y_prob)
    return results

# Example usage (to be replaced with real data/model integration)
if __name__ == "__main__":
    y_true = ['win', 'draw', 'loss', 'win', 'loss']
    y_pred = ['win', 'loss', 'loss', 'win', 'draw']
    y_prob = [0.8, 0.2, 0.1, 0.7, 0.3]  # Probabilities for 'win'
    results = evaluate_predictions(y_true, y_pred, y_prob)
    print(results) 