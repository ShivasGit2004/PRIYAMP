import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.uncertainty import compute_brier_score

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """
    Calculate comprehensive medical classification metrics.
    
    Args:
        y_true: binary ground truth labels (0 = Healthy, 1 = Unhealthy/PCOS)
        y_probs: predicted probabilities for positive class (Unhealthy/PCOS)
        threshold: decision threshold (default 0.5)
        
    Returns:
        dict containing:
            Accuracy, Precision, Sensitivity (Recall), Specificity, F1-Score, ROC-AUC, Brier Score,
            Confusion matrix (TP, FP, TN, FN)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0) # Sensitivity / Recall
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5
        
    brier = compute_brier_score(y_true, y_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity = TN / (TN + FP)
    
    return {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "sensitivity": float(round(sens, 4)),
        "specificity": float(round(spec, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(auc, 4)),
        "brier_score": float(round(brier, 4)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_samples": int(len(y_true))
    }
