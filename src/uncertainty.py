import torch
import torch.nn.functional as F
import numpy as np

def enable_dropout(model):
    """Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def monte_carlo_dropout_predict(model, input_tensor, num_samples=15):
    """
    Perform Monte Carlo Dropout inference over multiple stochastic forward passes.
    
    Args:
        model: PyTorch classification model with Dropout layers
        input_tensor: preprocessed image tensor (1, 3, H, W)
        num_samples: number of stochastic forward passes (default: 15)
        
    Returns:
        dict containing:
            mean_probs: np.ndarray [P(Healthy), P(Unhealthy)]
            variance: np.ndarray [Var(Healthy), Var(Unhealthy)]
            epistemic_uncertainty: float (average variance across classes)
            predictive_entropy: float (Shannon entropy in bits [0, 1])
            predicted_class: str ("Healthy" or "Unhealthy (PCOS)")
            confidence: float (mean probability of the predicted class)
            sample_predictions: list of probabilities across MC runs
    """
    model.eval()
    enable_dropout(model)
    
    probs_list = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list.append(probs)
            
    probs_array = np.array(probs_list)  # (num_samples, num_classes)
    
    # Calculate statistics across MC samples
    mean_probs = np.mean(probs_array, axis=0)
    var_probs = np.var(probs_array, axis=0)
    
    # Epistemic uncertainty = mean variance across classes
    epistemic_uncertainty = float(np.mean(var_probs))
    
    # Predictive Shannon Entropy: H = - sum(p * log2(p))
    # Normalized between 0 and 1 for 2 classes (max entropy = log2(2) = 1.0)
    entropy = -sum([p * np.log2(p + 1e-12) for p in mean_probs if p > 0])
    normalized_entropy = float(np.clip(entropy / 1.0, 0.0, 1.0))
    
    # Class 0: Healthy, Class 1: Unhealthy (PCOS)
    p_healthy = float(mean_probs[0])
    p_unhealthy = float(mean_probs[1])
    
    if p_unhealthy > p_healthy:
        pred_class = "Unhealthy (PCOS)"
        confidence = p_unhealthy
    else:
        pred_class = "Healthy"
        confidence = p_healthy
        
    return {
        "mean_probs": mean_probs.tolist(),
        "healthy_prob": p_healthy,
        "unhealthy_prob": p_unhealthy,
        "variance": var_probs.tolist(),
        "epistemic_uncertainty": epistemic_uncertainty,
        "predictive_entropy": normalized_entropy,
        "predicted_class": pred_class,
        "confidence": confidence,
        "mc_samples": num_samples,
        "mc_distribution": [float(p[1]) for p in probs_array] # Unhealthy prob across samples
    }


def compute_brier_score(y_true, y_prob):
    """
    Compute Brier score: Mean squared difference between predicted probability and actual binary outcome.
    Lower is better (0.0 is perfect calibration, 0.25 is random guessing).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))
