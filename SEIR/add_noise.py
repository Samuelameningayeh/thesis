import numpy as np

def Add_noise(data, reporting_rate, phi, seed=None):
    """
    Adds negative binomial noise to data with a fixed seed for reproducibility.

    Args:
        data: array-like, true (latent) cases, shape [patches, time]
        reporting_rate: float, observed/reporting probability
        phi: int or float, dispersion parameter for NegBin (higher = less variance)
        seed: int or None, seed for np.random

    Returns:
        observed_cases: np.ndarray of noisy observed cases
    """
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility

    data = np.array(data)
    observed_cases = []
    for i in range(data.shape[0]):
        case = []
        for t in range(data.shape[1]):
            mean_cases = reporting_rate * data[i, t]
            p = phi / (phi + mean_cases)
            # Avoid edge case for p > 1 (when mean_cases is 0)
            # p = np.clip(p, 0, 1)
            case.append(np.random.negative_binomial(phi, p))
        observed_cases.append(case)
    return np.array(observed_cases)

