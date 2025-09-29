import torch
from torch.utils.data import Dataset
from typing import Tuple, Callable, Union, List, Dict
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def check_stationarity(k_dim: int, n_lags: int, a_coeffs: np.ndarray, b_coeffs: np.ndarray, c_coeffs: np.ndarray, d_coeffs: np.ndarray) -> Tuple[bool, np.ndarray]:
    assert len(a_coeffs) == n_lags, f"Length of a_coeffs ({len(a_coeffs)}) must equal n_lags ({n_lags})"
    assert len(b_coeffs) == n_lags, f"Length of b_coeffs ({len(b_coeffs)}) must equal n_lags ({n_lags})"
    assert len(c_coeffs) == n_lags, f"Length of c_coeffs ({len(c_coeffs)}) must equal n_lags ({n_lags})"
    assert len(d_coeffs) == n_lags, f"Length of d_coeffs ({len(d_coeffs)}) must equal n_lags ({n_lags})"
    phi_matrices = []
    I_k = np.eye(k_dim)
    for i in range(n_lags):
        a, b = a_coeffs[i], b_coeffs[i]
        c, d = c_coeffs[i], d_coeffs[i]
        phi_i = np.block([[a * I_k, b * I_k], [d * I_k, c * I_k]])
        phi_matrices.append(phi_i)

    dim = 2 * k_dim * n_lags
    F = np.zeros((dim, dim))
    F[0:2*k_dim, :] = np.hstack(phi_matrices)
    if n_lags > 1:
        F[2*k_dim:, :-2*k_dim] = np.eye(2 * k_dim * (n_lags - 1))

    max_eigenvalue_modulus = np.max(np.abs(np.linalg.eigvals(F)))
    return max_eigenvalue_modulus < 1, F # Return the companion matrix F as well

def generate_stable_coeffs(k_dim: int, n_lags: int, is_coupled: bool = True, max_tries: int = 1000) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Finds random VAR coefficients that result in a stationary process
    using rejection sampling.
    """
    for _ in range(max_tries):
        # Sample coefficients from a uniform distribution.
        # The range [-1, 1] is a reasonable starting point.
        a = np.random.uniform(-1, 1, size=n_lags)
        c = np.random.uniform(-1, 1, size=n_lags)

        if is_coupled:
            b = np.random.uniform(-1, 1, size=n_lags)
            d = np.random.uniform(-1, 1, size=n_lags)
        else:
            # If not coupled, these coefficients are zero
            b = np.zeros(n_lags)
            d = np.zeros(n_lags)

        is_stable, F_matrix = check_stationarity(k_dim, n_lags, a, b, c, d)
        
        if is_stable:
            print("Found stable coefficients.")
            coeffs = {'a': a, 'b': b, 'c': c, 'd': d}
            return coeffs, F_matrix

    raise RuntimeError(f"Failed to find stable coefficients after {max_tries} attempts.")

# You would need the generate_var_data function from our previous conversation
def generate_var_data(n_samples: int, k_dim: int, n_lags: int, a_coeffs: np.ndarray, b_coeffs: np.ndarray, c_coeffs: np.ndarray, d_coeffs: np.ndarray, noise_cov_y: np.ndarray, noise_cov_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = np.zeros((n_samples, k_dim)), np.zeros((n_samples, k_dim))
    a, b = np.array(a_coeffs).reshape(-1, 1), np.array(b_coeffs).reshape(-1, 1)
    c, d = np.array(c_coeffs).reshape(-1, 1), np.array(d_coeffs).reshape(-1, 1)
    for t in range(n_lags, n_samples):
        y_lags, x_lags = Y[t-n_lags:t][::-1], X[t-n_lags:t][::-1]
        noise_epsilon = np.random.multivariate_normal(np.zeros(k_dim), noise_cov_y)
        noise_delta = np.random.multivariate_normal(np.zeros(k_dim), noise_cov_x)
        Y[t] = np.sum(a * y_lags, axis=0) + np.sum(b * x_lags, axis=0) + noise_epsilon
        X[t] = np.sum(c * x_lags, axis=0) + np.sum(d * y_lags, axis=0) + noise_delta
    return X, Y


def generate_controlled_var_data(n_samples: int, k_dim: int, n_lags: int, is_coupled: bool, target_avg_variance: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates stationary VAR data with user-defined controls.
    """
    # 1. Generate a set of stable coefficients
    coeffs, F_matrix = generate_stable_coeffs(k_dim, n_lags, is_coupled)
    
    # 2. Calculate stationary variance with initial noise covariance
    # Use identity matrix for the initial noise covariance
    stacked_dim = 2 * k_dim * n_lags
    noise_cov_stacked = np.zeros((stacked_dim, stacked_dim))
    noise_cov_stacked[0:2*k_dim, 0:2*k_dim] = np.eye(2 * k_dim)

    # Solve the Lyapunov equation for the stationary variance
    # This gives the variance of the stacked vector [Z_t, Z_{t-1}, ...]
    P = solve_discrete_lyapunov(F_matrix, noise_cov_stacked)
    
    # The variance of just Z_t is the top-left block
    var_z_initial = P[0:2*k_dim, 0:2*k_dim]
    
    # 3. Calculate scaling factor for the noise
    current_avg_variance = np.mean(np.diag(var_z_initial))
    scale_factor = target_avg_variance / current_avg_variance
    
    # The new noise covariance is the scaled identity matrix
    scaled_noise_cov = np.eye(k_dim) * scale_factor
    
    # 4. Generate the final data with the stable coeffs and scaled noise
    X_data, Y_data = generate_var_data(
        n_samples=n_samples, k_dim=k_dim, n_lags=n_lags,
        a_coeffs=coeffs['a'], b_coeffs=coeffs['b'],
        c_coeffs=coeffs['c'], d_coeffs=coeffs['d'],
        noise_cov_y=scaled_noise_cov, # Use the same scaled noise for both
        noise_cov_x=scaled_noise_cov
    )
    
    print(f"Generated data with actual average variance: {np.mean(np.var(np.hstack([X_data, Y_data]), axis=0)):.4f}")
    
    return X_data, Y_data