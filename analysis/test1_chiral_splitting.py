#!/usr/bin/env python3
"""
Test 1: Chiral Wave Frequency Splitting
Expected: Δf = 0.020 ± 0.005 Hz
"""

import numpy as np
from scipy.signal import spectrogram
from typing import Tuple, Dict

def decompose_circular(E_x: np.ndarray, E_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose E-field into left/right circular polarization."""
    E_L = (E_x + 1j * E_y) / np.sqrt(2)
    E_R = (E_x - 1j * E_y) / np.sqrt(2)
    return E_L, E_R

def multitaper_psd(signal: np.ndarray, fs: float = 1000, NW: float = 4, K: int = 7):
    """
    Multi-taper power spectral density estimation.
    Uses Slepian sequences for variance reduction.
    """
    from scipy.signal import windows, periodogram
    
    n = len(signal)
    # Generate Slepian windows (discrete prolate spheroidal sequences)
    tapers = windows.dpss(n, NW, K)
    
    psd_estimates = []
    for k in range(K):
        tapered = signal * tapers[k]
        f, Pxx = periodogram(tapered, fs=fs, window='boxcar')
        psd_estimates.append(Pxx)
    
    # Average over tapers
    psd_avg = np.mean(psd_estimates, axis=0)
    return f, psd_avg

def fit_gaussian_peak(freq: np.ndarray, psd: np.ndarray, f_center: float = 1.0) -> Tuple[float, float]:
    """
    Fit Gaussian peak around expected frequency.
    Returns: (peak_freq, sigma)
    """
    from scipy.optimize import curve_fit
    
    # Select range around peak
    mask = (freq > f_center * 0.5) & (freq < f_center * 1.5)
    f_fit = freq[mask]
    p_fit = psd[mask]
    
    if len(f_fit) < 3:
        return np.nan, np.nan
    
    def gaussian(f, A, mu, sigma):
        return A * np.exp(-(f - mu)**2 / (2 * sigma**2))
    
    try:
        popt, pcov = curve_fit(gaussian, f_fit, p_fit, p0=[np.max(p_fit), f_center, 0.1])
        return popt[1], np.sqrt(pcov[1,1])  # mu, sigma_mu
    except:
        return np.nan, np.nan

def test_chiral_splitting(E_x: np.ndarray, E_y: np.ndarray, fs: float = 1000) -> Dict:
    """
    Main test function.
    
    Returns:
        dict with: delta_f, sigma, z_score, verdict
    """
    # Decompose
    E_L, E_R = decompose_circular(E_x, E_y)
    
    # PSD
    f_L, psd_L = multitaper_psd(E_L, fs)
    f_R, psd_R = multitaper_psd(E_R, fs)
    
    # Find peaks (expected ~1 Hz for Alfvén waves)
    f_peak_L, sigma_L = fit_gaussian_peak(f_L, psd_L, 1.0)
    f_peak_R, sigma_R = fit_gaussian_peak(f_R, psd_R, 1.0)
    
    if np.isnan(f_peak_L) or np.isnan(f_peak_R):
        return {'verdict': 'FAILED', 'reason': 'Peak fitting failed'}
    
    # Calculate splitting
    delta_f = abs(f_peak_L - f_peak_R)
    sigma_delta = np.sqrt(sigma_L**2 + sigma_R**2)
    
    # Test against prediction: 0.020 ± 0.005 Hz
    expected = 0.020
    expected_sigma = 0.005
    
    z_score = (delta_f - expected) / np.sqrt(sigma_delta**2 + expected_sigma**2)
    
    # Verdict
    if z_score > 5:
        verdict = "DISCOVERY"
    elif z_score > 3:
        verdict = "EVIDENCE"
    else:
        verdict = "NULL"
    
    return {
        'delta_f': delta_f,
        'sigma': sigma_delta,
        'z_score': z_score,
        'verdict': verdict,
        'f_L': f_peak_L,
        'f_R': f_peak_R
    }

if __name__ == "__main__":
    # Synthetic test
    t = np.linspace(0, 30, 30000)  # 30s, 1kHz
    # Generate split frequencies
    f_L = 1.00
    f_R = 1.02  # 0.02 Hz splitting
    E_x = np.cos(2*np.pi*f_L*t) + np.cos(2*np.pi*f_R*t)
    E_y = np.sin(2*np.pi*f_L*t) - np.sin(2*np.pi*f_R*t)
    E_x += 0.1 * np.random.randn(len(t))  # Noise
    
    result = test_chiral_splitting(E_x, E_y)
    print(f"Test 1 Results: {result}")
