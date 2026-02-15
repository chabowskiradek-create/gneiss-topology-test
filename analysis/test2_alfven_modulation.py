#!/usr/bin/env python3
"""
Test 2: Alfvén Wave Spectral Modulation
Expected: A = (1.2 ± 0.4) × 10^-3 periodic modulation
"""

import numpy as np
from scipy.optimize import curve_fit

def kolmogorov_modulation(f, P0, alpha, A, f_mod):
    """
    Kolmogorov spectrum with periodic modulation.
    P(f) = P0 * f^-alpha * [1 + A * cos(2*pi*f/f_mod)]
    """
    kolm = P0 * f**(-alpha)
    mod = 1 + A * np.cos(2 * np.pi * f / f_mod)
    return kolm * mod

def test_alfven_modulation(freq: np.ndarray, psd: np.ndarray) -> dict:
    """
    Fit for periodic modulation in power spectrum.
    
    Expected:
        A = 1.2e-3 ± 0.4e-3
        f_mod ~ 0.1 Hz (corresponding to 10 km scale)
    """
    # Initial guess
    p0 = [1e-10, 5/3, 1.2e-3, 0.1]
    
    try:
        params, cov = curve_fit(kolmogorov_modulation, freq, psd, p0=p0, 
                               bounds=([0, 1, 0, 0.05], [np.inf, 3, 0.01, 0.5]))
        
        A_meas = params[2]
        sigma_A = np.sqrt(cov[2,2])
        
        # Reduced chi-squared
        residuals = psd - kolmogorov_modulation(freq, *params)
        chi2 = np.sum((residuals / np.std(psd))**2)
        chi2_red = chi2 / len(freq)
        
        # Z-score vs expected
        expected_A = 1.2e-3
        z_score = (A_meas - expected_A) / sigma_A
        
        # Verdict
        if (z_score > 5) and (0.8 < chi2_red < 1.2):
            verdict = "DISCOVERY"
        elif (z_score > 3) and (0.5 < chi2_red < 1.5):
            verdict = "EVIDENCE"
        else:
            verdict = "NULL"
            
        return {
            'A_measured': A_meas,
            'sigma_A': sigma_A,
            'z_score': z_score,
            'chi2_reduced': chi2_red,
            'verdict': verdict
        }
        
    except Exception as e:
        return {'verdict': 'ERROR', 'message': str(e)}

if __name__ == "__main__":
    # Synthetic test
    f = np.linspace(0.1, 10, 1000)
    P = kolmogorov_modulation(f, 1e-10, 5/3, 1.2e-3, 0.1)
    P += 0.1 * P * np.random.randn(len(f))  # 10% noise
    
    result = test_alfven_modulation(f, P)
    print(f"Test 2 Results: {result}")
