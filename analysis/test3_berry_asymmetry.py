#!/usr/bin/env python3
"""
Test 3: Berry Phase Azimuthal Asymmetry
Expected: A_phi = 0.06 ± 0.02 (m=1 Fourier mode)
"""

import numpy as np
from numpy.fft import fft2

def test_berry_asymmetry(f_dist: np.ndarray, alpha_bins: np.ndarray, 
                         energy_mask: slice = slice(None)) -> dict:
    """
    Analyze pitch-angle distribution for azimuthal asymmetry.
    
    Args:
        f_dist: 3D array (pitch_angle, azimuth, energy)
        alpha_bins: Pitch angle values (degrees)
        energy_mask: Slice selecting energy range
    
    Returns:
        dict with asymmetry amplitude and significance
    """
    # Select energy range
    f_selected = f_dist[:, :, energy_mask]
    
    # Average over energy if needed
    if f_selected.ndim == 3:
        f_selected = np.mean(f_selected, axis=2)
    
    # 2D FFT in (alpha, phi)
    f_fourier = fft2(f_selected)
    
    # Extract m=1 azimuthal mode (index 1 in second dimension)
    f_m1 = f_fourier[:, 1]
    f_m0 = f_fourier[:, 0]  # DC component
    
    # Asymmetry amplitude (normalized)
    A_phi_arr = np.abs(f_m1) / (np.abs(f_m0) + 1e-12)  # avoid div by 0
    
    # Average over pitch angles 30-60 degrees (mirror region)
    mask = (alpha_bins > 30) & (alpha_bins < 60)
    if np.sum(mask) == 0:
        return {'verdict': 'FAILED', 'reason': 'No bins in 30-60 range'}
    
    A_phi_mean = np.mean(A_phi_arr[mask])
    A_phi_std = np.std(A_phi_arr[mask]) / np.sqrt(np.sum(mask))
    
    # Test vs expected 0.06 ± 0.02
    expected = 0.06
    expected_sigma = 0.02
    
    z_score = (A_phi_mean - expected) / np.sqrt(A_phi_std**2 + expected_sigma**2)
    
    if z_score > 5:
        verdict = "DISCOVERY"
    elif z_score > 3:
        verdict = "EVIDENCE"
    else:
        verdict = "NULL"
    
    return {
        'A_phi': A_phi_mean,
        'sigma': A_phi_std,
        'z_score': z_score,
        'verdict': verdict
    }

if __name__ == "__main__":
    # Synthetic test: 6% asymmetry
    alpha = np.linspace(0, 90, 90)
    phi = np.linspace(0, 2*np.pi, 360)
    ALPHA, PHI = np.meshgrid(alpha, phi)
    
    # Distribution with m=1 asymmetry
    f = np.exp(-((ALPHA - 45)/20)**2) * (1 + 0.06 * np.cos(PHI))
    
    result = test_berry_asymmetry(f.T, alpha)
    print(f"Test 3 Results: {result}")
