# Decompose E-field into circular polarizations
E_L = (E_x + 1j*E_y) / sqrt(2)   # Left circular
E_R = (E_x - 1j*E_y) / sqrt(2)   # Right circular

# Power spectra via multi-taper method (NW=4, K=7 tapers)
freq_L, psd_L = multitaper_psd(E_L, fs=1000, NW=4, K=7)
freq_R, psd_R = multitaper_psd(E_R, fs=1000, NW=4, K=7)

# Peak detection with Gaussian fit
f_peak_L, sigma_L = fit_gaussian_peak(freq_L, psd_L)
f_peak_R, sigma_R = fit_gaussian_peak(freq_R, psd_R)

# Measured splitting
Delta_f = abs(f_peak_L - f_peak_R)
sigma_Delta_f = sqrt(sigma_L**2 + sigma_R**2)

# Significance test
z_score = (Delta_f - 0.020) / sigma_Delta_f

# Verdict
if z_score > 5 and both_peaks_significant:
    result = "DISCOVERY"
elif z_score > 3:
    result = "EVIDENCE"
else:
    result = "NULL"
