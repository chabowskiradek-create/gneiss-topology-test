# Compute power spectrum of perpendicular B-field
freq, psd = multitaper_psd(B_perp, fs=1000, NW=4, K=7)

# Estimate uncertainty from multi-taper variance
sigma_psd = sqrt(variance_multitaper(psd, K=7))

# Fit model with periodic modulation
def model(f, P0, alpha, A, f_mod):
    kolmogorov = P0 * f**(-alpha)
    modulation = 1 + A * cos(2*pi*f/f_mod)
    return kolmogorov * modulation

# Non-linear least squares fit
params, cov = curve_fit(model, freq, psd, 
                        sigma=sigma_psd,
                        p0=[1e-10, 5/3, 1.2e-3, 0.1])

# Extract modulation amplitude
A_measured = params[2]
sigma_A = sqrt(cov[2,2])

# Goodness of fit (chi-squared test)
chi2, p_value = chi2_test(model(freq, *params), psd, sigma_psd)

# Significance test
z_score = (A_measured - 1.2e-3) / sigma_A

# Verdict (requires BOTH amplitude match AND good fit)
chi2_reduced = chi2 / len(freq)  # reduced chi-squared
if (z_score > 5) and (0.8 < chi2_reduced < 1.2):
    result = "DISCOVERY"
elif (z_score > 3) and (0.5 < chi2_reduced < 1.5):
    result = "EVIDENCE"
else:
    result = "NULL"
