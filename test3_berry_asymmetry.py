# Load pitch-angle distribution f(E, alpha, phi)
# Select energy range to maximize signal-to-noise
f_selected = f_dist[:, :, energy_mask]

# 2D Fourier transform in (alpha, phi) space
f_fourier = fft2(f_selected, axes=(0,1))

# Extract m=1 azimuthal Fourier mode
f_m1 = f_fourier[:, 1, :]
f_m0 = f_fourier[:, 0, :]  # DC component

# Asymmetry amplitude (normalized)
A_phi_array = abs(f_m1) / abs(f_m0)

# Average over pitch angles 30°-60° (away from loss cone)
mask = (alpha_bins > 30) & (alpha_bins < 60)
A_phi_mean = mean(A_phi_array[mask])
sigma_A_phi = std(A_phi_array[mask]) / sqrt(sum(mask))

# Significance test
z_score = (A_phi_mean - 0.06) / sigma_A_phi

# Verdict
if z_score > 5:
    result = "DISCOVERY"
elif z_score > 3:
    result = "EVIDENCE"
else:
    result = "NULL"
