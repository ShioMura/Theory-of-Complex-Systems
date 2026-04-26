import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

# Refractory period estimated from Q1
tau0 = 0

# Use intervals after tau0
tau_filtered = tau[tau > tau0]

# Histogram
hist, bins = np.histogram(tau_filtered, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Keep only nonzero bins
mask = hist > 0

# Fit log(P(tau)) = a - lambda * tau
slope, intercept, r_value, p_value, std_err = linregress(
    bin_centers[mask],
    np.log(hist[mask])
)

lambda_est = -slope

print("Estimated lambda:", lambda_est)
print("Fit slope:", slope)
print("R squared:", r_value**2)

# Plot log histogram and fitted line
plt.plot(bin_centers[mask], hist[mask], "o", label="Data")
plt.plot(
    bin_centers[mask],
    np.exp(intercept + slope * bin_centers[mask]),
    label=f"Fit: λ = {lambda_est:.4f}"
)

plt.yscale("log")
plt.xlabel("Inter-spike interval τ (ms)")
plt.ylabel("P(τ)")
plt.title("Exponential decay check")
plt.legend()
plt.grid(True)

plt.savefig("q2_exponential_fit.png", dpi=300)
plt.show()