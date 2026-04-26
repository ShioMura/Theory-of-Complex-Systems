import numpy as np
import matplotlib.pyplot as plt

# Load data
t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

# Estimate tau0 (from observation)
tau0 = 0

# Plot
plt.hist(tau, bins='auto', density=True)
plt.axvline(tau0, color='r', linestyle='--', label=f'τ₀ ≈ {tau0}')

plt.xlabel("Inter-spike interval τ (ms)")
plt.ylabel("P(τ)")
plt.title("Distribution of inter-spike intervals")
plt.legend()
plt.grid(True)

plt.savefig("q1_histogram.png", dpi=300)
plt.show()