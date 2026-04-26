import numpy as np

# Load data
t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

# Empirical firing rate
f_data = 1 / np.mean(tau)

# Parameters from Q1 and Q2
tau0 = 0
lambda_est = 0.0960427443569115

# Theoretical firing rate
f_theory = 1 / (tau0 + 1 / lambda_est)

print("Mean inter-spike interval:", np.mean(tau), "ms")
print("Empirical firing rate:", f_data, "spikes/ms")
print("Theoretical firing rate:", f_theory, "spikes/ms")