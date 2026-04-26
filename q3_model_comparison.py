import numpy as np
import matplotlib.pyplot as plt

# Load data
t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

# Parameters from Q1 and Q2
tau0 = 0
lambda_est = 0.0960427443569115

# Model distribution
tau_vals = np.linspace(0, max(tau), 500)

P_model = np.where(
    tau_vals >= tau0,
    lambda_est * np.exp(-lambda_est * (tau_vals - tau0)),
    0
)

# Plot data and model
plt.hist(tau, bins=50, density=True, alpha=0.5, label="Data")
plt.plot(tau_vals, P_model, label=f"Model: λ = {lambda_est:.3f}, τ₀ = {tau0}")

plt.xlabel("Inter-spike interval τ (ms)")
plt.ylabel("P(τ)")
plt.title("Model vs Data")
plt.legend()
plt.grid(True)

plt.savefig("q3_model_comparison.png", dpi=300)
plt.show()