import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

tau0 = 2  # same as Q2
lambda_est = 0.5  # replace with your fitted value

tau_vals = np.linspace(0, max(tau), 200)

P_model = np.where(tau_vals >= tau0,
                   lambda_est * np.exp(-lambda_est * (tau_vals - tau0)),
                   0)

plt.hist(tau, bins=50, density=True, alpha=0.5, label="Data")
plt.plot(tau_vals, P_model, 'r', label="Model")
plt.legend()
plt.xlabel("τ")
plt.ylabel("P(τ)")
plt.title("Model vs Data")
plt.show()