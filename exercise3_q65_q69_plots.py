import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ==========================
# Load data
# ==========================
data = np.loadtxt("Data/US_SupremeCourt_n9_N895.txt", dtype=str)
data = np.array([[int(x) for x in row] for row in data])
spins = 2 * data - 1

N, n = spins.shape

# Load fitted Ising parameters
h = np.loadtxt("Data/hi_ussc_unsorted.txt")
J_values = np.loadtxt("Data/Jij_ussc_unsorted.txt")

J = np.zeros((n, n))
k = 0
for i in range(n):
    for j in range(i + 1, n):
        J[i, j] = J_values[k]
        J[j, i] = J_values[k]
        k += 1

# ==========================
# Enumerate all 2^n states
# ==========================
states = np.array(list(product([-1, 1], repeat=n)))

def energy_log_weight(s):
    return np.dot(h, s) + sum(J[i, j] * s[i] * s[j] for i in range(n) for j in range(i + 1, n))

log_weights = np.array([energy_log_weight(s) for s in states])
weights = np.exp(log_weights)
Z = np.sum(weights)
p_model_all = weights / Z

# ==========================
# Empirical probabilities p_D(s)
# ==========================
state_to_index = {tuple(s): idx for idx, s in enumerate(states)}

counts = np.zeros(len(states))
for s in spins:
    counts[state_to_index[tuple(s)]] += 1

p_data_all = counts / N

observed = counts > 0
p_data_obs = p_data_all[observed]
p_model_obs = p_model_all[observed]

# ==========================
# Q6.5: p_D(s) vs p_g(s)
# ==========================
plt.figure()
plt.scatter(p_data_obs, p_model_obs)

min_val = min(p_data_obs.min(), p_model_obs.min())
max_val = max(p_data_obs.max(), p_model_obs.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Empirical probability $p_D(s)$")
plt.ylabel(r"Model probability $p_{\mathbf{g}}(s)$")
plt.title(r"Empirical vs model state probabilities")
plt.grid(True)

plt.savefig("q6_5_cross_validation.png", dpi=300)
plt.show()

# ==========================
# Q6.6: model observables
# ==========================
mean_data = np.mean(spins, axis=0)
corr_data = (spins.T @ spins) / N

mean_model = np.sum(p_model_all[:, None] * states, axis=0)

corr_model = np.zeros((n, n))
for a, s in zip(p_model_all, states):
    corr_model += a * np.outer(s, s)

# Magnetisation comparison
plt.figure()
plt.scatter(mean_data, mean_model)
min_val = min(mean_data.min(), mean_model.min())
max_val = max(mean_data.max(), mean_model.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")

plt.xlabel(r"Data $\langle s_i \rangle_D$")
plt.ylabel(r"Model $\langle s_i \rangle$")
plt.title("Magnetisation: data vs model")
plt.grid(True)

plt.savefig("q6_6_magnetisation_fit.png", dpi=300)
plt.show()

# Correlation comparison
data_corr_vals = []
model_corr_vals = []

for i in range(n):
    for j in range(i + 1, n):
        data_corr_vals.append(corr_data[i, j])
        model_corr_vals.append(corr_model[i, j])

data_corr_vals = np.array(data_corr_vals)
model_corr_vals = np.array(model_corr_vals)

plt.figure()
plt.scatter(data_corr_vals, model_corr_vals)
min_val = min(data_corr_vals.min(), model_corr_vals.min())
max_val = max(data_corr_vals.max(), model_corr_vals.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")

plt.xlabel(r"Data $\langle s_i s_j \rangle_D$")
plt.ylabel(r"Model $\langle s_i s_j \rangle$")
plt.title("Pair correlations: data vs model")
plt.grid(True)

plt.savefig("q6_6_correlation_fit.png", dpi=300)
plt.show()

# ==========================
# Q6.7-Q6.9: distributions P(k)
# ==========================

# Data distribution P_D(k)
k_data = np.sum((spins + 1) / 2, axis=1).astype(int)
P_D = np.array([np.mean(k_data == k) for k in range(n + 1)])

# Independent model P_I(k)
p_plus = (mean_data + 1) / 2

P_I = np.zeros(n + 1)
for s in states:
    k_val = int(np.sum((s + 1) / 2))
    prob = 1.0
    for i in range(n):
        if s[i] == 1:
            prob *= p_plus[i]
        else:
            prob *= (1 - p_plus[i])
    P_I[k_val] += prob

# Pairwise Ising model P_M(k)
P_M = np.zeros(n + 1)
for prob, s in zip(p_model_all, states):
    k_val = int(np.sum((s + 1) / 2))
    P_M[k_val] += prob

ks = np.arange(n + 1)

# Q6.8: P_D vs P_I
plt.figure()
plt.plot(ks, P_D, "o-", label=r"Data $P_D(k)$")
plt.plot(ks, P_I, "s-", label=r"Independent $P_I(k)$")

plt.xlabel(r"Number of conservative votes $k$")
plt.ylabel(r"Probability")
plt.title(r"Data vs independent model")
plt.xticks(ks)
plt.legend()
plt.grid(True)

plt.savefig("q6_8_data_vs_independent.png", dpi=300)
plt.show()

# Q6.9: P_D vs P_I vs P_M
plt.figure()
plt.plot(ks, P_D, "o-", label=r"Data $P_D(k)$")
plt.plot(ks, P_I, "s-", label=r"Independent $P_I(k)$")
plt.plot(ks, P_M, "^-", label=r"Pairwise Ising $P_M(k)$")

plt.xlabel(r"Number of conservative votes $k$")
plt.ylabel(r"Probability")
plt.title(r"Vote-count distribution comparison")
plt.xticks(ks)
plt.legend()
plt.grid(True)

plt.savefig("q6_9_all_models_comparison.png", dpi=300)
plt.show()

print("Saved figures:")
print("q6_5_cross_validation.png")
print("q6_6_magnetisation_fit.png")
print("q6_6_correlation_fit.png")
print("q6_8_data_vs_independent.png")
print("q6_9_all_models_comparison.png")