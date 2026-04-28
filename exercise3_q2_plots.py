import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Data/US_SupremeCourt_n9_N895.txt", dtype=str)
data = np.array([[int(x) for x in row] for row in data])
spins = 2 * data - 1

mean_spin = np.mean(spins, axis=0)
corr = (spins.T @ spins) / spins.shape[0]

# Reorder from most liberal to most conservative
order = np.argsort(mean_spin)

mean_reordered = mean_spin[order]
corr_reordered = corr[np.ix_(order, order)]

# Reordered magnetisation
plt.bar(range(1, 10), mean_reordered)
plt.xlabel("Judge index after reordering")
plt.ylabel(r"$\langle s_i \rangle_D$")
plt.title("Reordered magnetisation")
plt.grid(True)
plt.savefig("q6_3_reordered_magnetisation.png", dpi=300)
plt.show()

# Reordered correlation matrix
plt.imshow(corr_reordered, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label=r"$\langle s_i s_j \rangle_D$")
plt.title("Reordered correlation matrix")
plt.savefig("q6_3_reordered_correlation_matrix.png", dpi=300)
plt.show()