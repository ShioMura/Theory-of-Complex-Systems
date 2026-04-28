import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("Data/US_SupremeCourt_n9_N895.txt", dtype=str)
data = np.array([[int(x) for x in row] for row in data])

# Convert 0 -> -1, 1 -> +1
spins = 2 * data - 1

# Compute magnetisation and pair averages
mean_spin = np.mean(spins, axis=0)
corr = (spins.T @ spins) / spins.shape[0]

# Reorder judges from most liberal to most conservative
order = np.argsort(mean_spin)

mean_reordered = mean_spin[order]
corr_reordered = corr[np.ix_(order, order)]

print("Original magnetisations:")
print(mean_spin)

print("\nReordering index:")
print(order)

print("\nReordered magnetisations:")
print(mean_reordered)

# Plot reordered magnetisations
plt.bar(range(1, 10), mean_reordered)
plt.xlabel("Judge index after reordering")
plt.ylabel(r"$\langle s_i \rangle_D$")
plt.title("Reordered magnetisation")
plt.grid(True)

plt.savefig("q6_3_reordered_magnetisation.png", dpi=300)
plt.show()

# Plot reordered correlation matrix
plt.imshow(corr_reordered, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label=r"$\langle s_i s_j \rangle_D$")
plt.title("Reordered correlation matrix")

plt.savefig("q6_3_reordered_correlation_matrix.png", dpi=300)
plt.show()