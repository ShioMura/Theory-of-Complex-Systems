import numpy as np

# Load data
data = np.loadtxt("Data/US_SupremeCourt_n9_N895.txt", dtype=str)

# Convert each row like "101111111" into digits
data = np.array([[int(x) for x in row] for row in data])

# Convert 0 -> -1 and 1 -> +1
spins = 2 * data - 1

N, n = spins.shape

print("Number of cases N:", N)
print("Number of judges n:", n)

# Empirical magnetisations <s_i>_D
mean_spin = np.mean(spins, axis=0)

# Empirical pair averages <s_i s_j>_D
pair_average = (spins.T @ spins) / N

print("\nEmpirical magnetisations <s_i>_D:")
print(mean_spin)

print("\nEmpirical pair averages <s_i s_j>_D:")
print(pair_average)