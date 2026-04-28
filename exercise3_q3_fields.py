import numpy as np

# Load data
data = np.loadtxt("Data/US_SupremeCourt_n9_N895.txt", dtype=str)
data = np.array([[int(x) for x in row] for row in data])
spins = 2 * data - 1

# Compute magnetisation
mean_spin = np.array([
    0.37653631, -0.33184358, 0.21117318, 0.40558659,
    0.22681564, -0.16201117, 0.45027933, -0.17094972, -0.09944134
])

# Compute fields
h = np.arctanh(mean_spin)

print("Magnetisations <s_i>_D:")
print(mean_spin)

print("\nInferred fields h_i:")
print(h)