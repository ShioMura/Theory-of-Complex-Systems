import numpy as np

# Load data
data = np.loadtxt("Data/US_SupremeCourt_n9_N895.txt", dtype=str)

N = len(data)
n = len(data[0])
total_possible_states = 2 ** n
unique_states = np.unique(data)
Nmax = len(unique_states)

print("n =", n)
print("2^n =", total_possible_states)
print("N =", N)
print("Nmax =", Nmax)