import numpy as np
import matplotlib.pyplot as plt

# Load fitted parameters
h = np.loadtxt("Data/hi_ussc_unsorted.txt")
J_values = np.loadtxt("Data/Jij_ussc_unsorted.txt")

n = len(h)

# Build Jij matrix from upper-triangular values
J = np.zeros((n, n))

k = 0
for i in range(n):
    for j in range(i + 1, n):
        J[i, j] = J_values[k]
        J[j, i] = J_values[k]
        k += 1
plt.imshow(J, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label=r"$J_{ij}$")
plt.title("Inferred coupling matrix")

plt.xlabel("Judge index")
plt.ylabel("Judge index")

plt.savefig("q6_4_inferred_couplings.png", dpi=300)
plt.show()

# Plot h_i
plt.bar(range(1, n + 1), h)
plt.xlabel("Judge index")
plt.ylabel(r"$h_i$")
plt.title("Inferred local fields")
plt.grid(True)

plt.savefig("q6_4_inferred_fields.png", dpi=300)
plt.show()

# Plot Jij matrix
plt.imshow(J, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label=r"$J_{ij}$")
plt.title("Inferred coupling matrix")

plt.savefig("q6_4_inferred_couplings.png", dpi=300)
plt.show()

print("h_i:")
print(h)

print("\nJij matrix:")
print(J)