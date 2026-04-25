import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

plt.hist(tau, bins=50, density=True)
plt.xlabel("Inter-spike interval τ (ms)")
plt.ylabel("P(τ)")
plt.title("Distribution of inter-spike intervals")
plt.show()