import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

t = np.loadtxt("Data/Data_neuron.txt")
tau = np.diff(t)

tau0 = 2  # adjust after Q1
tau_filtered = tau[tau > tau0]

hist, bins = np.histogram(tau_filtered, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.plot(bin_centers, hist, 'o')
plt.yscale('log')
plt.xlabel("τ")
plt.ylabel("log P(τ)")
plt.title("Exponential decay check")
plt.show()

mask = hist > 0
slope, intercept, *_ = linregress(bin_centers[mask], np.log(hist[mask]))
lambda_est = -slope

print("Estimated lambda:", lambda_est)