import numpy as np
import matplotlib.pyplot as plt

tau0 = 0
lambda_est = 0.0960427443569115

N = 1000

tau_sim = tau0 + np.random.exponential(1 / lambda_est, size=N)
t_sim = np.cumsum(tau_sim)

plt.hist(tau_sim, bins=50, density=True)

plt.xlabel("Inter-spike interval τ (ms)")
plt.ylabel("P(τ)")
plt.title("Simulated inter-spike intervals")
plt.grid(True)

plt.savefig("q4_simulation.png", dpi=300)
plt.show()

np.savetxt("q4_simulated_spike_times.txt", t_sim)