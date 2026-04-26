import numpy as np

def load_tau(filename="Data/2Data_neuron.txt"):
    t = np.loadtxt(filename)
    return np.diff(t)