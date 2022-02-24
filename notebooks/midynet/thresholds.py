import midynet
import numpy as np
from scipy.optimize import bisect

def dominant_eigenvalue(cfg):
    eigen_values = []
    N, E = cfg.size, cfg.edge_count.state
    for h in cfg.heterogeneity:
        degrees = midynet.util.degree_sequences.nbinom_degreeseq(N, E, h)
        eigen_value = max(np.sqrt(np.max(degrees)), (np.mean(degrees**2) - np.mean(degrees)) / np.mean(degrees))
        eigen_values.append(eigen_value)
    return np.array(eigen_values)

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def wc_thresholds(a=4, b=0.5, m=1):
    def h(x, c):
        return -b * x + (1 - x) * sigmoid(a * (c * x - m))

    def f1_to_solve(c):
        q = c * a
        x = (1 + np.sqrt(1 - 4 * (1 + b)/ q)) / (2 * (1 + b))
        return h(x, c)

    def f2_to_solve(c):
        q = c * a
        x = (1 - np.sqrt(1 - 4 * (1 + b)/ q)) / (2 * (1 + b))
        return h(x, c)
    c1, c2 = None, None
    if np.sign(f1_to_solve(4 * (1 + b) / a)) != np.sign(f1_to_solve(100)):
        c1 = bisect(f1_to_solve, a=4 * (1 + b) / a, b=100)
    if np.sign(f2_to_solve(4 * (1 + b) / a)) != np.sign(f2_to_solve(100)):
        c2 = bisect(f2_to_solve, a=4 * (1 + b) / a, b=100)
    return c1, c2