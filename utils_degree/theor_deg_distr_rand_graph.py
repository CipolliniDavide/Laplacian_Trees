import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


# def degree_distribution(n, m):
#     p = 2 * m / (n * (n - 1))
#     degrees = range(n)
#     probabilities = [comb(n-1, k) * (p**k) * ((1-p)**(n-1-k)) for k in degrees]
#     return degrees, probabilities

def rand_degree_distribution(n, m):
    p = 2 * m / (n * (n - 1))
    if p <= 0 or p >= 1:
        raise ValueError("Invalid probability value. Ensure 0 < p < 1.")
    degrees = range(n)
    probabilities = [comb(n-1, k) * (p**k) * ((1-p)**(n-1-k)) for k in degrees]

    return degrees, probabilities

def plot_degree_distribution(n, m, ax=None, loglog=False, show=False, set_line=False):
    degrees, probabilities = rand_degree_distribution(n, m)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title('Degree Distribution of Erdos-Renyi Random Graph')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Probability')
    if set_line:
        ax.plot(degrees, probabilities, color='orange', linestyle='--', linewidth=2, label='Erdos-Renyi Random')
    else:
        ax.bar(degrees, probabilities, color='orange', linestyle='--', linewidth=2, label='Erdos-Renyi Random')
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if show:
        plt.show()

if __name__ == '__main__':
    # if the network is very sparse there are computational issues
    n = 100  # Number of nodes
    m = 200  # Number of edges
    plot_degree_distribution(n, m, loglog=True, set_line=True, show=True)

