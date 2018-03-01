import numpy as np


def uniformrand(a, b):
    """Uniform distribution from a to b """
    return np.random.uniform(low=a, high=b)


def multinomialrand(N, q):
    """Multinomial distribution formed by N trials from an underlying
        distribution p[k]"""

    nn = np.random.multinomial(n=N, pvals=q)

    return nn
