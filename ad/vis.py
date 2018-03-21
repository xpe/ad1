import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from ad import gen


def plot(x, alpha=0.2):
    """Plot `n` points from the synthetic distribution."""
    df = pd.DataFrame(data=x)
    pd.plotting.scatter_matrix(df, alpha=alpha, figsize=(8, 8), diagonal='kde')
    plt.show(block=False)


def main(n=5000, d=10, b=10, g=2, alpha=0.03, seed=None):
    if seed:
        np.random.seed(seed)
    x = gen.observations(n, d, b, g)
    plot(x, alpha=alpha)


if __name__ == '__main__':
    main()
    input("Press enter to close the plot and exit the program.")
