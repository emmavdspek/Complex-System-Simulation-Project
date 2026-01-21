"""
function for analyzing the results
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import label
from scipy.ndimage import sum
from scipy.optimize import curve_fit


def show_grids(grids: list, iterations=[0, 200]):
    """Plots the supplied list of grids next to each other."""
    if len(grids) > 10:  # do not allow grid lists longer than 10
        print("The amount of grids supplied is to many, only plotting the first 10.")
        grids = grids[:10]

    fig, axs = plt.subplots(
        1, len(grids), figsize=(len(grids) * 2, 3), sharey=True, constrained_layout=True
    )
    for a in range(len(axs)):
        ax = axs[a]
        ax.imshow(grids[a], cmap="YlGn")
        ax.set_xlabel("x")
        ax.set_title(f"Iteration {iterations[a]}")
    axs[0].set_ylabel("y")
    plt.show()
    return


def powerlaw_exp(sizes, A=1e6, beta=1, s_char=1e6):
    return A * (sizes ** (-beta)) * np.exp(-sizes / s_char)


def cluster_size_distribution(grids: list, plot=True):
    size = np.shape(grids[0])[0]  # grid size
    size_occurrences = np.zeros(
        size**2
    )  # 1D array where indices correspond to cluster sizes
    N_grids = len(grids)
    for grid in grids:
        lw, num = label(grid)
        clust_sizes = sum(grid, lw, index=arange(lw.max() + 1))
        for size in clust_sizes:
            size_occurrences[int(size)] += 1
    size_occurrences /= N_grids  # normalize by the number of grids
    size_occurrences = np.trim_zeros(size_occurrences, "b")
    sizes = np.arange(1, len(size_occurrences) + 1)

    # fit to a general powerlaw with exponential cut-off
    popt, pcov = curve_fit(
        powerlaw_exp,
        sizes,
        size_occurrences,
        p0=[1.0e4, 1.5, 1.0e3],
        # bounds=(0.01, np.array([1.0e6, 1.0e2, 1.0e6])),
    )

    if plot:
        plt.plot(size_occurrences, "o", color="black")
        plt.plot(
            sizes,
            powerlaw_exp(sizes, *np.array([1.0e4, 1.5, 1.0e2])),
            "r-",
            label=r"fit: A=%5.3f, $\beta$=%5.3f, $s_c$=%5.3f" % tuple(popt),
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Cluster size")
        plt.ylabel("Mean occurrence")
        plt.legend()
        plt.show()
    return size_occurrences
