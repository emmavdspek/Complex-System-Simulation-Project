"""
functions for analyzing the results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# modules necessary for finding the cluster size distribution:
# (source: https://stackoverflow.com/questions/25664682/how-to-find-cluster-sizes-in-2d-numpy-array)
from pylab import arange
from scipy.ndimage import label
from scipy.ndimage import sum
import powerlaw


def show_grids(grids: list, iterations=[0, 200]):
    """
    Plots the supplied list of grids next to each other, to show evolution.
    Takes as arguments:
    - grids:        1D list of 2D arrays (grids) corresponding to iterations in the evolution of a CA.
    - iterations:   frame-numbers of the provided grids. Typically the first and last of a full evolution.
    """
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


def cluster_sizes(grids: list):
    """
    Computes the cluster size distribution of a given list of grids.
    This list can consist of 1 dataset (a single evolution of a CA), or multiple
    sets can be combined into one list.
    Argument:
    - grids:    1D list of 2D arrays (grids) corresponding to iterations of an evolved CA;
    Returns:
    - fit:      fit object from powerlaw module;
    - fig:      figure containing a plot of the data and the fit, can possibly be used to save later.
    """
    size_list = []
    for grid in grids:
        lw, num = label(grid)
        clust_sizes = sum(grid, lw, index=arange(lw.max() + 1))
        size_list += list(clust_sizes[clust_sizes != 0])

    fit = powerlaw.Fit(
        size_list,
        xmin=1,
        xmax=np.max(size_list) * 0.6,
        discrete=True,
    )

    return size_list, fit


def plot_cluster_size_distr(size_lists: list, fits: list):
    """
    Plots the complementary cumulative (ccdf) cluster size distribution (P(S>=s)).
    Arguments (returned by cluster_sizes()):
    - size_lists:    list of 1D arrays containing the sizes of all the clusters in one dataset;
    - fits:          list fit objects per dataset generated from the powerlaw library.
    """
    N_sets = len(size_lists)

    fig, ax = plt.subplots(figsize=(6, 6))
    # list of colors for all the plots
    color_list = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))[::5]

    for i in range(N_sets):
        fits[i].plot_pdf(
            size_lists[i],
            linear_bins=False,
            color=color_list[i],
            marker="o",
            markersize=1.5,
            linewidth=0,
        )
        alpha = fits[i].truncated_power_law.alpha  # scaling exponent of power law
        fits[i].truncated_power_law.plot_pdf(
            ax=ax,
            color=color_list[i],
            linewidth=1,
            label=r"$\alpha=$" + str(np.round(alpha, 2)),
        )
    ax.set_ylim(1e-10, 1e1)
    ax.set_xlabel(r"Cluster size $s$")
    # ax.set_ylabel(r"P($S\geq s$)")
    ax.set_ylabel(r"P($S=s$)")
    ax.legend(fontsize="small")

    return
