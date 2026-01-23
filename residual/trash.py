"""
Docstring for CA_Scanlon.py
2D cellular automaton with update rule according to Scanlon et al (2007).
Implements:
- grid initialization
- Scanlon update rule (still in progress)
- one full iteration of update steps
- plotting the grids
"""

import numpy as np
import matplotlib.pyplot as plt

from numba import jit
import time


def initialize_CA(p=0.5, size=500):
    """Initializes a (pseudo-)randomly generated grid of occupied (t) and unoccupied (o) sites,
    based on a probability of occupation p. Returns the grid."""
    grid = np.random.choice(np.array([0, 1]), size=(size, size), p=np.array([1 - p, p]))
    return grid


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


@jit(nopython=False)
def update_Scanlon2007(
    grid, random_sel, true_frac=0.3, k=3.0, M=25, dmin=1, BC="periodic"
):
    """Update algorithm according to the model presented in Scanlon et al. (2007).
    Input parameters: see evolve_CA.
    One additional parameter:
    - BC:   either "periodic" or "aperiodic", sets the boundary conditions for evaluating
            the neighborhood of each cell.
    Returns the CA grid after one iteration of updating."""
    current_grid = grid.copy()
    size = np.shape(current_grid)[0]

    # loop through all the randomly selected cells
    for i, j in random_sel:
        cum_distr = 0  # denominator
        numerator = 0  # numerator

        # loop through all neighbors and add contributions to the num./denom. of rho
        for dx in range(-M, M + 1):
            for dy in range(-M, M + 1):
                dist = np.sqrt((dx) ** 2 + (dy) ** 2)

                # CLOSED BOUNDARY CONDITIONS
                if BC == "aperiodic":
                    # if the neighbor is within bounds and within range, add contributions
                    if (
                        (0 <= (i + dx) < size)
                        and (0 <= (j + dy) < size)
                        and (0 < dist <= M)
                    ):
                        cum_distr += (dmin / dist) ** k
                        numerator += (dmin / dist) ** k * current_grid[i + dx, j + dy]

                # PERIODIC BOUNDARY CONDITIONS
                elif BC == "periodic":
                    # if the neighbor is within bounds and within range, add contributions
                    if 0 < dist <= M:
                        cum_distr += (dmin / dist) ** k
                        numerator += (dmin / dist) ** k * current_grid[
                            (i + dx) % size, (j + dy) % size
                        ]

        # define parameters necessary for the update rule
        rho = numerator / cum_distr  # rho_t in the paper
        frac_occ = np.sum(current_grid) / (size**2)  # fraction of vegetation
        random_nr = np.random.random()  # number to use for updating

        # if the cell is currently unoccupied, update according to the P(o->t) rule
        if current_grid[i, j] == 0:
            prob_flip = rho + (true_frac - frac_occ) / (1 - frac_occ)
            # if the probability exceeds that of a randomly generated number, flip the state to occupied
            if random_nr < prob_flip:
                grid[i, j] = 1
        # if the cell is currently occupied, update according to the P(t->o) rule
        elif current_grid[i, j] == 1:
            prob_flip = (1 - rho) + (frac_occ - true_frac) / (frac_occ)
            # if the probability exceeds that of a randomly generated number, flip the state to unoccupied
            if random_nr < prob_flip:
                grid[i, j] = 0

    return grid


def evolve_CA(
    size=500,
    p=0.5,
    true_frac=0.1,
    k=3.0,
    M=25,
    f_update=0.2,
    N_steps=200,
    skip=100,
    seed=0,
):
    """
    Evolves a 2D CA one timestep at a time, according to the update rule
    described in Scanlon et al. (2007). Input parameters are:
     - size:        the width and height of the CA grid;
     - p:           the initial fraction of occupied (vegetated) sites;
     - true_frac:   the natural fraction of occupied sites (governed by rainfall);
     - k:           parameter in the Pareto-distribution setting strength of local interactions;
     - M:           radius defining size of neighborhood for local interactions;
     - f_update:    fraction of sites to (possibly) update at each time-step;
     - N_steps:     number of iterations updating the grid;
     - skip:        number of starting iterations to ignore (considered equilibration);
     - seed:        for the numpy pseudo-random number generator.
    Returns an array of grid configurations of the CA.
    """
    np.random.seed(seed)
    grid = initialize_CA(p, size)
    grids = []
    N_update = int(f_update * size)  # number of cells to update at each step

    start = time.time()
    for n in range(N_steps):
        # randomly select a fraction of the sites to update
        random_sel = np.reshape(np.random.choice(size, N_update * 2), (N_update, 2))
        # update the grid one step
        grid = update_Scanlon2007(grid, random_sel, true_frac, k, M)
        # if we are beyond equilibration, save the grid to the list to return
        if n >= skip:
            grids.append(grid.copy())
    end = time.time()
    print(f"It took {end-start} seconds to compute all the data.")

    return grids


# FIRST ATTEMPT AT SOME ANALYSIS OF THE DATA, SHOULD BE MOVED TO ANALYSIS.PY EVENTUALLY

from pylab import *
from scipy.ndimage import label
from scipy.ndimage import sum
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import powerlaw


def powerlaw_exp(sizes, A=1e4, beta=1.5, s_char=1e2):
    """General function of a power law with an exponential cutoff.
    The arguments represent:
    - A:        height of the curve in loglog-space;
    - beta:     scaling exponent, slope of the straight part of the curve in loglog-space;
    - s_char:   characteristic size, point from which powerlaw is cut off.
    Outputs the y-coords correponding to the x-coords 'sizes'."""
    return A * (sizes ** (-beta)) * np.exp(-(sizes) / s_char)


def loglog_powerlaw_exp(log_sizes, A=1e4, beta=1.5, s_char=1e2):
    """General function of a power law with an exponential cutoff.
    The arguments represent:
    - A:        height of the curve in loglog-space;
    - beta:     scaling exponent, slope of the straight part of the curve in loglog-space;
    - s_char:   characteristic size, point from which powerlaw is cut off.
    Outputs the y-coords correponding to the x-coords 'sizes'."""
    return np.log(A) - beta * log_sizes + np.exp(-(log_sizes)) / s_char


def cluster_size_distribution(grids: list, plot=True):
    """Computes the cluster size distribution of a given list of grids.
    This list can consist of 1 dataset (a single evolution of a CA), or multiple
    sets can be combined into one list. The arguments are:
    - grids:    1D list of 2D arrays (grids) corresponding to iterations of am evolved CA;
    - plot:     boolean, if True the results are plotted as well, if False not.
    Returns a 1D array of mean occurrences of cluster sizes throughout all provided grids.
    """
    size = np.shape(grids[0])[0]  # grid size
    # size_occurrences = np.zeros(
    #     size**2
    # )  # 1D array where indices correspond to cluster sizes
    N_grids = len(grids)
    size_list = []
    for grid in grids:
        lw, num = label(grid)
        clust_sizes = sum(grid, lw, index=arange(lw.max() + 1))
        size_list += list(clust_sizes[clust_sizes != 0])
        # for size in clust_sizes:
        #     size_occurrences[int(size)] += 1
    # print(size_list)
    # size_occurrences = np.trim_zeros(size_occurrences, "b")
    # size_occurrences /= N_grids  # divide by the number of grids
    # size_occurrences /= np.sum(size_occurrences)  # normalize to get probabilities
    # sizes = np.arange(1, len(size_occurrences) + 1)

    # # fit to a general powerlaw with exponential cut-off
    # popt, pcov = curve_fit(
    #     powerlaw_exp,
    #     sizes,
    #     size_occurrences,
    #     p0=[1.0e4, 1.5, 1.0e3],
    #     bounds=([1e2, 0.1, 1e2], [1.0e6, 1.0e2, 1.0e5]),
    # )
    # log_appr_size_occurrences = np.where(
    #     size_occurrences == 0.0, 1e-5, size_occurrences
    # )
    # print(log_appr_size_occurrences)
    # fit to a general powerlaw with exponential cut-off
    # popt, pcov = curve_fit(
    #     loglog_powerlaw_exp,
    #     np.log(sizes),
    #     np.log(log_appr_size_occurrences),
    #     p0=[1.0e4, 1.5, 1.0e3],
    # bounds=([0.1, 0.01, 0.1], [1.0e5, 1.0e2, 1.0e5]),
    # )
    # print(popt)

    fit = powerlaw.Fit(size_list, discrete=True)
    beta = fit.truncated_power_law.alpha
    print("Beta: ", beta)

    if plot:
        # fig =
        # fig.plot(size_occurrences, "o", color="black")
        fig = powerlaw.plot_pdf(
            size_list,
            linear_bins=True,
            color="black",
            marker="o",
            markersize=3,
            linewidth=0,
        )
        fit.truncated_power_law.plot_pdf(ax=fig, color="red")
        # plt.plot(
        #     sizes,
        #     # powerlaw_exp(sizes, *popt),
        #     "r-",
        #     label=r"fit: A=%5.3f, $\beta$=%5.3f, $s_c$=%5.3f" % tuple(popt),
        # )
        # fig.set_xscale("log")
        # fig.set_yscale("log")
        fig.set_xlabel("Cluster size")
        fig.set_ylabel("Mean occurrence")
        # plt.legend()
        # plt.show()
    return size_list


# Test the functions
grids = evolve_CA(size=300, p=0.5, true_frac=0.3, k=3, M=20, N_steps=150, skip=0)
# show_grids([grids[0], grids[-1]], [0, 200])
cluster_size_distr = cluster_size_distribution(grids[100:])
