"""
functions for analyzing the results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# modules necessary for finding the cluster size distribution:
# (source: https://stackoverflow.com/questions/25664682/how-to-find-cluster-sizes-in-2d-numpy-array)
from pylab import arange
from scipy.ndimage import label
from scipy.ndimage import sum
import powerlaw

from matplotlib.animation import FuncAnimation, PillowWriter


def show_grids(grids: list, iterations=[0, 200]):
    """
    Plots the supplied list of grids next to each other, to show evolution.
    Takes as arguments:
    - grids:        1D list of 2D arrays (grids) corresponding to iterations in the evolution of a CA.
    - iterations:   frame-numbers of the provided grids. Typically the first and last of a full evolution.
    """
    if len(grids) > 10:  # do not allow grid lists longer than 10
        print("The number of grids supplied is too many, only plotting the first 10.")
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


def animate_grids(grids: list, dpi=100):
    """Create and save an animated GIF showing the temporal evolution of CA grids."""
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grids[0], cmap="YlGn", interpolation='nearest')
    ax.set_title('Iteration 0', fontsize=20)
    ax.axis('off')

    # Define animation function
    def animate(frame):
        i, grid = frame
        im.set_data(grid)
        ax.set_title(f'Iteration {i}', fontsize=20)
        return [im]

    # Set up helper function to display progress
    def progress_callback(current_frame, total_frames):
        """Shows saving progress"""
        if current_frame % 10 == 0:
            print(f'Saving frame {current_frame}/{total_frames} ...')

    # Create animation
    anim = FuncAnimation(fig, animate, enumerate(grids),
                         interval=50, save_count=len(grids))
    print("Saving animation ... (This can take a while depending on the dpi.)")
    filename = 'ca_simulation.gif'
    anim.save(filename, writer=PillowWriter(fps=20), dpi=dpi,
              progress_callback=progress_callback)
    print(f"Saved successfully as '{filename}'")


def animate_grids(grids: list, dpi=100):
    """Create and save an animated GIF showing the temporal evolution of CA grids."""
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grids[0], cmap="YlGn", interpolation='nearest')
    ax.set_title('Iteration 0', fontsize=20)
    ax.axis('off')

    # Define animation function
    def animate(frame):
        i, grid = frame
        im.set_data(grid)
        ax.set_title(f'Iteration {i}', fontsize=20)
        return [im]

    # Set up helper function to display progress
    def progress_callback(current_frame, total_frames):
        """Shows saving progress"""
        if current_frame % 10 == 0:
            print(f'Saving frame {current_frame}/{total_frames} ...')

    # Create animation
    anim = FuncAnimation(fig, animate, enumerate(grids),
                         interval=50, save_count=len(grids))
    print("Saving animation ... (This can take a while depending on the dpi.)")
    filename = 'ca_simulation.gif'
    anim.save(filename, writer=PillowWriter(fps=20), dpi=dpi,
              progress_callback=progress_callback)
    print(f"Saved successfully as '{filename}'")


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

    fit = powerlaw.Fit(size_list, xmin=0, xmax=np.max(size_list), discrete=True)

    return size_list, fit


def plot_cluster_size_distr(size_list, fit):
    """
    Plots the complementary cumulative (ccdf) cluster size distribution.
    Arguments (returned by cluster_sizes()):
    - size_list:    1D array containing the sizes of all the clusters in all evaluated iterations;
    - fit:          fit object generated from the powerlaw library.
    """

    beta = fit.truncated_power_law.alpha  # scaling exponent of power law

    fig = powerlaw.plot_ccdf(
        size_list,
        color="black",
        marker="o",
        markersize=4,
        linewidth=0,
    )
    fit.truncated_power_law.plot_ccdf(
        ax=fig, color="red", label=r"$\beta=$" + str(np.round(beta, 2))
    )
    fig.set_xlabel(r"Cluster size $s$")
    fig.set_ylabel(r"P($S\geq s$)")
    fig.legend()
    plt.show()

    return
