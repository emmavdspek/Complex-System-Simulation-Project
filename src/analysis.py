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

from matplotlib.animation import FuncAnimation, PillowWriter


def animate_multiple_phis(grids_by_phi, phi_values, dpi=100):
    """
    Create a side-by-side animation of CA evolutions for different phi values (weight of the local rule)
    
    grids_by_phi : dict {phi : list_of_grids}
    phi_values   : list of phi values to animate
    """

    # Number of frames = min length of all grid sequences
    T = min(len(grids_by_phi[phi]) for phi in phi_values)

    # Set up figure with 1 row and len(phi_values) columns
    fig, axes = plt.subplots(1, len(phi_values), figsize=(4*len(phi_values), 4))

    # If only one phi, axes is not a list
    if len(phi_values) == 1:
        axes = [axes]

    # Initialize images
    ims = []
    for ax, phi in zip(axes, phi_values):
        im = ax.imshow(grids_by_phi[phi][0], cmap="YlGn", interpolation='nearest')
        ax.set_title(f"phi = {phi}", fontsize=14)
        ax.axis("off")
        ims.append(im)

    # Animation function
    def animate(t):
        for im, ax, phi in zip(ims, axes, phi_values):
            im.set_data(grids_by_phi[phi][t])
        return ims

    # Create animation
    anim = FuncAnimation(
        fig,
        animate,
        frames=T,
        interval=100,
        blit=False
    )

    print("Saving animation...")
    anim.save("multi_phi_animation.gif", writer=PillowWriter(fps=10), dpi=dpi)
    print("Saved as multi_phi_animation.gif")


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
        size_list += list(clust_sizes[clust_sizes >= 10])

    fit = powerlaw.Fit(
        size_list,
        xmin=10,
        xmax=np.max(size_list),
        discrete=True,
    )

    return size_list, fit


def cluster_sizes_safe(grids):
    """
    Docstring for cluster_sizes_safe
    
    :param grids: same funciton as above, but safer for grids with small clusters
    Used for the optimized locabl/global code  
    Returns (size_list, fit) or (size_list, None) if no valid clusters
    """
    size_list = []

    for grid in grids:
        lw, num = label(grid)
        if num == 0:
            continue
        clust_sizes = sum(grid, lw, index=np.arange(lw.max() + 1))
        size_list += list(clust_sizes[clust_sizes >= 10])

    if len(size_list) == 0:
        return size_list, None

    fit = powerlaw.Fit(
        size_list,
        xmin=10,
        xmax=np.max(size_list),
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

    fig, ax = plt.subplots(figsize=(7, 5))
    # list of colors for all the plots
    color_list = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))[::5]

    for i in range(N_sets):
        fits[i].plot_ccdf(
            size_lists[i],
            color=color_list[i],
            marker="o",
            markersize=1.5,
            linewidth=0,
        )
        alpha = fits[i].truncated_power_law.alpha  # scaling exponent of power law
        fits[i].truncated_power_law.plot_ccdf(
            ax=ax,
            color=color_list[i],
            linewidth=1,
            label=r"$\alpha=$" + str(np.round(alpha, 2)),
        )
    ax.set_ylim(1e-6, 1e1)
    ax.set_xlabel(r"Cluster size $s$")
    ax.set_ylabel(r"P($S\geq s$)")
    ax.legend(ncol=3, fontsize="small")

    return fig


def has_vertical_percolation(grid): 
    """
    Docstring for has_vertical_percolation
    
    :param grid: returns True if there exists a connected
    vegetation cluster that touches both the top row and bottom row 
    """
    lw, num = label(grid) #label connected components, lw is the same size as the grid and contains 0 if cell emtpy, or other for vegation cluster; num is the number of clusters

    top_labels = set(lw[0, :]) #take all the labels in the top row 
    bottom_labels = set(lw[-1, :]) #"" in the bottom row

    #if the same label appears in both sets, we have percolation
    common = top_labels.intersection(bottom_labels)

    #label 0 = background, so ignore it 
    return any(lbl != 0for lbl in common)


def has_horizontal_percolation(grid): 
    """
    Docstring for has_horizontal_percolation
    
    :param grid: returns True if there exists a connected
    vegetation cluster that touches the left to the right 
    """
    lw, num = label(grid) #label connected components, lw is the same size as the grid and contains 0 if cell emtpy, or other for vegation cluster; num is the number of clusters

    left_labels = set(lw[:, 0]) #take all the labels in the top row 
    right_labels = set(lw[:, -1]) #"" in the bottom row

    #if the same label appears in both sets, we have percolation
    common = left_labels.intersection(right_labels)

    #label 0 = background, so ignore it 
    return any(lbl != 0for lbl in common)
