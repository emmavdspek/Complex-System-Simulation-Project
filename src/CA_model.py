"""
Docstring for CA_model.py
Some 2D cellular automaton update rules, and logic
Implements:
- grid initialization
- several update rules:
    -> Basic
    -> Neutral (still in progress)
    -> Scanlon et al. update rule
- one full evolution of iterations, according to update rule
"""

# Import necessary modules
import numpy as np
from numba import jit


def initialize_CA(p=0.5, size=500):
    """Initializes a (pseudo-)randomly generated grid of occupied (t) and unoccupied (o) sites,
    based on a probability of occupation p. Returns the grid."""
    grid = np.random.choice(np.array([0, 1]), size=(size, size), p=np.array([1 - p, p]))
    return grid


@jit(nopython=False)
def update_basic(grid, cells_to_update):
    """
    Function which decides what happens to a single cell
    :param state: empty or a tree (1)
    :param n_neighbors: if a cell has 3 or 4 neighbors, it becomes empty, otherwise a tree
    """
    nx, ny = grid.shape
    current_grid = grid.copy()

    for x, y in cells_to_update:
        up = current_grid[
            (x - 1) % nx, y
        ]  # the % nx and % ny make the grid wrap around (periodic boundaries)
        down = current_grid[(x + 1) % nx, y]
        left = current_grid[x, (y - 1) % ny]
        right = current_grid[x, (y + 1) % ny]
        n_neighbors = up + down + left + right

        if n_neighbors in (3, 4):
            grid[x, y] = 0
        else:
            grid[x, y] = 1

    return grid


@jit(nopython=False)
def update_Scanlon2007(
    grid, cells_to_update, true_frac=0.3, k=3.0, M=25, dmin=1, BC="periodic"
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
    for i, j in cells_to_update:
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

        # avoid dividing by zero later on
        assert frac_occ != 0 and frac_occ != 1

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
    update_rule=update_Scanlon2007,
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
    N_update = int(f_update * size**2)  # number of cells to update at each step

    for n in range(N_steps):
        # randomly select a fraction of the sites to update
        cells_to_update = np.reshape(
            np.random.choice(size, N_update * 2), (N_update, 2)
        )
        # update the grid one step
        if update_rule.__name__ == 'update_Scanlon2007':
            update_args = [cells_to_update, true_frac, k, M]
        elif update_rule.__name__ == 'update_basic':
            update_args = [cells_to_update]
        else:
            raise ValueError(f"Unknown update rule: {update_rule.__name__}")
        grid = update_rule(grid, *update_args)
        # if we are beyond equilibration, save the grid to the list to return
        if n >= skip:
            grids.append(grid.copy())

    return grids
