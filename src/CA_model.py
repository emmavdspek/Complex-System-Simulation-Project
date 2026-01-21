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
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
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
def compute_weight_matrix(M, k, dmin=1):
    """
    Precompute the distance-based weights for the neighborhood.
    Returns a (2M+1) x (2M+1) matrix with weights (dmin/dist)**k,
    and zero at the center (a cell doesn't count itself).
    """
    size = 2 * M + 1
    weights = np.zeros((size, size))
    
    for dx in range(-M, M + 1):
        for dy in range(-M, M + 1):
            dist = np.sqrt(dx**2 + dy**2)
            if 0 < dist <= M:
                weights[dx + M, dy + M] = (dmin / dist) ** k
    
    return weights

@jit(nopython=False)
def compute_local_density(grid, weights):
    """
    Compute the local weighted tree density for every cell simultaneously.
    
    For each cell, this calculates:
        rho = sum(weights * neighbor_states) / sum(weights)
    
    Using convolution, this becomes a single operation over the whole grid.
    """
    # Numerator: weighted sum of occupied neighbors
    weighted_neighbors = convolve(
        grid.astype(float), 
        weights, 
        mode='wrap'  # periodic boundary conditions
    )
    
    # Denominator: sum of all weights (same for every cell with periodic BC)
    total_weight = np.sum(weights)
    
    # Local density at each cell
    rho = weighted_neighbors / total_weight
    
    return rho

@jit(nopython=False)
def update_Scanlon2007_fast(grid, cells_to_update, true_frac, weights):
    """
    Optimized Scanlon update rule using precomputed weights and convolution.
    """
    size = grid.shape[0]
    
    # Compute local density for ALL cells at once (the expensive part, now fast)
    rho = compute_local_density(grid, weights)
    
    # Global vegetation fraction
    frac_occ = np.mean(grid)
    
    # Avoid division by zero
    if frac_occ == 0 or frac_occ == 1:
        return grid
    
    # Update only the selected cells
    for i, j in cells_to_update:
        rho_local = rho[i, j]
        random_nr = np.random.random()
        
        if grid[i, j] == 0:  # currently empty
            prob_flip = rho_local + (true_frac - frac_occ) / (1 - frac_occ)
            prob_flip = np.clip(prob_flip, 0, 1)  # clamp to valid range
            if random_nr < prob_flip:
                grid[i, j] = 1
                
        else:  # currently occupied
            prob_flip = (1 - rho_local) + (frac_occ - true_frac) / frac_occ
            prob_flip = np.clip(prob_flip, 0, 1)
            if random_nr < prob_flip:
                grid[i, j] = 0
    
    return grid


def evolve_CA_fast(
    size=500,
    p=0.5,
    update_rule=update_Scanlon2007_fast,
    true_frac=0.1,
    k=3.0,
    M=25,
    f_update=0.2,
    N_steps=200,
    skip=100,
    seed=0,
):
    """
    Optimized evolution function.
    """
    np.random.seed(seed)
    grid = initialize_CA(p, size)
    grids = []
    
    # Precompute weights once (not every iteration!)
    weights = compute_weight_matrix(M, k)
    
    # Fixed: use size**2 for total number of cells
    N_update = int(f_update * size**2)
    
    for n in range(N_steps):
        # Randomly select cells to update
        cells_to_update = np.column_stack([
            np.random.randint(0, size, N_update),
            np.random.randint(0, size, N_update)
        ])
        
        grid = update_rule(grid, cells_to_update, true_frac, weights)
        
        if n >= skip:
            grids.append(grid.copy())
    
    return grids


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
        if update_rule == update_Scanlon2007:
            update_args = [cells_to_update, true_frac, k, M]
        elif update_rule == update_basic:
            update_args = [cells_to_update]
        grid = update_rule(grid, *update_args)
        # if we are beyond equilibration, save the grid to the list to return
        if n >= skip:
            grids.append(grid.copy())

    return grids
