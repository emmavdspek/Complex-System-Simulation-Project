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


# =============================================================================
# Validation helpers
# =============================================================================

def _validate_probability(value, name):
    """Validate that a value is a valid probability in [0, 1]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_positive_int(value, name):
    """Validate that a value is a positive integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_grid(grid):
    """Validate that a grid is a 2D square binary array."""
    if not isinstance(grid, np.ndarray):
        raise TypeError(f"grid must be a numpy array, got {type(grid).__name__}")
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got {grid.ndim}D")
    if grid.shape[0] != grid.shape[1]:
        raise ValueError(f"grid must be square, got shape {grid.shape}")
    # Assert for internal invariant: values should be 0 or 1
    assert np.all((grid == 0) | (grid == 1)), "grid contains non-binary values"


# =============================================================================
# Initialization
# =============================================================================


def initialize_CA(p=0.5, size=500):
    """Initializes a (pseudo-)randomly generated grid of occupied (t) and unoccupied (o) sites,
    based on a probability of occupation p. Returns the grid."""
    _validate_probability(p, "p")
    _validate_positive_int(size, "size")

    grid = np.random.choice(np.array([0, 1]),
                            size=(size, size),
                            p=np.array([1 - p, p]))
    return grid

# =============================================================================
# Update rules
# =============================================================================

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

# =============================================================================
# Main evolution function
# =============================================================================

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
     - update_rule  the function to use as an update rule
     - true_frac:   the natural fraction of occupied sites (governed by rainfall);
     - k:           parameter in the Pareto-distribution setting strength of local interactions;
     - M:           radius defining size of neighborhood for local interactions;
     - f_update:    fraction of sites to (possibly) update at each time-step;
     - N_steps:     number of iterations updating the grid;
     - skip:        number of starting iterations to ignore (considered equilibration);
     - seed:        for the numpy pseudo-random number generator.
    Returns an array of grid configurations of the CA.
    """
    # === Input validation ===
    _validate_positive_int(size, "size")
    _validate_probability(p, "p")
    _validate_probability(true_frac, "true_frac")
    _validate_probability(f_update, "f_update")
    _validate_positive_int(N_steps, "N_steps")
    _validate_positive_int(M, "M")

    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
    if skip >= N_steps:
        raise ValueError(f"skip ({skip}) must be less than N_steps ({N_steps})")
    if M >= size // 2:
        raise ValueError(f"M ({M}) should be less than size/2 ({size//2})")

    # === Setup ===
    np.random.seed(seed)
    grid = initialize_CA(p, size)
    grids = []
    N_update = int(f_update * size**2)  # number of cells to update at each step

    if update_rule.__name__ == 'update_Scanlon2007':
        update_args = [true_frac, k, M]
    elif update_rule.__name__ == 'update_basic':
        update_args = []
    else:
        raise ValueError(f"Unknown update rule: {update_rule.__name__}")

    # === Main loop ===
    for n in range(N_steps):
        # randomly select a cells to update
        cells_to_update = np.column_stack([
            np.random.randint(0, size, N_update),
            np.random.randint(0, size, N_update)
        ])

        # update the grid one step
        grid = update_rule(grid, cells_to_update, *update_args)

        # Internal invariant check
        assert grid.shape == (size, size), "grid shape changed unexpectedly"

        # if we are beyond equilibration, save the grid to the list to return
        if n >= skip:
            grids.append(grid.copy())

    return grids
