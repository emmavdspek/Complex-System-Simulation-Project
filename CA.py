import numpy as np
import matplotlib.pyplot as plt

def initialize_CA(p=0.5, size=500):
    """Initializes a (pseudo-)randomly generated grid of occupied (t) and unoccupied (o) sites, 
    based on a probability of occupation p. Returns the grid."""
    grid = np.random.choice(np.array([0,1]), size=(size,size), p=np.array([p, 1-p]))
    return grid


def show_grids(grids:list):
    """Plots the supplied list of grids next to each other."""
    if len(grids) > 10:
        print("The amount of grids supplied is to many, only plotting the first 10.")
        grids = grids[:10]
    fig, axs = plt.subplots(1,len(grids), figsize=(len(grids)*2, 3), sharey=True, constrained_layout=True)
    for a in range(len(axs)):
        ax = axs[a]
        ax.imshow(grids[a], cmap="YlGn_r")
        ax.set_xlabel("x")
    axs[0].set_ylabel("y")
    plt.show()
    return


def update_Scanlon2007(grid, random_sel, true_frac=0.3, k=3.0, M=25):
    """Update algorithm according to the model presented in Scanlon et al. (2007).
    Input parameters: see evolve_CA.
    Returns the CA grid after one iteration of updating."""
    current_grid = grid.copy()
    size = np.shape(current_grid)[0]

    for (i,j) in random_sel:
        cum_distr = 0
        numerator = 0
        for dx in range(-M, M+1):
            for dy in range(-M, M+1):
                dist = np.sqrt((dx)**2 + (dy)**2)
                # if the neighbor is within the system and within range, add to the nom/denum contr. of rho
                if (0<=(i+dx)<size) and (0<=(j+dy)<size) and (0 < dist <= M):
                    cum_distr += (1/dist)**k
                    numerator += (1/dist)**k * current_grid[i+dx, j+dy]
        
        # define parameters necessary for the update rule
        rho = numerator/cum_distr
        frac_occ = np.sum(current_grid)/(size**2)
        random_nr = np.random.random()

        # if the cell is currently unoccupied, update according to the P(o->t) rule
        if current_grid[i,j] == 0:
            prob_flip = rho + (true_frac-frac_occ) / (1-frac_occ)
            # if the probability exceeds that of a randomly generated number, flip the state to occupied
            if random_nr < prob_flip:
                grid[i,j] = 1
        # if the cell is currently occupied, update according to the P(t->o) rule
        elif current_grid[i,j] == 1:
            prob_flip = (1-rho) + (frac_occ-true_frac) / (frac_occ)
            # if the probability exceeds that of a randomly generated number, flip the state to occupied
            if random_nr < prob_flip:
                grid[i,j] = 0

    return grid


def evolve_CA(size=500, p=0.5, true_frac=0.1, k=3.0, M=25, f_update=0.2, N_steps=200, skip=100, seed=0):
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
    N_update = int(f_update*size)

    for n in range(N_steps):
        # current_grid = grid.copy()
        random_sel = np.reshape(np.random.choice(size, N_update*2), (N_update, 2))
        grid = update_Scanlon2007(grid, random_sel, true_frac, k, M)
        if n >= skip:
            grids.append(grid.copy())
    return grids


# Test the functions
grids = evolve_CA(size=20, p=0.6, M=5, N_steps=100, skip=0)
show_grids([grids[0], grids[-1]])