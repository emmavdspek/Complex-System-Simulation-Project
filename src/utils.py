"""
helper functions for handling generated data (saving, loading)
"""

from pathlib import Path
import numpy as np


def save_data(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed):
    # name and directory of file to save the data to
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={true_frac}_k={k}_M={M}_seed={seed}.npy"

    # first check if the file already exists
    path = Path(filename)
    if path.exists():
        print("This file already exists, data cannot be saved.")
    else:
        np.save(filename, np.array(grids, dtype=bool), allow_pickle=True)
        print("Data successfully saved to: ", filename)
    return


def load_data(size, update_rule, true_frac, k, M, N_steps, skip, seed=0):
    # name and directory of file to retrieve data from
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={true_frac}_k={k}_M={M}_seed={seed}.npy"
    print("Loading the data from: ", filename)

    loaded_grids = np.load(filename)

    return loaded_grids
