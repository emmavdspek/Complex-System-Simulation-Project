"""
helper functions for handling generated data (saving, loading)
"""

from pathlib import Path
import numpy as np

from multiprocessing import Pool
import src.CA_model as CA


def save_data(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed):
    # name and directory of file to save the data to
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={np.round(true_frac,2)}_k={k}_M={M}_seed={seed}.npy"

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


def single_run_emma(size, p, update_rule, true_frac, k, M, N_steps, skip, seed):
    grids = CA.evolve_CA(
        size=size,
        p=p,
        update_rule=update_rule,
        true_frac=true_frac,
        k=k,
        M=M,
        N_steps=N_steps,
        skip=skip,
        seed=seed,
    )
    save_data(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed)
    return


def generate_parallel_emma(size, p, update_rule, true_fracs, k, M, N_steps, skip, seed):
    PROCESSES = 4
    seeds = np.arange(seed, seed + len(true_fracs))
    args_lists = [
        (size, p, update_rule, true_fracs[i], k, M, N_steps, skip, seeds[i])
        for i in range(len(true_fracs))
    ]
    with Pool(PROCESSES) as pool:
        pool.starmap(single_run_emma, args_lists)
    return


def save_data_lila(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed, phi):
    # name and directory of file to save the data to
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={np.round(true_frac,2)}_k={k}_M={M}_seed={seed}_phi={np.round(phi, 2)}.npy"

    # first check if the file already exists
    path = Path(filename)
    if path.exists():
        print("This file already exists, data cannot be saved.")
    else:
        np.save(filename, np.array(grids, dtype=bool), allow_pickle=True)
        print("Data successfully saved to: ", filename)
    return


def load_data_lila(size, update_rule, true_frac, k, M, N_steps, skip, seed, phi):
    # name and directory of file to retrieve data from
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={np.round(true_frac,2)}_k={k}_M={M}_seed={seed}_phi={np.round(phi,2)}.npy"
    print("Loading the data from: ", filename)

    loaded_grids = np.load(filename)

    return loaded_grids


def single_run_lila(size, p, update_rule, true_frac, k, M, N_steps, skip, seed, phi):
    grids = CA.evolve_CA(
        size=size,
        p=p,
        update_rule=update_rule,
        true_frac=true_frac,
        k=k,
        M=M,
        N_steps=N_steps,
        skip=skip,
        seed=seed,
        phi=phi,
    )
    save_data_lila(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed, phi)
    return


def generate_parallel_lila(
    size, p, update_rule, true_frac, k, M, N_steps, skip, seed, phis
):
    PROCESSES = 10
    seeds = np.arange(seed, seed + len(phis))
    args_lists = [
        (size, p, update_rule, true_frac, k, M, N_steps, skip, seeds[i], phis[i])
        for i in range(len(phis))
    ]
    with Pool(PROCESSES) as pool:
        pool.starmap(single_run_lila, args_lists)
    return
