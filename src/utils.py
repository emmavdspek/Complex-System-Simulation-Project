"""
helper functions for handling generated data (saving, loading)
"""

from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
import src.CA_model as CA


def load_data_wo_phi(size, update_rule, true_frac, k, M, N_steps, skip, seed=0):
    # name and directory of file to retrieve data from
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={true_frac}_k={k}_M={M}_seed={seed}.npy"
    print("Loading the data from: ", filename)

    loaded_grids = np.load(filename)

    return loaded_grids


def save_data(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed, phi=0.5):
    """Saves a generated evolution of grids into a .npy file. Arguments:
     - grids:       list of 2D grids corresponding to iteration steps of a simulation;
     - all other:   parameters that were used in generating the data.
    The data will be saved in a folder data/[UPDATE_RULE]/, under a name that contains all
    parameter settings. If this folder doesn't exist yet, it will be generated."""

    # name and directory of file to save the data to
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={np.round(true_frac,2)}_k={k}_M={M}_seed={seed}_phi={np.round(phi, 2)}.npy"

    # if the (sub)folders to save to do not exist yet, make them
    folder_path = Path(f"../data/{update_rule.__name__}")
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True)

    # first check if the file already exists
    path = Path(filename)
    if path.exists():
        print("This file already exists, data cannot be saved.")
    else:
        np.save(filename, np.array(grids, dtype=bool), allow_pickle=True)
        print("Data successfully saved to: ", filename)
    return


def load_data(size, update_rule, true_frac, k, M, N_steps, skip, seed, phi=0.5):
    """Loads data as saved by save_data(). Arguments are all the parameters settings
    as given to save_data()."""
    # name and directory of file to retrieve data from
    filename = f"../data/{update_rule.__name__}/size={size}_Nsteps={N_steps}_skip={skip}_truefrac={np.round(true_frac,2)}_k={k}_M={M}_seed={seed}_phi={np.round(phi,2)}.npy"
    print("Loading the data from: ", filename)

    loaded_grids = np.load(filename)

    return loaded_grids


# ------------------- PARALLELLIZED SIMULATION FUNCTIONS -------------------#


def single_run(size, p, update_rule, true_frac, k, M, N_steps, skip, seed, phi):
    """Function containing single job for the workers in the multiprocessing pool to complete.
    This consists of one simulating one full evolution of the CA given the set of parameters,
    as well as saving the data."""
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
    save_data(grids, size, update_rule, true_frac, k, M, N_steps, skip, seed, phi)
    return


def generate_parallel_true_fracs(
    size, p, update_rule, true_fracs, k, M, N_steps, skip, seed, phi=0.5
):
    """Function to generate and execute jobs (simulations) for multiprocessing.
    The parameters that are varied amongst the simulations are
     - true_frac:   a list of true_fracs should be supplied that sets the amount of jobs;
     - seed:        the supplied seed is interpreted as a 'starting' seed, and all parralel
                    jobs are supplied with seeds increased by 1 each time."""
    assert len(true_fracs) != 0, "You have not supplied a list of true_fracs to vary."

    PROCESSES = cpu_count()  # amount of processes that will be used (max)
    print(
        f"Starting parallel simulation of {len(true_fracs)} sets using {PROCESSES} processes..."
    )

    seeds = np.arange(seed, seed + len(true_fracs))
    args_lists = [
        (size, p, update_rule, true_fracs[i], k, M, N_steps, skip, seeds[i], phi)
        for i in range(len(true_fracs))
    ]
    with Pool(PROCESSES) as pool:
        pool.starmap(single_run, args_lists)

    print("... Finished!")
    return


def generate_parallel_phis(
    size, p, update_rule, true_frac, k, M, N_steps, skip, seed, phis
):
    """Function to generate and execute jobs (simulations) for multiprocessing.
    The parameters that are varied amongst the simulations are
     - phi:         a list of phis should be supplied that sets the amount of jobs;
     - seed:        the supplied seed is interpreted as a 'starting' seed, and all parralel
                    jobs are supplied with seeds increased by 1 each time."""

    assert len(phis) != 0, "You have not supplied a list of phis to vary."

    PROCESSES = cpu_count()  # amount of processes that will be used (max)
    print(
        f"Starting parallel simulation of {len(phis)} sets using {PROCESSES} processes..."
    )

    seeds = np.arange(seed, seed + len(phis))
    args_lists = [
        (size, p, update_rule, true_frac, k, M, N_steps, skip, seeds[i], phis[i])
        for i in range(len(phis))
    ]
    with Pool(PROCESSES) as pool:
        pool.starmap(single_run, args_lists)

    print("... Finished!")
    return
