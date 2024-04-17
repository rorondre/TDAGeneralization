import json
from pathlib import Path

import fire
import matplotlib
from loguru import logger
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.stats import linregress

from analysis.plot_magnitude import magnitude
from analysis.kendall import granulated_kendall_from_dict

from scipy.linalg import cho_factor, solve_triangular, solve
from tqdm import tqdm
from krypy.linsys import LinearSystem, Cg
from scipy.optimize import toms748

from sklearn.linear_model import LinearRegression


# helper functions
def func(x, a, b):
    return a * x + b

def guess_target_point(X, target_value, guess=10):
    # ts=[]
    # mags=[]

    def f(x, W=X):
        # mag = comp_mag(W, ts=[x])
        mag = magnitude_individual(W, t=x)
        return mag - target_value

    lower_guess = 0  # np.log(X.shape[0]-1)
    f_guess = f(guess)
    while (f_guess < 0):  # & (guess < 10**20)):
        lower_guess = guess
        guess = guess * 10
        f_guess = f(guess)
    max_iterations = 100
    conv = toms748(f, lower_guess, guess, maxiter=max_iterations, rtol=1e-05)
    # conv = mag_convergence(X, lower_guess, guess, f, max_iterations=100)

    return conv

def magnitude_individual(dist_matrix: np.ndarray, t: float = 1.) -> float:
    # from the PhD thesis of Salim, Shilan (2021)
    N = dist_matrix.shape[0]
    w = np.ones(N) / N
    linear_system = LinearSystem(np.exp(-t * dist_matrix), np.ones(N), self_adjoint=True, positive_definite=True)
    w = Cg(linear_system, x0=w).xk
    magnitude_weights = w.squeeze()
    return magnitude_weights.sum()

# Stupid function in place of the real one that Rayna is implementing
def mag_dim(dist_matrix: np.ndarray,
           initial_interval: float = 0.,
           end_interval: float = 1.,
           seed: int = 42,
           alpha: float = 1.) -> float:
    """
    This function:
     - takes as an input a distance matrix
     - use it to compute mag dim

    :param dm: distance matrix, should be of shape (N, N)
    """
    assert dist_matrix.ndim == 2, dist_matrix
    assert dist_matrix.shape[0] == dist_matrix.shape[1], dist_matrix.shape

    # fit a line
    target_value = 0.95*dist_matrix.shape[0]
    logger.info(f"The shape of the distance matrix is: {dist_matrix.shape[0]}")

    # #compute all magnitude
    guess = 10
    conv = guess_target_point(dist_matrix, target_value, guess=guess)
    low = 1 
    num_intervals = 100

    logts = np.linspace(np.log(low), np.log(conv), num=num_intervals)

    magnitudes = []

    for t in logts:
        mag = magnitude_individual(dist_matrix, t=np.exp(t))
        magnitudes.append(np.log(mag))

    # use linear regression to compute the dimension
    # X = logts
    X = logts.reshape(-1, 1)
    # print('X is ', X)
    y = magnitudes
    reg = LinearRegression().fit(X, y)
    mag_dim_auto = reg.coef_[0]
    logger.info(f"The dimension is {mag_dim_auto}")
    return mag_dim_auto

def plot_mag_dim_one_seed(seed_results: dict, ax, output_dir: str = None):
    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    mag_dim_tab = []
    lr_tab = []
    bs_tab = []

    complexity_key = "magnitude_dimension"

    for key in tqdm(seed_results.keys()):

        acc_gap_tab.append(seed_results[key]['acc_gap'])
        lr_tab.append(seed_results[key]['learning_rate'])
        bs_tab.append(seed_results[key]['batch_size'])

        dist_matrix_path = Path(seed_results[key]["saved_distance_matrix"])
        if not dist_matrix_path.exists():
            raise FileNotFoundError(str(dist_matrix_path))

        dist_matrix = np.load(str(dist_matrix_path))

        complexity = mag_dim(dist_matrix)
        mag_dim_tab.append(complexity)

        seed_results[key][complexity_key] = float(complexity)

    # TODO: make the markers vary with the batch size
    markers = "o"

    color_map = plt.cm.get_cmap('viridis_r')

    sc = ax.scatter(
        acc_gap_tab,
        mag_dim_tab,
        c=lr_tab,
        cmap=color_map,
        marker=markers,
        norm=matplotlib.colors.LogNorm()
    )

    plt.xlabel("Generalization error", weight="bold")
    # this might not work//needs to be improved
    plt.ylabel(r"$\mathrm{dim_{mag}}$", weight="bold")
    plt.grid()

    if output_dir is not None:
        save_path = Path(output_dir) / "mag_dim_vs_generalization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")

    granulated_kendalls = granulated_kendall_from_dict(
        seed_results,
        complexity_keys=[complexity_key]
    )

    return sc, granulated_kendalls, seed_results


def plot_mag_dim_one_seed_from_json(json_path: str):
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    plt.figure()
    ax = plt.axes()

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    sc, _, _ = plot_mag_dim_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_mag_dim_all_seed(json_path: str):
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    new_results = {}

    plt.figure()
    ax = plt.axes()

    for seed in results.keys():
        sc, granulated_kendalls, seed_results = plot_mag_dim_one_seed(results[seed], ax, output_dir=None)
        new_results[seed] = seed_results

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "mag_dim_vs_generalization_error.png"

    ax.grid(visible=True, which="minor")
    plt.grid(visible=True, which="minor")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()

    json_path = output_dir / "magnitude_dimensions_granulated_kendalls.json"
    with open(str(json_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)

    results_path = output_dir.parent / "all_results.json"
    with open(str(results_path), "w") as json_file:
        json.dump(new_results, json_file, indent=2)


if __name__ == "__main__":
    fire.Fire(plot_mag_dim_all_seed)















