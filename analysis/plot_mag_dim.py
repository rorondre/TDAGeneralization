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


# helper functions
def func(x, a, b):
    return a * x + b

# Stupid function in place of the real one that Rayna is implementing
def mag_dim(dist_matrix: np.ndarray,
           ts: list,
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
    magnitudes = []
    for t in ts:
        magnitude_single = magnitude(dist_matrix, t)
        magnitudes.append(magnitude_single)

    epsilon = 0.0001

    initial = initial_interval
    end = end_interval

    #potential setting of the values; needs to change per case
    initial = np.log(ts[1]) - epsilon
    end = np.log(ts[6]) + epsilon

    # this needs to be automated
    # initialguess = [3, 5]
    x_dist = np.log(ts)
    xdiff_norm = np.log(magnitudes[0])

    mask = (x_dist > initial) & (x_dist < end)
    fit, covariance = optimize.curve_fit(
        func,
        x_dist[mask],
        xdiff_norm[mask])

    plt.scatter(x_dist, xdiff_norm, label="data", s=2)
    x_fit = np.linspace(x_dist[mask].min(), x_dist[mask].max(), 100)
    y_fit = func(x_fit, *fit)
    plt.plot(x_fit, y_fit, c="red", label="fit")
    return linregress(x_fit, y_fit).slope

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

        complexity = mag_dim(dist_matrix, alpha=alpha)
        mag_dim_tab.append(complexity)

        seed_results[key][complexity_key] = complexity

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
        save_path = Path(output_dir) / "mag_dim_vs_generealization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")

    granulated_kendalls = granulated_kendall_from_dict(
        seed_results,
        complexity_keys=[complexity_key]
    )

    return sc, granulated_kendalls


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

    sc, _ = plot_mag_dim_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_mag_dim_all_seed(json_path: str):
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    plt.figure()
    ax = plt.axes()

    for seed in results.keys():
        sc, granulated_kendalls = plot_mag_dim_one_seed(results[seed], ax, output_dir=None)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "mag_dim_vs_generealization_error.png"

    ax.grid(visible=True, which="minor")
    plt.grid(visible=True, which="minor")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()

    json_path = output_dir / "magnitude_dimensions_granulated_kendalls.json"
    with open(str(json_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)


if __name__ == "__main__":
    fire.Fire(plot_mag_dim_all_seed)















