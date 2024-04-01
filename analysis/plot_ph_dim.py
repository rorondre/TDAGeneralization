import json
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm
from gph.python import ripser_parallel

from analysis.plot_E_alpha import E_alpha
from analysis.kendall import granulated_kendall_from_dict


# Stupid function in place of the real one that Rayna is implementing
def ph_dim(dist_matrix: np.ndarray,
           min_points=200,
           max_points=2000,
           point_jump=200,
           h_dim=0,
           seed: int = 42,
           alpha: float = 1.) -> float:
    """
    This function:
     - takes as an input a distance matrix
     - use it to compute PH dim

    :param dm: distance matrix, should be of shape (N, N)
    """
    assert dist_matrix.ndim == 2, dist_matrix
    assert dist_matrix.shape[0] == dist_matrix.shape[1], dist_matrix.shape

    np.random.seed(seed)

    if max_points is None:
        max_points = dist_matrix.shape[0]
    assert max_points <= dist_matrix.shape[0], (max_points, dist_matrix.shape[0])

    test_n = range(min_points, max_points, point_jump)
    lengths = []

    for points_number in test_n:

        sample_indices = np.random.choice(dist_matrix.shape[0], points_number, replace=False)
        dist_matrix_temp = dist_matrix[sample_indices, :][:, sample_indices]
        alpha_sum = E_alpha(dist_matrix_temp, h_dim=0, alpha=alpha)
        lengths.append(alpha_sum)

    lengths = np.array(lengths)

    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()

    error = ((y - (m * x + b)) ** 2).mean()

    logger.debug(f"Ph Dimension Calculation has an approximate error of: {error}.")

    return alpha / (1 - m)

def plot_ph_dim_one_seed(seed_results: dict, ax, output_dir: str = None):
    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    ph_dim_tab = []
    lr_tab = []
    bs_tab = []

    complexity_key = "ph_dim"

    for key in tqdm(seed_results.keys()):

        acc_gap_tab.append(seed_results[key]['acc_gap'])
        lr_tab.append(seed_results[key]['learning_rate'])
        bs_tab.append(seed_results[key]['batch_size'])

        dist_matrix_path = Path(seed_results[key]["saved_distance_matrix"])
        if not dist_matrix_path.exists():
            raise FileNotFoundError(str(dist_matrix_path))

        dist_matrix = np.load(str(dist_matrix_path))

        complexity = ph_dim(dist_matrix)
        ph_dim_tab.append(complexity)

        seed_results[key][complexity_key] = complexity
        ph_dim_tab.append(ph_dim(dist_matrix))

    # TODO: make the markers vary with the batch size
    markers = "o"

    color_map = plt.cm.get_cmap('viridis_r')

    sc = ax.scatter(
        acc_gap_tab,
        ph_dim_tab,
        c=lr_tab,
        cmap=color_map,
        marker=markers,
        norm=matplotlib.colors.LogNorm()
    )

    plt.xlabel("Generalization error", weight="bold")
    # this might not work//needs to be improved
    plt.ylabel(r"$\mathrm{dim_{PH}}$", weight="bold")
    plt.grid()

    if output_dir is not None:
        save_path = Path(output_dir) / "Ph_dim_vs_generealization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")

    granulated_kendalls = granulated_kendall_from_dict(
        seed_results,
        complexity_keys=[complexity_key]
    )

    return sc, granulated_kendalls


def plot_ph_dim_one_seed_from_json(json_path: str):
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    plt.figure()
    ax = plt.axes()

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    sc, _ = plot_ph_dim_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_ph_dim_all_seed(json_path: str):
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    plt.figure()
    ax = plt.axes()

    for seed in results.keys():
        sc, granulated_kendalls = plot_ph_dim_one_seed(results[seed], ax, output_dir=None)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "Ph_dim_vs_generealization_error.png"

    ax.grid(visible=True, which="minor")
    plt.grid(visible=True, which="minor")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()

    json_path = output_dir / "ph_dim_granulated_kendalls.json"
    with open(str(json_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)


if __name__ == "__main__":
    fire.Fire(plot_ph_dim_all_seed)















