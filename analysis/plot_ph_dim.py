import json
import multiprocessing
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
from analysis.pearson import granulated_pearson_from_dict

def ph_dim_multiprocessing(dist_matrix: np.ndarray,
           min_points=1,
           max_points=None,
           point_jump=50,
           h_dim=0,
           seed: int = 42,
           alpha: float = 1.,
           **kwargs) -> float:
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

    # creating the sample indices
    indices_list = []
    for points_number in test_n:
        sample_indices = np.random.choice(dist_matrix.shape[0], points_number, replace=False)
        indices_list.append(sample_indices)

    pool = multiprocessing.Pool(10)
    def f(s):
        dist_matrix_temp = dist_matrix[s, :][:, s]
        alpha_sum = E_alpha(dist_matrix_temp, h_dim=0, alpha=alpha)

    lengths = pool.map(f, indices_list)

    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()

    error = ((y - (m * x + b)) ** 2).mean()

    logger.debug(f"ph Dimension Calculation has an approximate error of: {error}.")

    return alpha / (1 - m)





# Stupid function in place of the real one that Rayna is implementing
def ph_dim(dist_matrix: np.ndarray,
           min_points=1,
           max_points=None,
           point_jump=200,
           h_dim=0,
           seed: int = 42,
           alpha: float = 1.,
           **kwargs) -> float:
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

    logger.debug(f"ph Dimension Calculation has an approximate error of: {error}.")

    return alpha / (1 - m)


def ph_dim_E_correction(dist_matrix: np.ndarray,
           min_points=200,
           max_points=2000,
           point_jump=200,
           h_dim=0,
           seed: int = 42,
           alpha: float = 1.,
           n: int = 50000) -> float:

    estimated_dim = max(0., ph_dim(
        dist_matrix,
        min_points=min_points,
        max_points=max_points,
        point_jump=point_jump,
        h_dim=h_dim,
        seed=seed,
        alpha=alpha
    ))

    corrected_E_alpha = E_alpha(dist_matrix, h_dim=0, alpha=estimated_dim)

    # return estimated_dim * np.log(np.sqrt(n)) + np.log(corrected_E_alpha)
    return np.sqrt((6. * np.log(corrected_E_alpha) + 2. * estimated_dim * np.log(n)))




def plot_ph_dim_one_seed(seed_results: dict,\
                         ax,\
                          output_dir: str = None,
                          min_points: int = 200,
                          max_points: int = None,
                          stem:str=""):
    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    ph_dim_tab = []
    lr_tab = []
    bs_tab = []

    complexity_key = "ph_dim"

    for key in tqdm(seed_results.keys()):
        
        # acc_gap_tab.append(seed_results[key]['acc_gap'])
        if 'worst_acc' in seed_results[key].keys():
            # acc_gap_tab.append(seed_results[key]['loss_gap'])
            acc_gap_tab.append(seed_results[key]['train_acc'] - seed_results[key]['worst_acc'])
        else:
            acc_gap_tab.append(seed_results[key]['acc_gap'])
        lr_tab.append(seed_results[key]['learning_rate'])
        bs_tab.append(seed_results[key]['batch_size'])

        dist_matrix_path = Path(seed_results[key]["saved_distance_matrix" + stem])
        logger.info(f"Using distance matrix {str(dist_matrix_path)}")
        if not dist_matrix_path.exists():
            raise FileNotFoundError(str(dist_matrix_path))

        dist_matrix = np.load(str(dist_matrix_path)) / 50000

        complexity = ph_dim(dist_matrix, min_points=min_points, max_points=max_points)
        ph_dim_tab.append(complexity)

        seed_results[key][complexity_key] = float(complexity)

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
        save_path = Path(output_dir) / "ph_dim_vs_generalization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")

    granulated_kendalls = granulated_kendall_from_dict(
        seed_results,
        complexity_keys=[complexity_key]
    )

    return sc, granulated_kendalls, seed_results


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

    sc, _, _ = plot_ph_dim_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_ph_dim_all_seed(json_path: str, \
                        min_points: int = 200,
                         max_points: int = 5000,
                         stem: str=""):
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    new_results = {}

    assert stem in ["", "_euclidean", "_01"], stem

    plt.figure()
    ax = plt.axes()

    for seed in results.keys():
        sc, granulated_kendalls, seed_results = plot_ph_dim_one_seed(results[seed],
                                                             ax, 
                                                             output_dir=None,
                                                             min_points=min_points,
                                                             max_points=max_points,
                                                             stem=stem)
        new_results[seed] = seed_results


    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / ("figures" + stem)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "ph_dim_vs_generalization_error.png"

    ax.grid(visible=True, which="both")
    plt.grid(visible=True, which="both")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()

    json_path = output_dir / "ph_dim_granulated_kendalls.json"
    with open(str(json_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)

    results_path = output_dir.parent / "all_results.json"
    with open(str(results_path), "w") as json_file:
        json.dump(new_results, json_file, indent=2)


if __name__ == "__main__":
    fire.Fire(plot_ph_dim_all_seed)















