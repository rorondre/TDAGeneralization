import json
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.linalg import cho_factor, solve_triangular, solve 
from tqdm import tqdm
from krypy.linsys import LinearSystem, Cg


# Stupid function in place of the real one that Rayna is implementing
def magnitude_slow(dist_matrix: np.ndarray, t: float = 1.) -> float:
    # this is a naive implementation, it can be improved
    if dist_matrix.shape[0] >= 1000:
        inv_fn = np.linalg.pinv
    else:
        inv_fn = np.linalg.inv
    try:
        Z = inv_fn(np.exp(-t * dist_matrix))
    except Exception as e:
        print(f'Exception: {e} for t: {t} perturbing matrix')
        D = (
                np.exp(-t * dist_matrix)
                +
                0.01 * np.identity(
            n=dist_matrix.shape[0]
        )  # perturb similarity mtx to invert
        )
        Z = inv_fn(D)
    return Z.sum()


def magnitude_cloesky(dist_matrix: np.ndarray, t: float = 1.) -> float:

    M = np.exp(-dist_matrix)
    c, lower = cho_factor(M)
    x = solve_triangular(c, np.ones(M.shape[0]), trans=1)

    return x.T @ x

def magnitude(dist_matrix: np.ndarray, t: float = 0.005) -> float:

    # from the PhD thesis of Salim, Shilan (2021)
    N=dist_matrix.shape[0]
    w = np.ones(N)/N
    linear_system = LinearSystem(np.exp(-t * dist_matrix), np.ones(N), self_adjoint = True, positive_definite = True)
    w = Cg(linear_system,  x0 = w).xk
    magnitude_weights = w.squeeze()

    return magnitude_weights.sum()





def plot_magnitude_one_seed(seed_results: dict, ax, output_dir: str=None):

    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    magnitude_tab = []
    lr_tab = []
    bs_tab = []

    for key in tqdm(seed_results.keys()):

        acc_gap_tab.append(seed_results[key]['acc_gap'])
        lr_tab.append(seed_results[key]['learning_rate'])
        bs_tab.append(seed_results[key]['batch_size'])

        dist_matrix_path = Path(seed_results[key]["saved_distance_matrix"])
        if not dist_matrix_path.exists():
            raise FileNotFoundError(str(dist_matrix_path))
        
        dist_matrix = np.load(str(dist_matrix_path))
        magnitude_tab.append(magnitude(dist_matrix))

    # TODO: make the markers vary with the batch size
    markers = "o"

    color_map = plt.cm.get_cmap('viridis_r')

    sc = ax.scatter(
        acc_gap_tab,
        magnitude_tab,
        c = lr_tab,
        cmap = color_map,
        marker = markers,
        norm=matplotlib.colors.LogNorm()
    )

    ax.set_yscale("log")

    plt.xlabel("Generalization error", weight="bold")
    plt.ylabel("Magnitude", weight="bold")
    plt.grid()
    
    if output_dir is not None:
        save_path = Path(output_dir) / "magnitude_vs_generealization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")
    
    return sc

    
def plot_magnitude_one_seed_from_json(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    plt.figure()
    ax = plt.axes()

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    sc = plot_magnitude_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_magnitude_all_seed(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    plt.figure()
    ax = plt.axes()

    for seed in results.keys():
        sc = plot_magnitude_one_seed(results[seed], ax, output_dir=None)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "magnitude_vs_generealization_error.png"

    ax.grid(visible=True, which="minor")
    plt.grid(visible=True, which="minor")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()


if __name__ == "__main__":
    fire.Fire(plot_magnitude_all_seed)












    


