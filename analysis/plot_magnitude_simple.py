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

from analysis.kendall import granulated_kendall_from_dict


# unused
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


# unused
def magnitude_cloesky(dist_matrix: np.ndarray, t: float = 1.) -> float:

    M = np.exp(-dist_matrix)
    c, lower = cho_factor(M)
    x = solve_triangular(c, np.ones(M.shape[0]), trans=1)

    return x.T @ x


def magnitude(dist_matrix: np.ndarray, t: float = 35., **kwargs) -> float:

    # from the PhD thesis of Salim, Shilan (2021)
    N=dist_matrix.shape[0]
    w = np.ones(N)/N
    linear_system = LinearSystem(np.exp(-t * dist_matrix), np.ones(N), self_adjoint = True, positive_definite = True)
    w = Cg(linear_system,  x0 = w).xk
    magnitude_weights = w.squeeze()

    return magnitude_weights.sum()

def magnitude_chaining(dist_matrix: np.ndarray,\
                        t_min: float = 0.01, 
                        t_max: float = 1.,
                        n_points: int = 10,
                         **kwargs) -> float:

    # ts = np.linspace(t_min, t_max, n_points)
    ts = np.exp(np.linspace(np.log(t_min), np.log(t_max)))
    integrands = []

    for k in range(len(ts)):
        mag = magnitude(dist_matrix, ts[k])
        integrands.append(np.sqrt(np.log(mag)) / (ts[k]**2))

    integrands = np.array(integrands)
    selected_index = np.argmax(integrands)
    logger.info(f"Selected scale of the magnitude: {ts[selected_index]}")

    integral = (integrands[:-1] * (ts[1:] - ts[:-1])).sum()

    return integral


def magnitude_integral(dist_matrix: np.ndarray,\
                        t_min: float = 0.01, 
                        t_max: float = np.sqrt(50000),
                        n_points: int = 10,
                         **kwargs) -> float:

    # ts = np.linspace(t_min, t_max, n_points)
    ts = np.exp(np.linspace(np.log(t_min), np.log(t_max)))
    integrands = []

    for k in range(len(ts)):
        mag = magnitude(dist_matrix, ts[k])
        integrands.append(np.sqrt(np.log(mag)) / (ts[k]))

    integrands = np.array(integrands)
    selected_index = np.argmax(integrands)
    logger.info(f"Selected scale of the magnitude: {ts[selected_index]}")

    integral = (integrands[:-1] * (ts[1:] - ts[:-1])).sum()

    return integral


def plot_magnitude_one_seed(seed_results: dict, ax, output_dir: str=None, t: float=1., stem: str="_01"):

    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    magnitude_tab = []
    lr_tab = []
    bs_tab = []

    complexity_key = "magnitude"

    for key in tqdm(seed_results.keys()):

        # acc_gap_tab.append(seed_results[key]['acc_gap'])
        if 'worst_acc' in seed_results[key].keys():
            acc_gap_tab.append(seed_results[key]['train_acc'] - seed_results[key]['worst_acc'])
            #acc_gap_tab.append(seed_results[key]['acc_gap'])
        else:
            acc_gap_tab.append(seed_results[key]['acc_gap'])
        lr_tab.append(seed_results[key]['learning_rate'])
        bs_tab.append(seed_results[key]['batch_size'])

        dist_matrix_path = Path(seed_results[key]["saved_distance_matrix" + stem])
        if not dist_matrix_path.exists():
            raise FileNotFoundError(str(dist_matrix_path))
        
        n = 5000
        if stem in ["", "_01"]:
            dist_matrix = np.load(str(dist_matrix_path)) / n
        else:
            dist_matrix = np.load(str(dist_matrix_path)) 

        #epsilon = 0.01
        # # t = np.sqrt(n) / np.max(dist_matrix[np.nonzero(dist_matrix)])
        # t = (np.log(5000) / epsilon) 
        # logger.info(f'Value of t: {t}')

        complexity = magnitude_chaining(dist_matrix)
        # complexity = 1. / (1. / complexity - np.exp(-t * epsilon))
        logger.info(f"Complexity: {complexity}") 

        magnitude_tab.append(complexity)

        seed_results[key][complexity_key] = float(complexity)

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
        save_path = Path(output_dir) / "magnitude_vs_generalization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")
    
    granulated_kendalls = granulated_kendall_from_dict(
        seed_results,
        complexity_keys=[complexity_key]
    )

    return sc, granulated_kendalls, seed_results

    
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

    sc, _, _ = plot_magnitude_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()

def plot_magnitude_all_seed(json_path: str, t:float=0.1, stem:str=""):

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
        sc, granulated_kendalls, seed_results = plot_magnitude_one_seed(results[seed], ax, output_dir=None, t=t, stem=stem)
        new_results[seed] = seed_results
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / ("figures" + stem)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "magnitude_vs_generalization_error.png"

    ax.grid(visible=True, which="both")
    plt.grid(visible=True, which="both")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()

    granulated_kendalls["t"] = t

    json_path = output_dir / "magnitude_granulated_kendalls.json"
    with open(str(json_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)

    results_path = output_dir.parent / "all_results.json"
    with open(str(results_path), "w") as json_file:
        json.dump(new_results, json_file, indent=2)


if __name__ == "__main__":
    fire.Fire(plot_magnitude_all_seed)












    


