import json
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm
from gph.python import ripser_parallel

from analysis.kendall import granulated_kendall_from_dict


# Stupid function in place of the real one that Rayna is implementing
def E_alpha(dist_matrix: np.ndarray, h_dim=0, alpha: float = 1.) -> float:

    diagrams = ripser_parallel(dist_matrix, maxdim=0, n_threads=5, metric="precomputed")['dgms']
    d = diagrams[h_dim]
    d = d[d[:, 1] < np.inf]
    alpha_sum = np.power((d[:, 1] - d[:, 0]), alpha).sum()

    return alpha_sum


def plot_E_alpha_one_seed(seed_results: dict, ax, output_dir: str=None, alpha: float=1., stem:str=""):

    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    E_alpha_tab = []
    lr_tab = []
    bs_tab = []

    complexity_key = "E_alpha"

    for key in tqdm(seed_results.keys()):

        # acc_gap_tab.append(seed_results[key]['acc_gap'])
        if 'worst_acc' in seed_results[key].keys():
            acc_gap_tab.append(seed_results[key]['train_acc'] - seed_results[key]['worst_acc'])
        else:
            acc_gap_tab.append(seed_results[key]['acc_gap'])
        lr_tab.append(seed_results[key]['learning_rate'])
        bs_tab.append(seed_results[key]['batch_size'])

        dist_matrix_path = Path(seed_results[key]["saved_distance_matrix" + stem])
        logger.info(f"Using distance matrix {str(dist_matrix_path)}")

        if not dist_matrix_path.exists():
            raise FileNotFoundError(str(dist_matrix_path))
        
        dist_matrix = np.load(str(dist_matrix_path))

        complexity = E_alpha(dist_matrix, alpha=alpha)
        E_alpha_tab.append(complexity)

        seed_results[key][complexity_key] = float(complexity)

    # TODO: make the markers vary with the batch size
    markers = "o"

    color_map = plt.cm.get_cmap('viridis_r')

    sc = ax.scatter(
        acc_gap_tab,
        E_alpha_tab,
        c = lr_tab,
        cmap = color_map,
        marker = markers,
        norm=matplotlib.colors.LogNorm()
    )

    ax.set_yscale("log")

    plt.xlabel("Generalization error", weight="bold")
    plt.ylabel(r"$\mathbf{E_\alpha}$", weight="bold")
    plt.grid()
    
    if output_dir is not None:
        save_path = Path(output_dir) / "E_alpha_vs_generalization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")

    granulated_kendalls = granulated_kendall_from_dict(
        seed_results,
        complexity_keys=[complexity_key]
    )
    
    return sc, granulated_kendalls, seed_results

    
def plot_E_alpha_one_seed_from_json(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    plt.figure()
    ax = plt.axes()

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    sc, _, _ = plot_E_alpha_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_E_alpha_all_seed(json_path: str, alpha: float = 1., stem: str=""):

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
        sc, granulated_kendalls, seed_results = plot_E_alpha_one_seed(results[seed],
                                     ax, 
                                     output_dir=None,
                                     alpha=alpha,
                                     stem=stem)
        new_results[seed] = seed_results
        

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / ("figures" + stem)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "E_alpha_vs_generalization_error.png"

    ax.grid(visible=True, which="minor")
    plt.grid(visible=True, which="minor")

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()

    granulated_kendalls["alpha"] = alpha

    json_path = output_dir / "E_alpha_granulated_kendalls.json"
    with open(str(json_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)

    # logger.debug(json.dumps(new_results, indent=2))
    
    results_path = output_dir.parent / "all_results.json"
    with open(str(results_path), "w") as json_file:
        json.dump(new_results, json_file, indent=2)



if __name__ == "__main__":
    fire.Fire(plot_E_alpha_all_seed)












    


