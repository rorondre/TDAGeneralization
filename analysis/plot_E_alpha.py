import json
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm


# Stupid function in place of the real one that Rayna is implementing
def E_alpha(dist_matrix: np.ndarray, alpha: float = 1.) -> float:
    return np.random.rand()


def plot_E_alpha_one_seed(seed_results: dict, ax, output_dir: str=None):

    num_exp = len(seed_results.keys())
    logger.info(f"Found {num_exp} experiments")

    acc_gap_tab = []
    E_alpha_tab = []
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
        E_alpha_tab.append(E_alpha(dist_matrix))

    # TODO: make the markers vary with the batch size
    markers = "x"

    color_map = plt.cm.get_cmap('viridis_r')

    sc = ax.scatter(
        acc_gap_tab,
        E_alpha_tab,
        c = lr_tab,
        cmap = color_map,
        marker = markers,
        norm=matplotlib.colors.LogNorm()
    )

    plt.xlabel("Generalization error", weight="bold")
    plt.ylabel(r"$\mathbf{E_\alpha}$", weight="bold")
    plt.grid()
    
    if output_dir is not None:
        save_path = Path(output_dir) / "E_alpha_vs_generealization_error.png"
        plt.savefig(str(save_path))
        logger.info(f"Saving figure in {str(save_path)}")
    
    return sc

    
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

    sc = plot_E_alpha_one_seed(results, ax, str(output_dir), save=True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    plt.close()


def plot_E_alpha_all_seed(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    plt.figure()
    ax = plt.axes()

    for seed in results.keys():
        sc = plot_E_alpha_one_seed(results[seed], ax, output_dir=None)

    cbar = plt.colorbar(sc)
    cbar.set_label("Learning rate")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "E_alpha_vs_generealization_error.png"

    plt.grid()

    plt.savefig(str(save_path))
    logger.info(f"Saved figure in {str(save_path)}")
    plt.close()


if __name__ == "__main__":
    fire.Fire(plot_E_alpha_all_seed)












    


