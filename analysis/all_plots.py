import json
from pathlib import Path
from typing import List

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import rc
from tqdm import tqdm

from analysis.plot_ph_dim import ph_dim
from analysis.plot_E_alpha import E_alpha
from analysis.plot_magnitude_simple import magnitude, positive_magnitude
from analysis.plot_magnitude_simple import magnitude_integral
from analysis.kendall import granulated_kendall_from_dict
from analysis.pearson import granulated_pearson_from_dict
from analysis.plot_from_json import plot_from_dict
from analysis.curvature import curvature

# list of other functions to use
FUNCTIONS = [curvature]

T_MAGNITUDE = np.sqrt(50000)


def compute_all_quantities_one_seed(
        json_path: str,
        alpha: float = 1.,
        t: float = T_MAGNITUDE,
        stems: List[str] = [
            "_01"
        ],
        min_points: int = 200,
        max_points: int = None,
        jump: int = 200,
        t_min: float = 1.,
        t_max: float = np.sqrt(50000),
        n_points_curvature: int = 20,
        save_plots: bool=True,
        generalization_key: str = "worst_gen",
        n: int=50000,
        pseudo_matrix_data_proportion: float = 0.1,
        pseudo_distance_type: str = "manhattan",
        theoretical_t: bool = True,
        functions: list = FUNCTIONS
    ):
    # WARNING: n should be (number of data points) * (pseudo_matrix_data_proportion)
    logger.warning(f"The specified value of n is {n}")

    logger.debug(f"Used stems: {stems}")

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    logger.info(f"Found {len(results.keys())} random seeds")

    # HACK: change it
    logger.warning(f"For now only the first random seed is used")
    seed = list(results.keys())[0]

    # number of datapoints, useful to renormalize some distance matrices
    # n = results[seed][]

    complexity_names = []

    new_json_path = json_path.parent / "all_results_augmented.json"
    if new_json_path.exists():
        with open(str(new_json_path), "r") as new_json_file:
            new_results = json.load(new_json_file)
    else:
        new_results = results.copy()

    # Computation of all the quantities
    # We loop through the experiments and compute everything
    for key in tqdm(results[seed].keys()):

        # Creation of the worst generalization error
        if generalization_key == "worst_gen":
            results[seed][key]["worst_gen"] = results[seed][key]['train_acc'] -\
                                             results[seed][key]['worst_acc']
        if generalization_key == "worst_gen_loss":
            results[seed][key]["worst_gen_loss"] = results[seed][key]['train_loss'] -\
                                             results[seed][key]['worst_loss']

        # Now we loop over the potential distance matrices
        for stem in stems:

            dist_matrix_path = Path(results[seed][key]["saved_distance_matrix" + stem])
            logger.info(f"Using distance matrix {str(dist_matrix_path)}")
            if not dist_matrix_path.exists():
                raise FileNotFoundError(str(dist_matrix_path))

            dist_matrix = np.load(str(dist_matrix_path))

            # HACK: renormalization of the matrix
            if stem in ["", "_01"]:
                assert pseudo_distance_type in ["euclidean", "manhattan"], pseudo_distance_type
                if pseudo_distance_type == "euclidean":
                    dist_matrix = dist_matrix / np.sqrt(pseudo_matrix_data_proportion * n)
                else:
                    dist_matrix = dist_matrix / (pseudo_matrix_data_proportion * n)

            if theoretical_t:
                t = np.sqrt(n)
            
            # We loop over the functions defined in functions
            for f in functions:
                complexity_name = f.__name__ + "_dist_matrix" + stem
                complexity_names.append(f.__name__ + "_dist_matrix" + stem)
                # kwargs should be allowed in all functions, otherwise there will be errors
                complexity = f(
                    dist_matrix,
                    alpha=alpha,
                    min_points=min_points,
                    max_points=max_points,
                    t=t,
                    jump=jump
                )
                new_results[seed][key].update({
                    "alpha": alpha,
                    "min_points": min_points,
                    "max_points": max_points,
                    "jump": jump,
                    "t": t
                })
                new_results[seed][key][complexity_name] = float(complexity)
    
    complexity_names = list(dict.fromkeys(complexity_names))
    logger.debug(f"Computed complexities: {complexity_names}")

    # Kendall coefficients 
    granulated_kendalls = granulated_kendall_from_dict(
        new_results[seed],
        generalization_key = generalization_key,
        complexity_keys = complexity_names
    )

    # We check if old granulated kendalls already exist
    kendall_path = json_path.parent / "granulated_kendalls.json"
    if kendall_path.exists():
        with open(str(kendall_path), "r") as old_kendall_file:
            old_kendalls = json.load(old_kendall_file)
        granulated_kendalls["granulated Kendalls"].update(old_kendalls["granulated Kendalls"])
        granulated_kendalls["Kendall tau"].update(old_kendalls["Kendall tau"])
    with open(str(kendall_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)

    # Pearson coefficients 
    granulated_pearsons = granulated_pearson_from_dict(
        new_results[seed],
        generalization_key = generalization_key,
        complexity_keys = complexity_names
    )

    # We check if old granulated kendalls already exist
    pearson_path = json_path.parent / "granulated_pearsons.json"
    if pearson_path.exists():
        with open(str(pearson_path), "r") as old_pearson_file:
            old_pearsons = json.load(old_pearson_file)
        granulated_pearsons["granulated pearsons"].update(old_pearsons["granulated pearsons"])
        granulated_pearsons["pearson tau"].update(old_pearsons["pearson tau"])
    with open(str(pearson_path), "w") as json_file:
        json.dump(granulated_pearsons, json_file, indent=2)
    
    # Saving the new results file
    # in a new one to avoid overwriting
    try:
        with open(str(new_json_path), "w") as json_file:
            json.dump(new_results, json_file, indent=2)
    except:
        logger.warning(f"Unable to save computed quantities in {str(new_json_path)}")

    # Plots
    for stem in stems:
        for f in functions:

            complexity_name = f.__name__ + "_dist_matrix" + stem
            output_dir = json_path.parent / ("figures" + stem)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (output_dir / f.__name__).with_suffix(".png")

            plot_from_dict(
                new_results[seed],
                output_path=output_path,
                generalization_key=generalization_key,
                complexity_key=complexity_name,
                ylabel=f.__name__.replace("_", " ")
            )

if __name__ == "__main__":
    fire.Fire(compute_all_quantities_one_seed)
            
