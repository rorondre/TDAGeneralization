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
from analysis.plot_magnitude import ... as magnitude_rayna
from analysis.plot_magnitude_simple import magnitude as magnitude_simple
from analysis.kendall import granulated_kendall_from_dict
from analysis.plot_from_json import plot_from_dict

# list of other functions to use
FUNCTIONS = [E_alpha, magnitude_simple, ph_dim]


def compute_all_quantities_one_seed(
        json_path: str,
        alpha: float = 1.,
        t: float = 0.1,
        stems: List[str] = [
            "",
            "_01",
            "_euclidean"
        ],
        min_points: int = 200,
        max_points: int = None,
        jump: int = 200,
        save_plots: bool=True,
        generalization_key: str = "worst_gen",
        n: int=5000
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
                dist_matrix = dist_matrix / n
            if stem in ["pseudo_l2"]:
                dist_matrix = dist_matrix / np.sqrt(n)

            
            # We loop over the functions defined in functions
            for f in FUNCTIONS:
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
                results[seed][key].update({
                    "alpha": alpha,
                    "min_points": min_points,
                    "max_points": max_points,
                    "jump": jump,
                    "t": t
                })
                results[seed][key][complexity_name] = float(complexity)
    
    complexity_names = list(dict.fromkeys(complexity_names))
    logger.debug(f"Computed complexities: {complexity_names}")

    # Kendall coefficients 
    granulated_kendalls = granulated_kendall_from_dict(
        results[seed],
        generalization_key = generalization_key,
        complexity_keys = complexity_names
    )
    kendall_path = json_path.parent / "granulated_kendalls.json"
    with open(str(kendall_path), "w") as json_file:
        json.dump(granulated_kendalls, json_file, indent=2)
    
    # Saving the new results file
    # in a new one to avoid overwriting
    new_json_path = json_path.parent / "all_results_augmented.json"
    try:
        with open(str(new_json_path), "w") as json_file:
            json.dump(results, json_file, indent=2)
    except:
        logger.warning(f"Unable to save computed quantities in {str(new_json_path)}")

    # Plots
    for stem in stems:
        for f in FUNCTIONS:

            complexity_name = f.__name__ + "_dist_matrix" + stem
            output_dir = json_path.parent / ("figures" + stem)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (output_dir / f.__name__).with_suffix(".png")

            plot_from_dict(
                results[seed],
                output_path=output_path,
                generalization_key=generalization_key,
                complexity_key=complexity_name,
                ylabel=f.__name__.replace("_", " ")
            )

if __name__ == "__main__":
    fire.Fire(compute_all_quantities_one_seed)
            
