import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.pyplot import rc
from tqdm import tqdm

from analysis.plot_E_alpha import E_alpha
from analysis.plot_magnitude_simple import magnitude, positive_magnitude

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)

class UnmergedExperiment(BaseException):
    pass

RESULTS = "all_results"

def main(
    result_dir: str,
    n: int = 50000,
    first_seed: bool = True,
    alpha: float = 1.,
    distance_type: int = 1,
    stem: str = ""
):

    result_dir = Path(result_dir)
    results_path = (result_dir / RESULTS).with_suffix(".json")

    if not results_path.exists():
        raise UnmergedExperiment(f"Experiment {result_dir.stem} has not been merged, use analysis/merge_json.py")
    with open(str(results_path), "r") as results_file:
        results = json.load(results_file)

    assert distance_type in [1,2], distance_type

    # For now we only work with the first seed
    if not first_seed:
        raise NotImplementedError("Several seeds support not implemented yet")
    seed = list(results.keys())[0]
    logger.warning(f"Only using the first seed {seed}")

    results_seed = results[seed]
    logger.info(f"Found {len(results_seed.keys())} experiments")

    # We collect values of pseudo_proportion, E_alpha and magnitude
    pseudo_proportions = []
    E_alphas = []
    magnitudes = []

    # Loop over all results
    for key in tqdm(list(results_seed.keys())):

        pp = results_seed[key]["pseudo_matrix_data_proportion"]
        pseudo_proportions.append(pp)
        
        # Loading of the distance matrix
        dist_matrix_path = Path(results_seed[key]["saved_distance_matrix" + stem])
        if not dist_matrix_path.exists():
            raise FileNotFoundError(f"Distance matrix {str(dist_matrix_path)} not found")
        dist_matrix = np.load(str(dist_matrix_path))

        # Normalization
        if distance_type == 1:
            dist_matrix = dist_matrix / (int(pp * n))
        elif distance_type == 2:
            dist_matrix = dist_matrix / np.sqrt(int(pp * n))

        # E_alpha
        E_alpha_comp = E_alpha(dist_matrix, alpha=alpha)
        E_alphas.append(E_alpha_comp)

        # magnitude
        magnitude_comp = positive_magnitude(dist_matrix, t=np.sqrt(n))
        magnitudes.append(magnitude_comp)

        # Writing the new JSON file
        results_seed[key]["E_alpha"] = E_alpha_comp
        results_seed[key]["positive_magnitude"] = magnitude_comp


    # Sorting everything
    pseudo_proportions = np.array(pseudo_proportions)
    magnitudes = np.array(magnitudes)
    E_alphas = np.array(E_alphas)

    indices = np.argsort(pseudo_proportions)
    pseudo_proportions = pseudo_proportions[indices]
    magnitudes = magnitudes[indices]
    E_alphas = E_alphas[indices]

    logger.debug(pseudo_proportions)
    logger.debug(E_alphas)
    logger.debug(magnitudes)

    E_reference = E_alphas[-1] 
    magnitude_reference = magnitudes[-1] 

    # plots
    plt.figure()
    plt.bar(
        100. * pseudo_proportions - 3.,
        100. * (E_alphas - E_reference) / E_reference,
        color = "r",
        label = r"$E_\alpha$",
        width = 3.
    )
    plt.bar(
        100. * pseudo_proportions,
        100. * (magnitudes - magnitude_reference) / magnitude_reference,
        color = "k",
        label = r"$\mathrm{PMag}(\sqrt{n}\mathcal{W})$",
        width = 3.
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Relative variation (%)")
    plt.xlabel("Proportion of the data (%)")

    # Saving the figure
    output_path = results_path.parent / "figures" / f"batch_experiment{stem}.png"
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving batch experiment figure in {str(output_path)}")
    plt.savefig(str(output_path))

    plt.close()

    # Saving the new JSON file
    with open(str(results_path), "r") as new_json_file:
        json.dump(results, new_json_file)
    logger.info(f"Saved the new JSON file in {str(results_path)}")

if __name__ == "__main__":
    fire.Fire(main)