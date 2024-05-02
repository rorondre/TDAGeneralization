import json
from pathlib import Path
from typing import List

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from analysis.plot_magnitude_simple import magnitude, positive_magnitude


def main(
    results_path: str,
    n: int = 50000,
    t: int = None,
    distance_type: int = 1,
    stem: str = "",
    first_seed: bool = True,
    plot: bool = True
    ):

    results_path = Path(results_path)
    result_dir = results_path.parent

    assert results_path.exists(), str(results_path)
    with open(str(results_path), "r") as results_file:
        results = json.load(results_file)

    assert distance_type in [0,1,2], distance_type

    # For now we only work with the first seed
    if not first_seed:
        raise NotImplementedError("Several seeds support not implemented yet")
    seed = list(results.keys())[0]
    logger.warning(f"Only using the first seed {seed}")

    results_seed = results[seed]
    logger.info(f"Found {len(results_seed.keys())} experiments")

    magnitudes = []
    positive_magnitudes = []

    # Loop over all results
    for key in tqdm(list(results_seed.keys())):

        # pp = results_seed[key]["pseudo_matrix_data_proportion"]
        pp = 0.1
        
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

        if t is None:
            t = np.sqrt(n)

        # magnitude
        magnitudes.append(magnitude(dist_matrix, t=t))

        # positive magnitude
        positive_magnitudes.append(positive_magnitude(dist_matrix, t=t))

    # Sorting everything
    magnitudes = np.array(magnitudes)
    positive_magnitudes = np.array(positive_magnitudes)

    indices = np.argsort(magnitudes)
    magnitudes = magnitudes[indices]
    positive_magnitudes = positive_magnitudes[indices]

    if plot:
        plt.figure()

        plt.scatter(magnitudes, positive_magnitudes, marker="x", color="k")
        plt.plot(
            np.linspace(np.min(magnitudes), np.max(magnitudes), 1000),
            np.linspace(np.min(magnitudes), np.max(magnitudes), 1000),
            "--",
            color="r"
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel(r"$\mathrm{PMag}(\sqrt{n}\mathcal{W})$")
        plt.xlabel(r"$\mathrm{Mag}(\sqrt{n}\mathcal{W})$")

        plt.grid(visible=True, which="both")
            

        # Saving the figure
        output_path = results_path.parent / "figures" / f"positive_mag_variation{stem}.png"
        if not output_path.parent.is_dir():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving positive Mag comparison figure in {str(output_path)}")
        plt.savefig(str(output_path))

        plt.close()
    
    else:
        return magnitudes, positive_magnitudes


# HACK This is a bit hacky for now, to be improved
def several_plots(
    results: List[str] = [
        "final_results/2024-04-08_09_51_31/all_results.json",
        "final_results/2024-04-18_17_22_36/all_results.json",
        "final_results/2024-04-29_22_30_35/all_results.json"
    ],
    distance_types: List[int] = [0, 2, 1],
    labels = [
        r"Euclidean distance",
        r"$\ell^2$ pseudometric",
        r"$\ell^1$ pseudometric"
    ],
    colors = ["blue", "orange", "green"],
    stems = ["_euclidean", "", ""]
    ):

    assert len(results) == len(distance_types)

    m = np.inf
    M = -np.inf

    plt.figure()

    for k in tqdm(range(len(results))):

        magnitudes, positive_magnitudes = main(
            results[k],
            distance_type=distance_types[k],
            plot=False,
            stem=stems[k]
        )

        m = min(np.min(magnitudes), m)
        M = max(np.max(positive_magnitudes), M)

        plt.scatter(magnitudes, positive_magnitudes, marker="x", label=labels[k], color=colors[k])

    plt.plot(
        np.linspace(m, M, 1000),
        np.linspace(m, M, 1000),
        "--",
        color="r"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$\mathrm{PMag}(\sqrt{n}\mathcal{W})$")
    plt.xlabel(r"$\mathrm{Mag}(\sqrt{n}\mathcal{W})$")

    plt.legend()

    plt.grid(visible=True, which="both")

    # Saving the figure
    output_path = Path("figures") / f"positive_mag_variation.png"
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving positive Mag comparison figure in {str(output_path)}")
    plt.savefig(str(output_path))

    plt.close()



if __name__ == "__main__":
    # fire.Fire(main)
    fire.Fire(several_plots)


