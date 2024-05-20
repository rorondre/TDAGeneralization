import json
from pathlib import Path

import fire
import numpy as np
from loguru import logger

from plot_E_alpha import E_alpha
from plot_magnitude_simple import magnitude, positive_magnitude
from plot_ph_dim import ph_dim


def main(distance_matrix_path: str, alpha: float = 1., t: float = np.sqrt(50000)):

    distance_matrix_path = Path(distance_matrix_path)
    assert distance_matrix_path.exists(), str(distance_matrix_path)

    dist_matrix = np.load(str(distance_matrix_path))

    results = {}

    logger.info(f"Computing alpha weighted lifetime sums with alpha={alpha}")
    results["E_alpha"] = float(E_alpha(dist_matrix))

    logger.info(f"Computing magnitude with t={t}")
    results["magnitude"] = float(magnitude(dist_matrix))

    logger.info(f"Computing positive magnitude with t={t}")
    results["positive_magnitude"] = float(positive_magnitude(dist_matrix))

    logger.info("Computing PH dimension")
    results["ph_dim"] = float(ph_dim(dist_matrix))

    logger.info("Results" + json.dumps(results, indent=2))

    save_path = distance_matrix_path.parent / "topological_complexities.json"

    with open(str(save_path), "w") as json_file:
        json.dump(results, json_file, indent=2)
    logger.info(f"Saved the topological complexities in {str(save_path)}")

if __name__ == "__main__":
    fire.Fire(main)