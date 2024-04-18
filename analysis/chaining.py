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

from analysis.plot_magnitude_simple import magnitude

DIST_MATRIX = "/home/bdupuis/measure_data_dependent_dimensions/final_results/2024-04-08_09_51_31/441801/distance_matrices/dist_matrix_17_euclidean.npy"

def chaining_magnitude_curve(dist_matrix_path: str=DIST_MATRIX,\
                                t_min: float = 0.1,
                                t_max: float = 100.,
                                n_points: int = 10):

    dist_matrix_path = Path(dist_matrix_path)
    assert dist_matrix_path.exists(), str(dist_matrix_path)

    dist_matrix = np.load(str(dist_matrix_path)) 

    # ts = np.linspace(t_min, t_max, n_points)
    ts = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_points))

    magnitudes = []

    for k in tqdm(range(len(ts))):
        magnitudes.append(magnitude(dist_matrix, t=ts[k]))
    
    logger.debug(magnitudes)
    magnitudes = np.sqrt(np.log(np.array(magnitudes)))

    plt.figure()

    plt.plot(ts, magnitudes, color="k", label="magnitude")
    plt.plot(ts, magnitudes / (ts * ts), label="integrand")

    plt.xscale("log")

    plt.legend()
    plt.grid(visible=True, which="both")

    output_path = "tests_directory/chaining.png"
    if not Path(output_path).parent.is_dir():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved figure in {output_path}")

    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire(chaining_magnitude_curve)

