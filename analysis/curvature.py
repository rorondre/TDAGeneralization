import json
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm
from gph.python import ripser_parallel

from analysis.plot_ph_dim import ph_dim
from analysis.plot_magnitude_simple import magnitude


def curvature(dist_matrix: np.ndarray,
           min_points=200,
           max_points=2000,
           point_jump=200,
           h_dim=0,
           seed: int = 42,
           alpha: float = 1.,
           t_min: float = 0.1,
           t_max: float = 2000.,
           n_points_curvature: int = 20,
           save_path: str = None,
           **kwargs) -> float:

    # We first compute the PH dimension
    logger.info("Computation of the intrinsic dimension...")
    dimension = ph_dim(
        dist_matrix,
        min_points=min_points,
        max_points=max_points,
        point_jump=point_jump,
        alpha=alpha
    )
    logger.info(f"Intrinsic dimension: {dimension}")

    # Then we compute magnitude for several values of t
    ts = np.linspace(t_min, t_max, n_points_curvature)
    magnitudes = []

    logger.info(f"Computation of {n_points_curvature} magnitudes values...")
    for k in tqdm(range(len(ts))):
        magnitudes.append(magnitude(dist_matrix, t=ts[k]))

    # Finall we perform the polynomial regression
    data_matrix = np.stack([
        np.ones(len(ts)),
        np.power(ts, dimension - 2),
        np.power(ts, dimension)
    ]).T

    magnitudes = np.array(magnitudes)

    linear_estimator = np.linalg.inv(data_matrix.T @ data_matrix) @ (data_matrix.T @ magnitudes)

    assert linear_estimator.ndim == 1, linear_estimator.shape
    assert len(linear_estimator) == 3, linear_estimator.shape

    result = linear_estimator[1]

    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving magnitude curve in {str(save_path)}")
        
        plt.figure()
        plt.plot(ts, magnitudes, color="r", label="Ground truth")

        polynomial = data_matrix @ linear_estimator
        assert polynomial.shape == ts.shape, polynomial.shape
        plt.plot(ts, polynomial, color="k", label="Fitted polynomial")

        plt.grid(visible=True, which="both")
        plt.title(f"Obtained curvature: {result}")
        plt.legend()
        plt.savefig(str(save_path / "curvature.png"))
        plt.close()

    return result


def main(dist_matrix_path: str, save_path: str = "tests_directory"):

    dist_matrix_path = Path(dist_matrix_path)
    assert dist_matrix_path.exists(), str(dist_matrix_path)

    logger.info(f"Loading distance matrix from {str(dist_matrix_path)}")
    dist_matrix = np.load(str(dist_matrix_path)) / 5000

    result = curvature(dist_matrix, save_path=save_path)
    logger.info(f"curvature: {result}")


if __name__ == "__main__":
    fire.Fire(main)
