import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
from loguru import logger
from matplotlib import ticker
from sklearn import datasets, manifold
from tqdm import tqdm

def plot_3d(points, points_color):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    # fig.suptitle(title, size=16)
    # col = ax.plot(x, y, z)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    # fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    # ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())



def plot_trajectory(results_path: str, keys: list = ["17"], stem = ""):

    results_path = Path(results_path)
    assert results_path.exists(), str(results_path)

    with open(str(results_path), "r") as json_file:
        results = json.load(json_file)
    
    seed = list(results.keys())[0]
    logger.warning(f"Using only the first seed {seed}")

    if keys is not None:
        assert all([key in results[seed].keys() for key in keys]), (keys, list(results[seed].keys()))
    else:
        keys = list(results[seed].keys())
    logger.info(f"Performing MDS on {len(keys)} distance matrices.")

    save_path = results_path.parent / ("trajectories" + stem)
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    for key in tqdm(keys):

        dist_matrix_path = Path(results[seed][key]["saved_distance_matrix" + stem])
        assert dist_matrix_path.exists(), str(dist_matrix_path)

        dist_matrix = np.load(str(dist_matrix_path))

        # TODO: is this the right way to do it?
        mds = manifold.MDS(n_components=3, metric=False, dissimilarity = "precomputed")
        trajectory = mds.fit_transform(dist_matrix)

        assert trajectory.ndim == 2, trajectory.shape
        assert trajectory.shape == (dist_matrix.shape[0], 3)

        trajectory_path = (save_path / ("trajectory_" + str(key))).with_suffix(".png")

        plt.figure()
        plot_3d(trajectory, np.linspace(0., 1., dist_matrix.shape[0]))
        plt.savefig(trajectory_path)
        plt.close()

    logger.info(f"All results saved in {str(save_path)}")


if __name__ == "__main__":
    fire.Fire(plot_trajectory)

        


