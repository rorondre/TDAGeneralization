import json
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import rc

MARKERS = ["o", "^", "s", "+", "D", "v", "*", "p"]

plt.style.use("_classic_test_patch")
# plt.style.use("ggplot")

font = {'weight' : 'bold'}
matplotlib.rc('font', **font)

# rc('font', **{'family': 'sans-serif', 'sans-s/erif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

def plot_from_dict(results: dict,
                    output_path: str,
                   generalization_key: str = "acc_gap",
                   complexity_key: str = "magnitude",
                   yscale: str = "log",
                   save: bool = True,
                   colorbar: bool = True,
                   ylabel: str = "Magnitude"
                   ):

    experiments = [k for k in results.keys() if
                   generalization_key in results[k].keys() and
                   complexity_key in results[k].keys()]

    logger.info(f"Found {len(experiments)} converged experiments")

    color_map = plt.cm.get_cmap("viridis_r")

    all_bs = list(set([results[k]["batch_size"] for k in experiments]))

    for b_idx in np.argsort(all_bs):

        b = all_bs[b_idx]
        b_experiments = [k for k in experiments if results[k]["batch_size"] == b]

        dim_tab = [results[k][complexity_key] for k in b_experiments]
        gen_tab = [results[k][generalization_key] for k in b_experiments]
        lr_tab = [results[k]["learning_rate"] for k in b_experiments]

        sc = plt.scatter(
            np.array(gen_tab),
            np.array(dim_tab),
            c=np.array(lr_tab),
            marker=MARKERS[b_idx],
            label=str(b),
            cmap=color_map,
            norm=matplotlib.colors.LogNorm()
        )

    if yscale == "log":
        plt.yscale("log")

    plt.xlabel("Generalization gap", loc="right", fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")

    if colorbar:
        cbar = plt.colorbar(sc)
        cbar.set_label("Learning rate", fontweight="bold")

    plt.grid(visible=True, which="both")
    plt.legend(title="Batch size")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figure in {str(output_path)}")
    plt.savefig(str(output_path))

    plt.close()

    

def plot_from_json(json_path: str,
                   generalization_key: str = "acc_gap",
                   complexity_key: str = "magnitude",
                   scale: str = "log",
                   save: bool = True,
                   colorbar: bool = True,
                   ylabel: str = "Magnitude"
                   ):

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"{str(json_path)} not found.")

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    if save:
        output_path = json_path.parent / f"{json_path.parent.name}_plot.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    seed = list(results.keys())[0]
    logger.warning(f"Only using seed {seed}")
    plot_from_dict(
        results[seed],
        str(output_path),
        generalization_key,
        complexity_key,
        scale,
        save,
        colorbar,
        ylabel
    )


if __name__ == "__main__":
    fire.Fire(plot_from_json)
