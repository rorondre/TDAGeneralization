import json
from pathlib import Path
from typing import List

import fire
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from loguru import logger
from matplotlib import rc
from tqdm import tqdm


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use("ggplot")

font = {'weight' : 'bold'}
matplotlib.rc('font', **font)



# Master psi plot

MARKERS = ["x", "o", "*", "^", "s", "+", "D", "v", "p"]
COLORS = ["b", "orange", "g", "r", "k"]

COLORS_KEYS = {
    r"$E_1$": "b",
    r"$\mathrm{Mag}$": "orange",
    r"$\mathrm{dim}_{\mathrm{PH}}$": "g"
}

# SAVE_DIR = Path("figures") / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]
SAVE_DIR = Path("figures") 

# Convention is (bs, lr)
COEFFS = {
    "ViT - CIFAR10": {
        r"$E_1$": [0.69, 1.0],
        r"$\mathrm{Mag}$": [0.62, 0.53],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [-0.6, 1.0]
    },
    "ViT - CIFAR100": {
        r"$E_1$": [-0.66, 1.0],
        r"$\mathrm{Mag}$": [0.42, 0.28],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [0.29, 0.99]
    },
    "CaiT - CIFAR10": {
        r"$E_1$": [0.78, 0.98],
        r"$\mathrm{Mag}$": [0.78, 0.88],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [-0.08, 0.99]
    },
    "CaiT - CIFAR100": {
        r"$E_1$": [-0.32, 0.83],
        r"$\mathrm{Mag}$": [-0.32, 0.63],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [0.31, 0.99]
    },
    "Swin - CIFAR10": {
        r"$E_1$": [0.58, 0.97],
        r"$\mathrm{Mag}$": [0.57, 0.97],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [-0.57, 0.97]
    },
    "GatedGCN - MNIST": {
        r"$E_1$": [0.93, 0.42],
        r"$\mathrm{Mag}$": [0.93, 0.70],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [0.14, -0.14]
    },
    "GraphSage - MNIST": {
        r"$E_1$": [0.76, 0.52],
        r"$\mathrm{Mag}$": [0.76, 0.54],
        r"$\mathrm{dim}_{\mathrm{PH}}$": [0.02, 0.01]
    }
}



def psi_plot(coeffs: dict=COEFFS, save_dir: str=str(SAVE_DIR)):

    keys = list(coeffs.keys())

    marker_keys = {keys[k]: MARKERS[k] for k in range(len(keys))}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    c = 20

    for model_data in keys:

        for comp in coeffs[model_data]:

            if comp == r"$E_1$":
                ax.scatter(coeffs[model_data][comp][0], coeffs[model_data][comp][1],\
                                marker=marker_keys[model_data], color=COLORS_KEYS[comp],
                                label=model_data)
            else:
                ax.scatter(coeffs[model_data][comp][0], coeffs[model_data][comp][1],\
                                marker=marker_keys[model_data], color=COLORS_KEYS[comp])
                

            # Uncomment for anotations

            # r = np.random.uniform()
            # i = 1 if r > 0.5 else -1

            # r = np.random.uniform()
            # j = 1 if r > 0.5 else -1

            # if comp == r"$\mathrm{Mag}$":
            #     ax.annotate(comp, xy=coeffs[model_data][comp], xytext=(-c,-c), 
            #             textcoords='offset points', ha='center', va='bottom',
            #             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
            #                             color='red'))
            #     ax.scatter(coeffs[model_data][comp][0], coeffs[model_data][comp][1],\
            #                 marker=marker_keys[model_data], color="b",
            #                 label=model_data)
            # else:
            #     ax.annotate(comp, xy=coeffs[model_data][comp], xytext=(c,-c), 
            #             textcoords='offset points', ha='center', va='bottom',
            #             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
            #                             color='red'))
            #     ax.scatter(coeffs[model_data][comp][0], coeffs[model_data][comp][1],\
            #                 marker=marker_keys[model_data], color="b")


    ax.set_xlabel(r"$\psi_{\mathrm{bs}}$", fontweight="bold")
    ax.set_ylabel(r"$\psi_{\mathrm{lr}}$", fontweight="bold")

    ax.plot(np.linspace(-1., 1., 100), np.zeros(100), linewidth=2, color="k")
    ax.plot(np.zeros(100), np.linspace(-1., 1., 100), linewidth=2, color="k")

    ax.grid(True)

    leg1 = ax.legend(loc="lower left", title = "Model - dataset")

    patchList = []
    for key in COLORS_KEYS:
            data_key = mpatches.Patch(color=COLORS_KEYS[key], label=key)
            patchList.append(data_key)

    ax.legend(handles=patchList, loc = "lower right", title="Topological quantities")

    ax.add_artist(leg1)

    # ax2 = plt.subplot(221)
    # plt.plot([0,1])
    # ax.patch.set_facecolor('gray')
    # ax.patch.set_alpha(0.2)

    # Create a Rectangle patch
    rect = matplotlib.patches.Rectangle([0.2, 0.2], 0.85, 0.85, \
                            linewidth=1, edgecolor='r', facecolor='none', linestyle="--")

    # Add the patch to the Axes
    ax.add_patch(rect)

    save_path = Path(save_dir) / "psi_plot.png"
    if not save_path.parent.is_dir():
        save_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(str(save_path))
    logger.info(f"Saved psi plot in {str(save_path)}")
    plt.close()



if __name__ == "__main__":
    fire.Fire(psi_plot)









# Plot with several quantities on the same plot
# def plot_all_quantities(
#         json_path: str,
#         generalization_key: str = "worst_gen",
#         complexities: list = [
#             "positive_magnitude_dist_matrix",
#             "E_alpha_dist_matrix"
#         ],
#         labels: list = [
#             r"$\mathrm{PMag}(\sqrt{n})$",
#             r"$E_1$",
#         ]
#     ):


