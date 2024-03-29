import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.simulation import asymptotic_constant


def plot_bound(gen_tab, bound_tab, output_dir: str,
                sigma_values, alpha_values,
                    log_scale: bool = True, stem: str="",
                    xlabel:str="Accuracy gap (%)"):
    
    output_dir = Path(output_dir)

    gen_tab = np.array(gen_tab)
    bound_tab = np.array(bound_tab)
    ids = np.where(gen_tab > 0)[0]
    gen_tab = gen_tab[ids]
    bound_tab = bound_tab[ids]
    sigma_values = np.array(sigma_values)[ids]
    alpha_values = np.array(alpha_values)[ids]

    # Colormap
    color_map = plt.cm.get_cmap('viridis_r')

    plt.figure(figsize=(9,5))
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")

    output_path = (output_dir / ("estimated bound versus generalization_alpha_"  + stem )).with_suffix(".png")
    color_map = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(gen_tab,
                     bound_tab,
                     c=alpha_values,
                     cmap=color_map)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\mathbf{\alpha}$")

    plt.xlabel(xlabel, weight="bold")
    plt.ylabel("Estimated bound", weight="bold")
    logger.info(f"Saving a bound plot in {str(output_path)}")
    plt.grid()
    plt.savefig(str(output_path), bbox_inches='tight')
    plt.close()


def plot_one_seed(gen_grid, sigma_tab, alpha_tab, output_dir: str, deviation_grid = None):

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    n_sigma = len(sigma_tab)
    n_alpha = len(alpha_tab)

    logger.info(f"Saving all figures in {output_dir}")

    # Actually, in our case, sigma is n_params, but it works the same
    for s in tqdm(range(n_sigma)):

        plt.figure()
        if deviation_grid is not None:
            plt.plot(alpha_tab, gen_grid[s, :], color="r")
            plt.fill_between(alpha_tab, 
                             gen_grid[s, :] - deviation_grid[s, :],
                               gen_grid[s, :] + deviation_grid[s, :],
                               color="r",
                               alpha=0.25)
        else:
            plt.scatter(alpha_tab, gen_grid[s, :])          

        # Saving the figure
        fig_name = (f"{sigma_tab[s]}").replace(".","_")
        output_path = (output_dir / fig_name).with_suffix(".png")
        plt.savefig(str(output_path))
        plt.close()

    for a in tqdm(range(n_alpha)):

            plt.figure()
            if deviation_grid is not None:
                plt.plot(alpha_tab, gen_grid[:, a], color="r")
                plt.fill_between(alpha_tab, 
                                gen_grid[:, a] - deviation_grid[:, a],
                                gen_grid[:, a] + deviation_grid[:, a],
                                color="r",
                                alpha=0.25)
            else:
                plt.scatter(sigma_tab, gen_grid[:, a])
            plt.xscale("log")        

            # Saving the figure
            fig_name = (f"alpha_{alpha_tab[a]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))
            plt.close()
    

def plot_gen_dim(json_path: str):
    """
    take the average result path and plot gen / dim
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    n_params_ids = {}
    i = 0
    for key in results.keys():
        if results[key]["n_params"] not in n_params_ids.keys():
            n_params_ids[results[key]["n_params"]] = i
            i += 1
    
    n_alpha = 1 + max(results[k]["id_alpha"] for k in results.keys())
    n_width = 1 + max(n_params_ids[k] for k in n_params_ids.keys())

    alpha_tab = np.zeros(n_alpha)
    n_params_tab = np.zeros(n_width)
    acc_gen_grid = np.zeros((n_width, n_alpha))
    acc_gen_grid_deviation = np.zeros((n_width, n_alpha))

    example_key = list(results.keys())[0]
    deviations: bool = ("acc_generalization_deviation" in list(results[example_key].keys()))

    for k in tqdm(results.keys()):

        n_params_tab[n_params_ids[results[k]["n_params"]]] = results[k]["n_params"]
        alpha_tab[results[k]["id_alpha"]] = results[k]["alpha"]

        acc_gen_grid[
            n_params_ids[results[k]["n_params"]],
            results[k]["id_alpha"]
        ] = results[k]["acc_generalization"]

        if deviations:
            acc_gen_grid_deviation[
                    n_params_ids[results[k]["n_params"]],
                    results[k]["id_alpha"]
                ] = results[k]["acc_generalization_deviation"]

    # Plot everything
    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    plot_one_seed(acc_gen_grid, n_params_tab, alpha_tab, str(output_dir), acc_gen_grid_deviation)



def analyze_one_seed(json_path: str, R: float=1.):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    example_key = list(results.keys())[0]
    deviations: bool = ("acc_generalization_deviation" in list(results[example_key].keys()))

    # determine n_width
    d_list = []
    for k in results.keys():
        if results[k]["n_params"] not in d_list:
            d_list.append(results[k]["n_params"])
    n_width = len(d_list)
    
    # Collect n_alpha and n_sigma
    n_sigma = 1 + max(results[k]["id_sigma"] for k in results.keys())
    n_alpha = 1 + max(results[k]["id_alpha"] for k in results.keys())

    # Collect sigma_tab, alpha_tab, and the generalization grids
    sigma_tab = np.zeros(n_sigma)
    sigma_factor_tab = np.zeros(n_sigma)
    alpha_tab = np.zeros(n_alpha)
    normalization_tab = np.zeros(n_alpha)
    acc_gen_grid = np.zeros((n_width, n_alpha))
    gradient_grid = np.zeros((n_width, n_alpha))
    acc_gen_grid_deviation = np.zeros((n_width, n_alpha)) if deviations else None 

    n_params_dict = {} # dict n_params: an index used for plotting
    n_params_idx = 0

    gen_tab = []
    acc_bound_tab = []
    acc_tab = []
    sigma_values = []
    sigma_factor_values = []
    alpha_values = []
    
    for k in results.keys():

        # Estimate the actual value of the bound
        n = results[k]["n"]
        alpha = results[k]["alpha"]
        n_params = results[k]["n_params"]
        sigma = results[k]["sigma"]  # true value, without normalization by the dim
        sigma_factor = results[k]["sigma"] * np.sqrt(n_params)
        gradient = results[k]["gradient_mean"]

        if n_params not in n_params_dict.keys():
            n_params_dict[n_params] = n_params_idx
            n_params_idx += 1


        # TODO: this is ugly and suboptimal, find better
        sigma_tab[results[k]["id_sigma"]] = sigma
        sigma_factor_tab[results[k]["id_sigma"]] = sigma_factor
        alpha_tab[results[k]["id_alpha"]] = results[k]["alpha"]

        acc_gen_grid[
            n_params_dict[n_params],
            results[k]["id_alpha"]
        ] = results[k]["acc_generalization"]

        gradient_grid[
            n_params_dict[n_params],
            results[k]["id_alpha"]
        ] = results[k]["gradient_mean"]

        if deviations:
            acc_gen_grid_deviation[
                    n_params_dict[n_params],
                    results[k]["id_alpha"]
                ] = results[k]["acc_generalization_deviation"]


        # Collect generalization error ad sigma and alpha, for colored plots
        gen_tab.append(results[k]["loss_generalization"])
        acc_tab.append(results[k]["acc_generalization"])
        sigma_values.append(sigma)
        sigma_factor_values.append(sigma_factor)
        alpha_values.append(results[k]["alpha"])

        # in pytorch sgd, decay is the true decay, not post lr
        horizon = results[k]["horizon"] + results[k]["n_ergodic"]
        lr = results[k]["eta"]

        # bs = results[k]["batch_size"]

        constant = asymptotic_constant(alpha, n_params)

        # The value of R can be changed here
        # In the example provided (the one in the main section of the paper),
        # R is estimated to approximately 7
        R = 7

        # The factor two appearing here comes from an optimized PAC-Bayesian generalization bound for bounded losses
        acc_bound_tab.append(100. * np.sqrt((constant * horizon * lr * gradient)/\
                         (2. * n * np.power(R, 2. - alpha) * np.power(sigma, alpha))))


    n_params_tab = list(n_params_dict.keys())
    
    # Plot everything
    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")


    # plot_one_seed(acc_gen_grid, sigma_factor_tab, alpha_tab, str(output_dir), acc_gen_grid_deviation)
    plot_bound(acc_tab, acc_bound_tab, output_dir, sigma_factor_values,\
                alpha_values, log_scale=False, stem=f"accuracy_R_{R}")


def main(json_path: str):

    analyze_one_seed(json_path)
    plot_gen_dim(json_path)


if __name__ == "__main__":
     fire.Fire(main)










    

    