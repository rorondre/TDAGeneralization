import json
import itertools
from pathlib import Path
from typing import List

import fire
import numpy as np
from loguru import logger
from scipy.stats import pearsonr





def granulated_pearson_from_dict(results: dict,
                            generalization_key: str = "acc_gap",
                            hyperparameters_keys: List[str] = ["batch_size", "learning_rate"],
                            complexity_keys: str = ["E_alpha"]):
    # resutls should be a seed results dict

    experiments = [k for k in results.keys() if len(results[k].keys()) > 0]
    logger.info(f"Found {len(experiments)} converged experiments")

    gen_tab = np.array([results[key][generalization_key] for key in experiments])
    unique_hyp_dict = {
        hyp_key: np.unique(np.array([results[key][hyp_key] for key in experiments]))
        for hyp_key in hyperparameters_keys
    }

    ## Computation of true pearson's tau
    whole_pearson = {}
    for comp in complexity_keys:
        comp_tab = np.array([results[key][comp] for key in experiments])
        whole_tau = pearsonr(gen_tab, comp_tab).statistic
        whole_pearson[comp] = whole_tau 

    granulated_pearsons = {}

    for complexity in complexity_keys:

        granulated = 0.
        granulated_pearsons[complexity] = {}

        for hyperparameter in hyperparameters_keys:

            logger.info(f"Computing pearson coefficients for complexity {complexity} and hyperparameter {hyperparameter}")

            other_hyperparameters = hyperparameters_keys.copy()
            other_hyperparameters.remove(hyperparameter)
            other_values = [unique_hyp_dict[key] for key in other_hyperparameters]

            cartesian_product = list(itertools.product(*other_values))
            m = len(cartesian_product)

            pearson = 0.

            for other_hyp_set in cartesian_product:

                gen_tab = [results[key][generalization_key] for key in experiments
                           if all([results[key][hyp] == other_hyp_set[i]
                                   for i, hyp in enumerate(other_hyperparameters)])]

                complexity_tab = [results[key][complexity] for key in experiments
                                  if all([results[key][hyp] == other_hyp_set[i]
                                          for i, hyp in enumerate(other_hyperparameters)])]

                ktau = pearsonr(gen_tab, complexity_tab)
                pearson += ktau.statistic / m
                # TODO: the pearsonr function computes p-values, they should be logged in as well

            granulated_pearsons[complexity][hyperparameter] = pearson
            granulated += pearson

        granulated_pearsons[complexity]["average granulated pearson coefficient"] = granulated /\
            len(hyperparameters_keys)

    final_results = {
        "granulated pearsons": granulated_pearsons,
        "pearson tau": whole_pearson
    }

    logger.info(f"Results: \n {json.dumps(final_results, indent=2)}")

    return final_results



# Unused at the moment
def granulated_pearson_from_json(json_path: str,
                                 generalization_key: str = "loss_gap",
                                 complexity_keys: List[str] = ["ph_dim_euclidean", "ph_dim_losses_based"],
                                 hyperparameters_keys: List[str] = ["batch_size", "learning_rate"]):

    if not Path(json_path).exists():
        raise FileNotFoundError()

    with open(json_path, "r") as json_file:
        results = json.load(json_file)
    experiments = [k for k in results.keys() if len(results[k].keys()) > 0]
    logger.info(f"Found {len(experiments)} converged experiments")

    gen_tab = np.array([results[key][generalization_key] for key in experiments])
    unique_hyp_dict = {
        hyp_key: np.unique(np.array([results[key][hyp_key] for key in experiments]))
        for hyp_key in hyperparameters_keys
    }

    ## Computation of true pearson's tau
    whole_pearson = {}
    for comp in complexity_keys:
        comp_tab = np.array([results[key][comp] for key in experiments])
        whole_tau = pearsonr(gen_tab, comp_tab).statistic
        whole_pearson[comp] = whole_tau 

    granulated_pearsons = {}


    for complexity in complexity_keys:

        granulated = 0.
        granulated_pearsons[complexity] = {}

        for hyperparameter in hyperparameters_keys:

            logger.info(f"Computing pearson coefficients for complexity {complexity} and hyperparameter {hyperparameter}")

            other_hyperparameters = hyperparameters_keys.copy()
            other_hyperparameters.remove(hyperparameter)
            other_values = [unique_hyp_dict[key] for key in other_hyperparameters]

            cartesian_product = list(itertools.product(*other_values))
            m = len(cartesian_product)

            pearson = 0.

            for other_hyp_set in cartesian_product:

                gen_tab = [results[key][generalization_key] for key in experiments
                           if all([results[key][hyp] == other_hyp_set[i]
                                   for i, hyp in enumerate(other_hyperparameters)])]

                complexity_tab = [results[key][complexity] for key in experiments
                                  if all([results[key][hyp] == other_hyp_set[i]
                                          for i, hyp in enumerate(other_hyperparameters)])]

                ktau = pearsonr(gen_tab, complexity_tab)
                pearson += ktau.statistic / m
                # TODO: the pearsonr function computes p-values, they should be logged in as well

            granulated_pearsons[complexity][hyperparameter] = pearson
            granulated += pearson

        granulated_pearsons[complexity]["average granulated pearson coefficient"] = granulated /\
            len(hyperparameters_keys)

    # logger.info(f"Results: \n {json.dumps(granulated_pearsons, indent=2)}")

    output_path = Path(json_path).parent / "granulated_pearsons.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results in {str(output_path)}")

    final_results = {
        "granulated pearsons": granulated_pearsons,
        "pearson tau": whole_pearson
    }

    with open(str(output_path), "w") as output_file:
        json.dump(final_results, output_file, indent=2)

    logger.info(f"Results: \n {json.dumps(final_results, indent=2)}")

    return final_results



if __name__ == "__main__":
    fire.Fire(granulated_pearson_from_json)
