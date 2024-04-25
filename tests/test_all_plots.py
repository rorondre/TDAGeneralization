import json
from pathlib import Path

from analysis.all_plots import compute_all_quantities_one_seed
from analysis.plot_E_alpha import E_alpha
from analysis.plot_magnitude_simple import magnitude


RESULT_EXAMPLES_JSON = Path("results_example/2024-03-28_14_53_25/all_results.json")

FIGURES_DIR = Path("results_example/2024-03-28_14_53_25/figures")


def test_all_plots():

    functions_list = [E_alpha, magnitude]

    compute_all_quantities_one_seed(
        str(RESULT_EXAMPLES_JSON),
        stems = [""],
        generalization_key = "acc_gap",
        functions = functions_list
    ) 

    assert (FIGURES_DIR / "E_alpha.png").exists(), \
                str(FIGURES_DIR / "E_alpha.png")
    assert (FIGURES_DIR / "magnitude.png").exists(), \
                str(FIGURES_DIR / "magnitude.png")

    assert (RESULT_EXAMPLES_JSON.parent / "all_results_augmented.json").exists(), \
                str(RESULT_EXAMPLES_JSON.parent / "all_results_augmented.json")
    assert (RESULT_EXAMPLES_JSON.parent / "granulated_kendalls.json").exists(), \
                str(RESULT_EXAMPLES_JSON.parent / "granulated_kendalls.json")
    assert (RESULT_EXAMPLES_JSON.parent / "granulated_pearsons.json").exists(), \
                str(RESULT_EXAMPLES_JSON.parent / "granulated_pearsons.json")