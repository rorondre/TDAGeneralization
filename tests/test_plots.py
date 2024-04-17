import json
from pathlib import Path

from analysis.plot_E_alpha import plot_E_alpha_all_seed
from analysis.plot_magnitude_simple import plot_magnitude_all_seed
from analysis.plot_ph_dim import plot_ph_dim_all_seed


RESULT_EXAMPLES_JSON = "results_example/2024-03-28_14_53_25/all_results.json"

FIGURES_DIR = Path("results_example/2024-03-28_14_53_25/figures")


def test_plot_E_alpha():

    plot_E_alpha_all_seed(RESULT_EXAMPLES_JSON, stem="")

    assert (FIGURES_DIR / "E_alpha_vs_generalization_error.png").exists()
    assert (FIGURES_DIR / "E_alpha_granulated_kendalls.json").exists()
    assert (FIGURES_DIR.parent / "all_results.json").exists()

    with open(str(FIGURES_DIR.parent / "all_results.json"), "r") as json_file:
        results = json.load(json_file)
        for key1 in results.keys():
            for key2 in results[key1].keys():
                assert "E_alpha" in results[key1][key2].keys()

def test_plot_magnitude():

    plot_magnitude_all_seed(RESULT_EXAMPLES_JSON, stem="")

    assert (FIGURES_DIR / "magnitude_vs_generalization_error.png").exists()
    assert (FIGURES_DIR / "magnitude_granulated_kendalls.json").exists()
    assert (FIGURES_DIR.parent / "all_results.json").exists()

    with open(str(FIGURES_DIR.parent / "all_results.json"), "r") as json_file:
        results = json.load(json_file)
        for key1 in results.keys():
            for key2 in results[key1].keys():
                assert "magnitude" in results[key1][key2].keys()

# def test_plot_ph_dim():

#     plot_ph_dim_all_seed(RESULT_EXAMPLES_JSON, min_points=2, max_points=None)

#     assert (FIGURES_DIR / "ph_dim_vs_generalization_error.png").exists()
#     assert (FIGURES_DIR / "ph_dim_granulated_kendalls.json").exists()
#     assert (FIGURES_DIR.parent / "all_results.json").exists()

#     with open(str(FIGURES_DIR.parent / "all_results.json"), "r") as json_file:
#         results = json.load(json_file)
#         for key1 in results.keys():
#             for key2 in results[key1].keys():
#                 assert "ph_dim" in results[key1][key2].keys()


