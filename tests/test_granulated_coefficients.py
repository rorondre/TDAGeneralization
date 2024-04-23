import json
from pathlib import Path

from analysis.kendall import granulated_kendall_from_dict
from analysis.pearson import granulated_pearson_from_dict

JSON_PATH = Path("results_example/2024-03-28_14_53_25/all_results_augmented.json")

KENDALL_KEYS = [
    "granulated Kendalls",
    "Kendall tau"
]

PEARSON_KEYS = [
    "granulated pearsons",
    "pearson tau"
]

def test_kendall():

    assert JSON_PATH.exists(), str(JSON_PATH)
    
    with open(str(JSON_PATH), "r") as json_file:
        results = json.load(json_file)
    
    # we only test on the first seed
    results_seed = results[list(results.keys())[0]]

    granulated_coefficients = granulated_kendall_from_dict(
        results_seed,
        generalization_key = "acc_gap",
        hyperparameters_keys = ["batch_size", "learning_rate"],
        complexity_keys = ["E_alpha"]
    )

    assert all(key in granulated_coefficients.keys() \
                for key in KENDALL_KEYS)


def test_pearson():

    assert JSON_PATH.exists(), str(JSON_PATH)
    
    with open(str(JSON_PATH), "r") as json_file:
        results = json.load(json_file)
    
    # we only test on the first seed
    results_seed = results[list(results.keys())[0]]

    granulated_coefficients = granulated_pearson_from_dict(
        results_seed,
        generalization_key = "acc_gap",
        hyperparameters_keys = ["batch_size", "learning_rate"],
        complexity_keys = ["E_alpha"]
    )

    assert all(key in granulated_coefficients.keys() \
                for key in PEARSON_KEYS)

    


