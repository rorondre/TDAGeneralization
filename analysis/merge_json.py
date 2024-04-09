import json
from pathlib import Path

import fire
import numpy as np
from loguru import logger
from tqdm import tqdm



def merge_json(result_dir: str):

    result_dir = Path(result_dir)
    assert result_dir.is_dir(), str(result_dir)

    # We create one json file for each seed
    seed_dir_list = [seed for seed in result_dir.glob("*") if seed.is_dir() and "figure" not in seed.stem]
    logger.info(f"Found experiments for {len(seed_dir_list)} seeds")

    final_results = {}

    for seed in tqdm(seed_dir_list):

        # Create the resulting dict
        seed_results = {}

        json_list = [p for p in seed.rglob("*.json") if p.stem.startswith("exp_")]
        n = 0 
        for p in json_list:
            with open(str(p), "r") as json_file:
                res = json.load(json_file)
                res["worst_acc"] = res["worst_acc"]
                seed_results.update({str(n): res})
            n += 1
        final_results.update({str(seed.stem): seed_results})

        output_path = seed / f"results_seed_{seed.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Collecting the results of {len(json_list)} experiments in {str(output_path)}")

        with open(str(output_path), "w") as output_file:
            json.dump(seed_results, output_file, indent=2)

    # all_results
    output_path = result_dir / f"all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting all results in {str(output_path)}")

    with open(str(output_path), "w") as output_file:
        json.dump(final_results, output_file, indent=2)


if __name__ == "__main__":
    fire.Fire(merge_json)

    


