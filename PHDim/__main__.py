import datetime
import json
import os
from pathlib import Path
from typing import Optional

import fire
import numpy as np
from loguru import logger
from pydantic import BaseModel

from PHDim.train_risk_analysis import main as risk_analysis


class AnalysisOptions(BaseModel):

    """
    All hyperparameters of the experiement are defined here
    """

    iterations: int = 1 # Number of iterations before convergence (denoted tau in the paper)
    log_weights: bool = False  # Whether we want to save final weights of the experiment
    log_distance_matrix: bool = True # Whether we want to save the distance matrix from each experiment
    batch_size_eval: int = 5000  # batch size used for evaluation
    lrmin: float = 0.005  # minimum learning rate in teh experiment
    lrmax: float = 0.1  # maximum learning rate in the experiment
    bs_min: int = 32  # minimum batch size in the experiment
    bs_max: int = 256  # maximum batch sie in the experiment
    eval_freq: int = 3000  # at which frequency we evaluate the model (training and validation sets)
    dataset: str = "cifar10"  # dataset we use
    data_path: str = "~/data/"  # where to find the data
    model: str = "vit"  # model
    save_folder: str = "./results"  # Where to save the results
    depth: int = 5  # depth of the network (for FCNN)
    width: int = 200  # width of the network (for FCNN)
    optim: str = "Adam"  # Optimizer
    min_points: int = 2  # minimum number of points used to compute the PH dimension
    num_exp_lr: int = 6  # Number of batch sizes we use
    num_exp_bs: int = 6  # Number of learning rates we use
    compute_dimensions: bool = True  # whether or not we compute the PH dimensions
    project_name: str = "ph_and_magnitude"  # project name for WANDB logging
    initial_weights: Optional[str] = None  # Initial weights path if possible
    # ripser_points: int = 5000  # Maximum number of points used to compute the PH dimension
    ripser_points: int = 5000  # Maximum number of points used to compute the PH dimension
    seed: int = 1234  # seed
    jump: int = 20  # number of finite sets drawn to compute the PH dimension, not used at the moment
    additional_dimensions: bool = True  
    data_proportion: float = 1. # Proportion of data to use (between 0 and 1), used for pytests
    JL_projection: float = 0.05 # Proportion of data to use (between 0 and 1), used for pytests
    compute_euclidean_dimension: bool = True
    worst_case_gen_freq: int = 1 # period of evaluation of the test accuracy
    compute_activation_dimension: bool = True
    additional_identifier: str = ""
    pseudo_metric_type: str = "manhattan" # Which type of data-dependent pseudo_distance to use
    pseudo_matrix_data_proportion: float = 0.1 # Proportion of the data that is used to estimate the pseudo distance matrices

    def __call__(self):

        save_folder = Path(self.save_folder)

        exp_folder_temp = save_folder / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]
        exp_folder_temp.mkdir(parents=True, exist_ok=True)

        log_file = exp_folder_temp / "parameters.log.json"
        log_file.touch()

        exp_folder = exp_folder_temp / "results"
        exp_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving log file in {log_file}")
        with open(log_file, "w") as log:
            json.dump(self.dict(), log, indent=2)

        if self.log_weights:
            weights_dir = exp_folder / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            weights_dir = None

        if self.log_distance_matrix:
            dist_matrix_dir = exp_folder / "distance_matrices"
            dist_matrix_dir.mkdir(parents=True, exist_ok=True)
        else:
            dist_matrix_dir = None

        if self.lrmin > self.lrmax:
            raise ValueError(f"lrmin ({self.lrmin}) should be smaller than or equal to lmax ({self.lrmax})")

        # Defining the grid of hyperparameters
        lr_tab = np.exp(np.linspace(np.log(self.lrmin), np.log(self.lrmax), self.num_exp_lr))
        bs_tab = np.linspace(self.bs_min, self.bs_max, self.num_exp_bs, dtype=np.int64)

        experiment_results = {}

        logger.info(f"Launching {self.num_exp_lr * self.num_exp_bs} experiences")

        group = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]
        exp_num = 0
        experiment_results = {}

        for k in range(min(self.num_exp_lr, len(lr_tab))):
            for j in range(min(self.num_exp_bs, len(bs_tab))):

                # Initial weights should be stored in

                if self.log_weights:
                    save_weights_file = weights_dir / f"weights_{exp_num}.pth"
                else:
                    save_weights_file = None

                if self.log_distance_matrix:
                    save_distance_matrix_file = dist_matrix_dir / f"dist_matrix_{exp_num}.npy"
                else:
                    save_distance_matrix_file = None

                # Here the seed is not changed
                logger.info(f"EXPERIENCE NUMBER {k}:{j}")

                exp_dict = risk_analysis(
                    self.iterations,
                    int(bs_tab[j]),
                    self.batch_size_eval,
                    lr_tab[k],
                    self.eval_freq,
                    self.dataset,
                    self.data_path,
                    self.model,
                    str(exp_folder),
                    self.depth,
                    self.width,
                    self.optim,
                    self.min_points,
                    self.seed,
                    save_weights_file,
                    save_distance_matrix_file,
                    self.compute_dimensions,
                    self.initial_weights,
                    ripser_points=self.ripser_points,
                    jump=self.jump,
                    additional_dimensions=self.additional_dimensions,
                    data_proportion=self.data_proportion,
                    proportion_eval=0, # This argument is not used anymore
                    id_lr=0,
                    pseudo_matrix_data_proportion=self.pseudo_matrix_data_proportion,
                    freeze=False,
                    compute_euclidean_dimension=True,
                    JL_projection=self.JL_projection,
                    worst_case_gen_freq=self.worst_case_gen_freq,
                    compute_activation_dimension=self.compute_activation_dimension,
                    additional_identifier=self.additional_identifier,
                    pseudo_metric_type=self.pseudo_metric_type
                )

                experiment_results[exp_num] = exp_dict

                save_path = Path(exp_folder) / f"results_{exp_num}.json"
                with open(str(save_path), "w") as save_file:
                    json.dump(experiment_results, save_file, indent=2)

                # Remove previously saved file
                if exp_num >= 1:
                    if (Path(exp_folder) / f"results_{exp_num - 1}.json").exists():
                        os.remove(Path(exp_folder) / f"results_{exp_num - 1}.json")

                exp_num += 1

        return str(exp_folder)


if __name__ == "__main__":
    fire.Fire(AnalysisOptions)
