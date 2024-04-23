import os
import sys
import argparse
from util import Logger, hash_dict
import logging
import os

from PHDim.train_risk_analysis import main as train_risk_analysis


def main(args_):
    """"""

    ''' generate experiment hash and set up redirect of output streams '''
    exp_hash = hash_dict(args_.__dict__)
    # TODO: I removed logs saving, do we need it?
    # if args_.result_dir is not None:
    #     os.makedirs(args_.result_dir, exist_ok=True)
    #     log_file_path = os.path.join(args_.result_dir, '%s.log ' % exp_hash)
    #     logger = Logger(log_file_path)
    #     sys.stdout = logger
    #     sys.stderr = logger

    logger = logging.getLogger("root")

    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()

    logger.addFilter(CheckTypesFilter())

    """"""""""""""""""""""""""""""""""" Hyperparameter selection """""""""""""""""""""""""""""""""""""""""""""""


    # if args_.script == "batch":
    #     executed_function = run_and_save_one_simulation_batch
    # elif args_.script == "full_batch":
    #     executed_function = run_and_save_one_simulation
    # else:
    #     raise NotImplementedError()
    executed_function = train_risk_analysis

    
    executed_function(args_.iterations,
                        args_.batch_size_train,
                        args_.batch_size_eval,
                        args_.lr,
                        args_.eval_freq,
                        args_.dataset,
                        args_.data_path,
                        args_.model,
                        args_.save_folder,
                        args_.depth,
                        args_.width,
                        args_.optim,
                        args_.min_points,
                        args_.seed,
                        args_.save_weights_file,
                        args_.save_distance_matrix_file,
                        args_.compute_dimensions,
                        args_.weight_file,
                        args_.ripser_points,
                        args_.jump,
                        args_.additional_dimensions,
                        args_.data_proportion,
                        args_.proportion_eval,
                        args_.id_lr,
                        args_.pseudo_matrix_data_proportion,
                        args_.freeze,
                        args_.compute_euclidean_dimension,
                        args_.worst_case_gen_freq,
                        args_.JL_projection,
                        args_.compute_softmax_dimension,
                        args_.additional_identifier,
                        args_.pseudo_metric_type
                        )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PH and Magnitude')

    # general args
    # parser.add_argument('--mode', type=str, default="slurm")
    parser.add_argument('--mode', type=str, default="euler_slurm")
    parser.add_argument('--long', type=int, default=0)
    parser.add_argument('--env', type=int, default=0)

    # Parameters specific to this run
    parser.add_argument('--id_lr', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size_train', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_weights_file', type=str, default=None)
    parser.add_argument('--save_distance_matrix_file', type=str)
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--save_folder', type=str)

    # Parameters that are shared among all runs
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--batch_size_eval', type=int, default=200) # Batch size for evaluation, could be increased with good GPUs to speed up
    parser.add_argument('--eval_freq', type=int, default=1000000)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--data_path', type=str, default="~/data/")
    parser.add_argument('--model', type=str, default="vit")
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--min_points', type=int, default=200)
    parser.add_argument('--save_weights', type=int, default=1)
    parser.add_argument('--compute_dimensions', type=int, default=1)
    parser.add_argument('--ripser_points', type=int, default=8000)
    parser.add_argument('--data_proportion', type=float, default=1.)
    parser.add_argument('--pseudo_matrix_data_proportion', type=float, default=0.01)
    parser.add_argument('--freeze', type=int, default=0)
    parser.add_argument('--compute_euclidean_dimension', type=int, default=1)
    parser.add_argument('--worst_case_gen_freq', type=int, default=100) # frequency to compute test accuracy. If you dont want it, use 100000000000
    parser.add_argument('--JL_projection', type=float, default=0.05) 
    parser.add_argument('--compute_softmax_dimension', type=int, default=1) # whether to compute the Euclidean distance matrices
    parser.add_argument('--additional_identifier', type=str, default="")
    parser.add_argument('--pseudo_metric_type', type=str, default="euclidean") # Optimizer


    # Unused arguments, needing to be implemented or removed
    parser.add_argument('--proportion_eval', type=int, default=50000) # proportion to eval the pseudometric, unused at the moment
    parser.add_argument('--jump', type=int, default=1) # unused at the moment
    parser.add_argument('--additional_dimensions', type=int, default=1) # unused at the moment
    

    args = parser.parse_args()
    main(args)
