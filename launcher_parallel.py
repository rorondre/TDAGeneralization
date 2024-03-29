import datetime
import json
import time

from experiment_utils import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import run_exp
import argparse
import numpy as np
import copy
from pathlib import Path
from loguru import logger


# Path to initial weights
WEIGHT_FILE = "weights/checkpoint_95_3.pth"



# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# TODO: I am not sure if we need those in that case
applicable_configs = {}
default_configs = {}
search_ranges = {}

                       

# TODO: is this useful?
# check consistency of configuration dicts
# assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}


def main(args_):
    
    lrs = list(np.exp(np.linspace(np.log(args_.lr_min), \
                                     np.log(args_.lr_max), \
                                     args_.n_lr)))
    if args_.bs_min < 1:
        bss = [-1 for _ in range(args_.n_bs)]
    else:
        assert args_.bs_min <= args_.bs_max, (args_.bs_min, args_.bs_max)
        bss = list(np.power(2., np.linspace(
            np.log2(args_.bs_min),
            np.log2(args_.bs_max),
            args_.n_bs
        )).astype(np.int64))                        
    
    mode = args_.mode
    rds = np.random.RandomState(args_.seed)
    assert args_.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(args_.num_seeds_per_hparam,)))

    # determine name of experiment
    if not Path(args_.result_dir).is_dir():
        Path(args_.result_dir).mkdir(parents=True, exist_ok=True)
    
    exp_path = Path(args_.result_dir) / args_.date
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)

    arg_path = exp_path / "arguments.json"
    with open(str(arg_path), "w") as exp_file:
        logger.info(f"Saving argments in {str(arg_path)}")
        json.dump(args_.__dict__, exp_file, indent=2)

    command_list = []
    exp_num = 0 # Count the total number of exps that have been launched

    for j in range(args.num_seeds_per_hparam):
    
        for b in range(len(bss)):
            for e in range(len(lrs)):

                # transfer flags from the args
                flags = copy.deepcopy(args.__dict__)
                [flags.pop(key) for key in
                ['num_cpus', 'num_gpus',\
                'num_seeds_per_hparam', 'seed',
                'date',
                "n_lr", "n_bs", "lr_min", "lr_max", "bs_min", "bs_max",
                "save_weights", "result_dir", "weight_file"]]

                # randomly sample flags
                for flag in default_configs:
                    if flag in search_ranges:
                        flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
                    else:
                        flags[flag] = default_configs[flag]

                flags['id_lr'] = e
                flags['lr'] = lrs[e]
                flags['batch_size_train'] = bss[b]
                flags['seed'] = init_seeds[j]

                run_results_folder = exp_path / str(init_seeds[j])
                if not run_results_folder.is_dir():
                    run_results_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"{str(run_results_folder)}")

                weights_dir = run_results_folder / "weights" 
                if args_.save_weights == 1:
                    flags["save_weights_file"] = str(weights_dir / f"weights_{exp_num}.pth")
                if not weights_dir.is_dir():
                    weights_dir.mkdir(parents=True, exist_ok=True)

                dist_matrix_dir = run_results_folder/ "distance_matrices" 
                flags["save_distance_matrix_file"] = str(dist_matrix_dir / f"dist_matrix_{exp_num}.npy")
                if not dist_matrix_dir.is_dir():
                    dist_matrix_dir.mkdir(parents=True, exist_ok=True)

                if args_.weight_file == 1:
                    flags["weight_file"] = WEIGHT_FILE     

                flags['save_folder'] = str(run_results_folder)

                cmd = generate_base_command(run_exp,
                                            flags=dict(**flags))
                command_list.append(cmd)

                exp_num+=1

    # submit jobs
    generate_run_commands(command_list, 
                          num_gpus=args.num_gpus,
                            num_cpus=args.num_cpus, 
                            mode=mode, 
                            promt=False,
                            long=args_.long)

    logger.info(f"Launched a total of {exp_num} experiments")



"""
Test Commmand
PYTHONPATH=$PWD python launcher_parallel.py --num_gpus 0 --n_lr 2 --n_bs 2 --depth 0 --result_dir tests_directory --num_seeds_per_hparam 2 --dataset mnist --model fc --depth 2 --width 10 --iterations 2 --ripser_points 5 --min_points 3 --weight_file 0 --data_proportion 0.001 --n_lr 2 --n_bs 2 --bs_max 64 --bs_min 32
"""

"""
Test Commmand VIT
PYTHONPATH=$PWD python launcher_parallel.py --num_gpus 1 --num_cpus 1 --n_lr 1 --n_bs 1 --result_dir tests_directory --num_seeds_per_hparam 1 --dataset cifar10 --model vit --iterations 2 --ripser_points 3 --min_points 1 --weight_file 1 --data_proportion 0.001 --bs_max 4 --bs_min 2
"""

"""
Test Commmand VIT Informatics cluster
PYTHONPATH=$PWD python launcher_parallel.py --mode informatics_slurm --num_gpus 1 --num_cpus 1 --n_lr 1 --n_bs 1 --result_dir tests_directory --num_seeds_per_hparam 1 --dataset cifar10 --model vit --iterations 2 --ripser_points 3 --min_points 1 --weight_file 1 --data_proportion 0.001 --bs_max 4 --bs_min 2
"""


if __name__ == '__main__':

    SEED = int(str(time.time()).split(".")[1])


    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0])
    parser.add_argument('--mode', type=str, default="euler_slurm")
    parser.add_argument('--long', type=int, default=0) # Whether we do the long or the short simulation
    # Currently short is 8 hours and long is 24 hours

    parser.add_argument('--num_cpus', type=int, default=20)
    parser.add_argument('--num_gpus', type=int, default=3)

    # Variation of learning rate and batch size --> GRID
    parser.add_argument('--lr_min', type=float, default=1.e-6)
    parser.add_argument('--lr_max', type=float, default=1.e-4)
    parser.add_argument('--bs_min', type=int, default=64)
    parser.add_argument('--bs_max', type=int, default=512)
    parser.add_argument('--n_lr', type=int, default=4)
    parser.add_argument('--n_bs', type=int, default=4)

    # Parameters which are launcher specific
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=1)    

    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)

    # Parameters that are shared among all runs
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--batch_size_eval', type=int, default=256) # Batch size for evaluation, could be increased with good GPUs to speed up
    parser.add_argument('--eval_freq', type=int, default=1000000)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--data_path', type=str, default="~/data/")
    parser.add_argument('--model', type=str, default="vit")
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--min_points', type=int, default=200)
    parser.add_argument('--save_weights', type=int, default=1)
    parser.add_argument('--compute_dimensions', type=int, default=1)
    parser.add_argument('--weight_file', type=int, default=1)  # Initial weights
    parser.add_argument('--ripser_points', type=int, default=5000)
    parser.add_argument('--data_proportion', type=float, default=1.)
    parser.add_argument('--pseudo_matrix_data_proportion', type=float, default=0.01)
    
    
    # Unused arguments, needing to be implemented or removed
    parser.add_argument('--proportion_eval', type=int, default=50000) # proportion to eval the pseudometric, unused at the moment
    parser.add_argument('--jump', type=int, default=1) # unused at the moment
    parser.add_argument('--additional_dimensions', type=int, default=1) # unused at the moment


    args = parser.parse_args()
    main(args)
