from collections import deque
from pathlib import Path
import os
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from models import alexnet, fc_mnist, vgg, fc_cifar, lenet, vit
from models.build_model import create_model

from models.vit import VisionTransformer
from models.swin import SwinTransformer
from functools import partial

from models.vgg import vgg as make_vgg
from topology import fast_ripser
from utils import accuracy, LabelSmoothingCrossEntropy
from PHDim.dataset import get_data_simple, preprocess_data_from_frozen_net
from PHDim.eval import eval, recover_eval_tensors, eval_on_tensors

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.random_projection import SparseRandomProjection


# from utils import LabelSmoothingCrossEntropy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"


class UnknownWeightFormatError(BaseException):
    ...


def get_weights(net):
    with torch.no_grad():
        w = []

        # TODO: improve this?
        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w).cpu().numpy()


def main(
        iterations: int = 10000,
         batch_size_train: int = 100,
         batch_size_eval: int = 100,
         lr: float = 1.e-1,
         eval_freq: int = 1000,
         dataset: str = "cifar10",
         data_path: str = "~/data/",
         model: str = "swin",
         save_folder: str = "results",
         depth: int = 5,
         width: int = 50,
         optim: str = "SGD",
         min_points: int = 200,
         seed: int = 42,
         save_weights_file: str = None,
         save_distance_matrix_file: str = None,
         compute_dimensions: bool = True,
         # weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/best_weights/checkpoint_could_be_it.pth',
         # weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/best_weights/vit_cifar100_check.pth',
        weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/best_weights/swin_cifar10_properly_pretrained.pth',
        ripser_points: int = 500,
         jump: int = 20,
         additional_dimensions: bool = False,
         data_proportion: float = 1.,
         proportion_eval: int = 500,
         id_lr: int = 0,
         pseudo_matrix_data_proportion: float = 0.01,
         freeze: bool=False,
         compute_euclidean_dimension: bool=False,
         worst_case_gen_freq: int = None,
         JL_projection: float = None):


    # Creating files to save results
    save_folder = Path(save_folder)
    assert save_folder.exists(), str(save_folder)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"on device {str(device)}")
    logger.info(f"Random seed ('torch.manual_seed'): {seed}")
    print('The seed is ', seed)
    torch.manual_seed(seed)

    # training setup
    if dataset not in ["mnist", "cifar10", "cifar100"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10', 'cifar100']")
    train_loader, test_loader_eval,\
        train_loader_eval, num_classes = get_data_simple(dataset,
                                                         data_path,
                                                         batch_size_train,
                                                         batch_size_eval,
                                                         subset=data_proportion)

    # Number of datapoints
    data_number = len(train_loader.sampler)
    logger.debug(f"number of data points: {data_number}")

    # The max is a hack to have the pytest pass and run fast
    pseudo_matrix_data_number = max(1,int(data_number * pseudo_matrix_data_proportion))
    logger.info(f"Pseudo matrix data number: {pseudo_matrix_data_number}")

    # Some params for CIFAR:
    BATCH_NORM = False
    SCALE = 64

    # TODO: use the args here
    if model not in ["fc", "alexnet", "vgg", "lenet", "vit", "swin"]:
        raise NotImplementedError(f"Model {model} not implemented, should be in ['fc', 'alexnet', 'vgg', 'lenet', 'vit', 'swin']")
    if model == 'fc':
        if dataset == 'mnist':
            input_size = 28**2
            net = fc_mnist(input_dim=input_size, width=width, depth=depth, num_classes=num_classes).to(device)
        elif dataset == 'cifar10':
            net = fc_cifar().to(device)
    elif model == 'alexnet':
        if dataset == 'mnist':
            net = alexnet(input_height=28, input_width=28, input_channels=1, num_classes=num_classes).to(device)
        else:
            net = alexnet(ch=SCALE, num_classes=num_classes).to(device)
    elif model == 'vgg':
        net = make_vgg(depth=depth, num_classes=num_classes, batch_norm=BATCH_NORM).to(device)
    elif model == "lenet":
        if dataset == "mnist":
            net = lenet(input_channels=1, height=28, width=28).to(device)
        else:
            net = lenet().to(device)
    elif model == "vit":
        if dataset == 'cifar10':
            # Specific parameters for vit and Cifar10 -- to be added to args
            n_classes = 10
            img_size = 32
            # default values, this can be done in a better way
            vit_mlp_ratio = 2 # MLP layers in the transformer encoder'
            sd = 0.1 #rate of stochastic depth
            # model = create_model(img_size, n_classes)
            patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8: patch size for ViT
            net = VisionTransformer(img_size=[img_size],
                                      patch_size=patch_size,
                                      in_chans=3,
                                      num_classes=n_classes,
                                      embed_dim=192,
                                      depth=9,
                                      num_heads=12,
                                      mlp_ratio=vit_mlp_ratio,
                                      qkv_bias=True,
                                      drop_path_rate=sd,
                                      norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
        elif dataset == 'cifar100':
            # Specific parameters for CIFAR100
            n_classes = 100
            img_size = 32
            # default values, this can be done in a better way
            vit_mlp_ratio = 2 # MLP layers in the transformer encoder'
            sd = 0.1 #rate of stochastic depth
            # model = create_model(img_size, n_classes)
            patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8: patch size for ViT
            net = VisionTransformer(img_size=[img_size],
                                      patch_size=patch_size,
                                      in_chans=3,
                                      num_classes=n_classes,
                                      embed_dim=192,
                                      depth=9,
                                      num_heads=12,
                                      mlp_ratio=vit_mlp_ratio,
                                      qkv_bias=True,
                                      drop_path_rate=sd,
                                      norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
        else:
            raise NotImplementedError("Only CIFAR10 and CIFAR100 are supported with ViT for now.")

    elif model == "swin":
        if dataset == 'cifar10':
            # Specific parameters for vit and Cifar10 -- to be added to args
            n_classes = 10
            img_size = 32
            mlp_ratio = 2
            window_size = 4
            sd = 0.1  # rate of stochastic depth
            patch_size = 2 if img_size == 32 else 4

            net = SwinTransformer(img_size=img_size,
                                  window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4],
                                  num_heads=[3, 6, 12], num_classes=n_classes,
                                  mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=sd).to(device)
        else:
            raise NotImplementedError("Only CIFAR10 is supported with Swin for now.")

    else:
        NotImplementedError("Model architecture not implemented . . .")

    logger.info("Network: ")
    print(net)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")
    print(n_parameters)

    if weight_file is not None:
        logger.info(f"Loading weights from {str(weight_file)}")
        if Path(weight_file).suffix == ".pth":
            if torch.cuda.is_available():
                logger.warning("Loading weights on GPU")
                net.load_state_dict(torch.load(weight_file))
            else:
                logger.warning("Loading weights on CPU")
                net.load_state_dict(torch.load(weight_file), map_location=torch.device("cpu"))
        elif Path(weight_file).suffix == ".pyT":
            net = torch.load(str(weight_file))
        else:
            raise UnknownWeightFormatError(f"Extension {Path(weight_file).suffix} unknown")

    net = net.to(device)

    # Optional creation of the Gaussian projection matrix deduced from Johnson-Lindenstrauss lemma
    # This is based on sparse projections
    if JL_projection is not None and JL_projection > 0. and JL_projection < 1.:
        logger.info(f"Defining random projections with relative variation of {round(100. * JL_projection, 2)}%")
        random_projection = SparseRandomProjection(eps = JL_projection)
        random_projection.fit(csr_matrix((ripser_points, n_parameters)))
        logger.info(f"Number of projected components: {random_projection.n_components_}")

    if freeze and model == "vit":
        logger.info("Freezing all network except head")
        logger.info("Computing all embeddings...")
        train_loader = preprocess_data_from_frozen_net(
            lambda x: net.get_before_head(x),
            train_loader,
            shuffle=True
        )
        train_loader_eval = preprocess_data_from_frozen_net(
            lambda x: net.get_before_head(x),
            train_loader_eval,
            shuffle=False
        )
        test_loader_eval = preprocess_data_from_frozen_net(
            lambda x: net.get_before_head(x),
            test_loader_eval,
            shuffle=False
        )
        logger.info("Embedding computed on all dataloaders ✅")

        net = net.head.to(device)

    logger.info("Network: ")
    print(net)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")

    crit = nn.CrossEntropyLoss(weight=None, size_average=None,
                               ignore_index=- 100, reduce=None,
                               reduction='mean', label_smoothing=0.1).to(device)
    crit_unreduced = nn.CrossEntropyLoss(weight=None, size_average=None,
                                         ignore_index=- 100, reduce=None,
                                         reduction='none', label_smoothing=0.1).to(device)

    # Try to get the LabelSmoothingCrossEntropy to work
    if model == 'vit':

        # crit = LabelSmoothingCrossEntropy()
        crit = nn.CrossEntropyLoss(weight=None, size_average=None,
                          ignore_index=- 100, reduce=None,
                          reduction='mean', label_smoothing=0.1).to(device)
        crit_unreduced = nn.CrossEntropyLoss(weight=None, size_average=None,
                          ignore_index=- 100, reduce=None,
                          reduction='none', label_smoothing=0.1).to(device)

    else:
        crit = nn.CrossEntropyLoss().to(device)
        crit_unreduced = nn.CrossEntropyLoss(reduction="none").to(device)



    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)

    # Recovering evaluation tensors (made to speed up the experiment)
    eval_x, eval_y = recover_eval_tensors(train_loader_eval)
    test_x, test_y = recover_eval_tensors(test_loader_eval)

    logger.debug(f"eval shape: {eval_x.shape}")

    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    # training logs per iteration
    training_history = []

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []

    # HACK
    worst_case_acc = [100000]

    # initialize results of the experiment, returned if didn't work
    exp_dict = {}

    # weights
    weights_history = deque([])
    loss_history = deque([])
    loss_01_history = deque([])

    STOP = False  # Do we have enough point for persistent homology
    CONVERGED = False  # has the experiment converged?

    # Defining optimizer 
    opt = getattr(torch.optim, optim)(
        net.parameters(),
        lr=lr,
    )

    logger.info("Starting training")
    for i, (x, y) in enumerate(circ_train_loader):

        # Sequentially running evaluation step
        # first record is at the initial point
        if i % eval_freq == 0 and (not CONVERGED):
            net.eval()
            # Evaluation on validation set
            logger.info(f"Evaluation at iteration {i}")
            te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
            evaluation_history_TEST.append([i, *te_hist])
            logger.info(f"Evaluation on test set at iteration {i} finished ✅, accuracy: {round(te_hist[1], 3)}")

            # Evaluation on training set
            tr_hist, losses, _ = eval(train_loader_eval, net, crit_unreduced, opt)
            logger.info(f"Training accuracy at iteration {i}: {round(tr_hist[1], 3)}%")

            # Stopping criterion based on 100% accuracy
            if (int(tr_hist[1]) == 100) and (CONVERGED is False):
                logger.info(f'All training data is correctly classified in {i} iterations! ✅')
                CONVERGED = True

            loss_train = losses.sum().item()
            logger.info(f"Loss sum at iteration {i}: {loss_train}")

        net.train()

        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # take the step
        opt.step()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])

        if i > iterations:
            CONVERGED = True
            if not compute_dimensions:
                STOP=True

        if CONVERGED:
            # we are reaching here
            net.eval()

            # TODO: clean this

            if compute_dimensions:

                with torch.no_grad():

                    # it fails at this line:
                    # clear cache
                    # torch.cuda.empty_cache()

                    tr_hist, losses, _, losses_01 = eval_on_tensors(eval_x[:pseudo_matrix_data_number,...],\
                                                        eval_y[:pseudo_matrix_data_number,...], 
                                                        net, 
                                                        crit_unreduced, 
                                                        return_01=True)
                    assert losses.ndim == 1, losses.shape
                    assert len(losses) == pseudo_matrix_data_number, (losses.shape, pseudo_matrix_data_number)
                    # tr_hist, losses, _ = eval(train_loader_eval, net, crit_unreduced, opt)
                    evaluation_history_TRAIN.append([i, *tr_hist])

                    loss_history.append(losses.cpu())
                    loss_01_history.append(losses_01.cpu())
                    if compute_euclidean_dimension:
                        if JL_projection is not None and JL_projection > 0. and JL_projection < 1.:
                            weights_history.append(
                                random_projection.transform(get_weights(net)[np.newaxis,...])[0,...]
                                )

                        else:             
                            weights_history.append(get_weights(net))

                    # Validation history --> Useful only if we really compute the worst error
                    # te_hist, _, _ = eval_on_tensors(test_x, test_y, net, crit_unreduced)
                    if worst_case_gen_freq is not None and i % worst_case_gen_freq == 0:
                        # te_hist, _, _ = eval(test_loader_eval, net, crit_unreduced, opt)
                        te_hist, *_ = eval_on_tensors(test_x, test_y, net, crit_unreduced)
                        worst_case_acc.append(te_hist[1].item())
                        logger.info(f"Test accuracy at iteration {i}: {te_hist[1]}")
                    

        # clear cache
        # torch.cuda.empty_cache()

        if (len(loss_history) >= ripser_points) and compute_dimensions:
            STOP = True

        # final evaluation and saving results
        if STOP and CONVERGED:

            # if no convergence, we don't record
            if len(loss_history) < ripser_points - 1:
                logger.warning("Experiment did not converge")
                break

            # Some logging
            with torch.no_grad():
                te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
                tr_hist, *_ = eval(train_loader_eval, net, crit_unreduced, opt)

            evaluation_history_TEST.append([i + 1, *te_hist])
            evaluation_history_TRAIN.append([i + 1, *tr_hist])

            # Turn collected iterates (both weights and losses) into numpy arrays
            if compute_euclidean_dimension:
                weights_history_np = np.stack(tuple(weights_history))
                del weights_history
                assert weights_history_np.shape[0] == ripser_points, (weights_history_np.shape, ripser_points)
                if JL_projection is not None and JL_projection > 0. and JL_projection < 1.:
                    assert weights_history_np.shape[1] == random_projection.n_components_, weights_history_np.shape

            loss_history_np = torch.stack(tuple(loss_history)).cpu().numpy()
            loss_01_history_np = torch.stack(tuple(loss_01_history)).cpu().numpy().astype(np.int32)

            metric = 'euclidean'
            metric_pseudo = 'manhattan'

            if compute_euclidean_dimension:
                distance_matrix_euclidean = pairwise_distances(weights_history_np, metric=metric)
            pseudo_distance_matrix = pairwise_distances(loss_history_np, metric=metric_pseudo)
            pseudo_distance_matrix_01 = pairwise_distances(loss_01_history_np, metric=metric_pseudo)

            test_acc = evaluation_history_TEST[-1][2]
            train_acc = evaluation_history_TRAIN[-1][2]

            if worst_case_gen_freq is not None:
                logger.debug(f"wors_acc: {worst_case_acc}")
                print(worst_case_acc)
                worst_gen = min(worst_case_acc)
            else:
                worst_gen = "non_computed"

            exp_dict = {
                "train_acc": train_acc,
                "eval_acc": test_acc,
                "acc_gap": train_acc - test_acc,
                "loss_gap": te_hist[0] - tr_hist[0],
                "test_loss": te_hist[0],
                "learning_rate": lr,
                "batch_size": int(batch_size_train),
                "LB_ratio": lr / batch_size_train,
                "depth": depth,
                "width": width,
                "model": model,
                "iterations": i,
                "dataset": dataset,
                "worst_acc": worst_gen
            }

            save_distance_matrix_file_01 = (Path(save_distance_matrix_file).parent / \
                                            (Path(save_distance_matrix_file).stem + "_01")).with_suffix(".npy")

            if save_distance_matrix_file is not None:
                logger.info(f"Saving pseudo distance matrix in {str(save_distance_matrix_file)} ✅")
                np.save(str(save_distance_matrix_file), pseudo_distance_matrix)
                exp_dict["saved_distance_matrix"] = str(save_distance_matrix_file)

                logger.info(f"Saving pseudo distance matrix 01 in {str(save_distance_matrix_file_01)} ✅")
                np.save(str(save_distance_matrix_file_01), pseudo_distance_matrix_01)
                exp_dict["saved_distance_matrix_01"] = str(save_distance_matrix_file_01)

                if compute_euclidean_dimension:
                    euclidean_dm_file = (Path(save_distance_matrix_file).parent / \
                                        (Path(save_distance_matrix_file).stem + "_euclidean")).with_suffix(".npy")
                    logger.info(f"Saving Euclidean distance matrix in {str(save_distance_matrix_file)} ✅")
                    np.save(str(euclidean_dm_file), distance_matrix_euclidean)
                    exp_dict["saved_distance_matrix_euclidean"] = str(euclidean_dm_file)
                else:
                    exp_dict["saved_distance_matrix_euclidean"] = "non_computed"
                    logger.info("Not computing Euclidean distance matrix, as specified by arguments")

            else:
                logger.warning("No distance matrix has been saved")

            break

    # Saving weights if specified
    if save_weights_file is not None:
        torch.save({
                'model_state_dict': net.state_dict(),
                'epoch': iterations,
            },
                str(save_weights_file)
            )
        logger.info(f"Saved last weights in {str(save_weights_file)} ✅")
        # torch.save(net.state_dict(), str(save_weights_file))
        exp_dict["saved_weights"] = str(save_weights_file)
    else:
        logger.warning("Weights are not saved, as specified by arguments.")

    # Saving Exp_dict
    exp_dict_path = save_folder / f"exp_lr_{id_lr}_bs_{batch_size_train}.json"
    with open(str(exp_dict_path), 'w') as exp_dict_file:
        json.dump(exp_dict, exp_dict_file, indent=2)

    return exp_dict

if __name__ == '__main__':
    main()
