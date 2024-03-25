from collections import deque
from pathlib import Path
import os
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger

from models import alexnet, fc_mnist, vgg, fc_cifar, lenet, vit
from models.build_model import create_model

from models.vit import VisionTransformer
from functools import partial

from models.vgg import vgg as make_vgg
from topology import fast_ripser
from utils import accuracy, LabelSmoothingCrossEntropy
from PHDim.dataset import get_data_simple
from PHDim.eval import eval, recover_eval_tensors, eval_on_tensors

from sklearn.metrics.pairwise import pairwise_distances

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
         model: str = "vit",
         save_folder: str = "results",
         depth: int = 5,
         width: int = 50,
         optim: str = "SGD",
         min_points: int = 200,
         seed: int = 42,
         save_weights_file: str = None,
         save_distance_matrix_file: str = None,
         compute_dimensions: bool = True,
         # weight_file: str = None,
         #weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/model_weights/vit_cifar10/vit_cifar10_patch4_input32.pth',
         #weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/vit-vit-cifar10/checkpoint_trial_convereged_weights_with_pretraining.pth',
         #weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/vit-vit-cifar10/checkpoint_trial_convereged_weights.pth',
         #weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/vit-vit-cifar10/checkpoint_trial_convereged_weights_with_pretraining_small_test_3000_it.pth',

         # Training accuracy of 100 produced with the following weights, after 76000 iterations
         #weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/vit-vit-cifar10/checkpoint_trial_final_converged.pth',

         #weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/vit-vit-cifar10/checkpoint_trial_final_converged_final_more_than_3000_really_final.pth',

         weight_file: str = 'C:/Users/Rayna/PycharmProjects/measure_data_dependent_dimensions/PHDim/save_finetuned/best_weights/checkpoint_could_be_it.pth',

        # ripser_points: int = 10,
         ripser_points: int = 500,
         jump: int = 20,
         additional_dimensions: bool = False,
         data_proportion: float = 1.,
         # For the evaluation of the pseudometric, unused for the moment
         proportion_eval: int = 500,
         id_lr: int = 0):

    # Creating files to save results
    save_folder = Path(save_folder)
    assert save_folder.exists(), str(save_folder)

    # TODO: make it work with this folder
    # exp_folder = save_folder / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    # exp_folder.mkdir(parents=True, exist_ok=True)

    # model_name = model
    # model_name += f"-{model}-{dataset}"
    # save_path = os.path.join(os.getcwd(), 'save_finetuned', model_name)
    # if save_path:
    #     os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"on device {str(device)}")
    logger.info(f"Random seed ('torch.manual_seed'): {seed}")
    print('The seed is,', seed)
    torch.manual_seed(seed)

    # training setup
    if dataset not in ["mnist", "cifar10"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10']")
    train_loader, test_loader_eval,\
        train_loader_eval, num_classes = get_data_simple(dataset,
                                                         data_path,
                                                         batch_size_train,
                                                         batch_size_eval,
                                                         subset=data_proportion)

    # Some params for CIFAR:
    BATCH_NORM = False
    SCALE = 64

    # TODO: use the args here
    if model not in ["fc", "alexnet", "vgg", "lenet", "vit"]:
        raise NotImplementedError(f"Model {model} not implemented, should be in ['fc', 'alexnet', 'vgg']")
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

        else:
            raise NotImplementedError("Only CIFAR10 is supported with VIT for now.")

    logger.info("Network: ")
    print(net)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Number of params: ")
    print(n_parameters)

    if weight_file is not None:
        logger.info(f"Loading weights from {str(weight_file)}")
        if Path(weight_file).suffix == ".pth":
            print('we want to be here!')
            # model_dict = net.state_dict()
            # print("loading pretrained weights . . .")
            # state_dict = torch.load(weight_file, map_location="cpu")
            # model_state_dict = state_dict['model_state_dict']
            # for key, value in state_dict.items():
            #     print('The keys are')
            #     print(key, value)
            # checkpoint_key = 'student'
            # if checkpoint_key is not None and checkpoint_key in state_dict:
            #     print(f"Take key {checkpoint_key} in provided checkpoint dict")
            #     state_dict = state_dict[checkpoint_key]
            # # remove `module.` prefix
            # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # # remove `backbone.` prefix induced by multicrop wrapper
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in
            #               zip(model_dict.keys(), state_dict.values())}
            # # model.load_state_dict(state_dict, strict=False)
            # net.load_state_dict(state_dict, strict=False)
            # net.load_state_dict(torch.load(str(weight_file)))
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(weight_file))
            else:
                net.load_state_dict(torch.load(weight_file), map_location="cpu")
            # model.load_state_dict(torch.load(PATH))
        elif Path(weight_file).suffix == ".pyT":
            net = torch.load(str(weight_file))
        else:
            raise UnknownWeightFormatError(f"Extension {Path(weight_file).suffix} unknown")

    net = net.to(device)
    crit = nn.CrossEntropyLoss(weight=None, size_average=None,
                               ignore_index=- 100, reduce=None,
                               reduction='mean', label_smoothing=0.1).to(device)
    crit_unreduced = nn.CrossEntropyLoss(weight=None, size_average=None,
                                         ignore_index=- 100, reduce=None,
                                         reduction='none', label_smoothing=0.1).to(device)

    # Try to get the LabelSmoothingCrossEntropy to work
    if model == 'vit':
        print('we are here')
        print('Adding special loss')
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

    # training logs per iteration
    training_history = []

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []

    # initialize results of the experiment, returned if didn't work
    exp_dict = {}

    # weights
    # weights_history = deque([])
    loss_history = deque([])

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
            logger.info(f"Loss sum at iteration{i}: {loss_train}")

        net.train()

        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        if i % 1000 == 0:
            logger.info(f"Loss at iteration {i}: {loss.item()}")

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])

        # take the step
        opt.step()

        if i > iterations:
            CONVERGED = True
            if not compute_dimensions:
                STOP=True

        if CONVERGED:
            # we are reaching here
            net.eval()

            # TODO: clean this
            # we need to save the weights we have converged to so that it is easy to finetune
            checkpoint_name = 'checkpoint_trial'
            # add the training and test accuracy value at the checkpoint
            # checkpoint_name += f"-{round(tr_hist[1], 3)}-{round(te_hist[1], 3)}"
            # weights_path = save_folder / "converged_weights.npy"
            #TODO: get the saving to work with the exp_folder path
            # weights_path = exp_folder / "converged_weights.pth"
            # print('dm_path', weights_path)
            # logger.info(f"Print path to saved weights", weights_path)
            # torch.save({
            #     'model_state_dict': net.state_dict(),
            #     'epoch': iterations,
            #     # 'optimizer_state_dict': optimizer.state_dict(),
            #     # 'scheduler': scheduler.state_dict(),
            # },
            #     os.path.join(save_path, 'checkpoint_trial_further.pth')
            #     # weights_path
            # )

            # weights_history.append(get_weights(net))

            # It might be a good idea to save the weights at this point
            # Saving weights if specified

            # if save_weights_file is not None:
            #     logger.info(f"Saving last weights in {str(save_weights_file)}")
            #     torch.save(net.state_dict(), str(save_weights_file))
            #     exp_dict["saved_weights"] = str(save_weights_file)

            if compute_dimensions:
                # it fails at this line:
                # clear cache
                torch.cuda.empty_cache()
                proportion_eval_x = eval_x[:500,...]
                proportion_eval_y = eval_y[:500,...]

                #tr_hist, losses, _ = eval_on_tensors(proportion_eval_x, proportion_eval_y, net, crit_unreduced)
                tr_hist, losses, _ = eval(train_loader_eval, net, crit_unreduced, opt)
                evaluation_history_TRAIN.append([i, *tr_hist])
                loss_history.append(losses.cpu())

            # Validation history
            #te_hist, _, _ = eval_on_tensors(test_x, test_y, net, crit_unreduced)
            te_hist, _, _ = eval(test_loader_eval, net, crit_unreduced, opt)
        # clear cache
        torch.cuda.empty_cache()

        if (len(loss_history) >= ripser_points) and compute_dimensions:
            STOP = True
            # weights_history.popleft()
            # loss_history.popleft()

        # final evaluation and saving results
        if STOP and CONVERGED:

            # if no convergence, we don't record
            if len(loss_history) < ripser_points - 1:
                logger.warning("Experiment did not converge")
                break

            # Some logging
            logger.debug('eval time {}'.format(i))
            te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
            tr_hist, *_ = eval(train_loader_eval, net, crit_unreduced, opt)

            evaluation_history_TEST.append([i + 1, *te_hist])
            evaluation_history_TRAIN.append([i + 1, *tr_hist])

            # Turn collected iterates (both weights and losses) into numpy arrays
            # if compute_dimensions:
            #     weights_history_np = np.stack(tuple(weights_history))
            #     del weights_history

            loss_history_np = torch.stack(tuple(loss_history)).cpu().numpy()
  

            metric = 'euclidean'
            metric_pseudo = 'manhattan'
            # w = weights_history_np
            w_pseudo = loss_history_np

            # dm = pairwise_distances(w, metric=metric)

            pseudo_distance_matrix = pairwise_distances(w_pseudo, metric=metric_pseudo)

            # np.array(pseudo_distance_matrix)

            # dm_path_folder = exp_folder / 'distance_matrix_folder'
            # dm_path_folder.mkdir(parents=True, exist_ok=True)
            # TODO: make this work with dm_path_folder

            test_acc = evaluation_history_TEST[-1][2]
            train_acc = evaluation_history_TRAIN[-1][2]

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
                "last_losses": f"outputs_loss_{lr}_{batch_size_train}.npy",
                "iterations": i,

            }

            if save_distance_matrix_file is not None:
                logger.info(f"Saving pseudo distance matrix in {str(save_distance_matrix_file)}")
                np.save(str(save_distance_matrix_file), pseudo_distance_matrix)
                exp_dict["saved_distance_matrix"] = str(save_distance_matrix_file)

            break

    # Saving weights if specified
    if save_weights_file is not None:
        torch.save({
                'model_state_dict': net.state_dict(),
                'epoch': iterations,
            },
                str(save_weights_file)
            )
        logger.info(f"Saved last weights in {str(save_weights_file)}")
        # torch.save(net.state_dict(), str(save_weights_file))
        exp_dict["saved_weights"] = str(save_weights_file)

    # Saving Exp_dict
    exp_dict_path = save_folder / f"exp_lr_{id_lr}_bs_{batch_size_train}.json"
    with open(str(exp_dict_path), 'w') as exp_dict_file:
        json.dump(exp_dict, exp_dict_file, indent=2)

    return exp_dict

if __name__ == '__main__':
    main()
