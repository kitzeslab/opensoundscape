# todo: save models with extras in similar way to resnet_binary
# (includes train/valid scores/preds)

# adapted from zhmiao
# github.com/zhmiao/BirdMultiLabel/blob/master/src/algorithms/plain_resnet.py

import os
import numpy as np
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict

# from tqdm import tqdm
import random

import torch
import torch.optim as optim
from torch.nn.functional import softmax
from sklearn.metrics import jaccard_score, hamming_loss, precision_recall_fscore_support

from opensoundscape.torch.architectures.distreg_resnet_architecture import (
    DistRegResNetClassifier,
)
from opensoundscape.torch.architectures.plain_resnet import PlainResNetClassifier
from opensoundscape.torch.models.utils import BaseModule, get_dataloader
from opensoundscape.metrics import multiclass_metrics, binary_metrics


# NOTE: Turning off all logging for now. may want to use logging module in future


class PytorchModel(BaseModule):

    """
    Generic Pytorch Model with .train() and .predict()

    flexible architecture, optimizer, loss function, parameters

    for tutorials see []
    """

    # TODO: move hyperparameters into self.hyperparameters? (dictionary?)
    def __init__(self, architecture, classes):
        """if you want to change parameters,
        first create the object then modify them
        """
        super(PytorchModel, self).__init__()

        self.name = "PytorchModel"

        # model characteristics
        self.classes = classes  # train_dataset.labels
        print(f"created {self.name} model object with {len(self.classes)} classes")

        ### network parameters ###
        self.weights_init = "ImageNet"
        self.num_layers = 18  # can use 50 for resnet50
        self.sampler = None  # can be "imbalanced"
        self.current_epoch = 0

        ### architecture ###
        # (feature extraction + classifier + loss fn)
        self.network = architecture
        # TODO: can it be easier to change the loss function?

        ### training parameters ###
        # defaults partially from https://github.com/zhmiao/BirdMultiLabel/blob/master/configs/XENO/multi_label_reg_10_091620.yaml
        # optimizer
        self.opt_net = None  # don't set directly. initialized during training
        self.optimizer_cls = optim.SGD  # or torch.optim.Adam, etc
        self.optimizer_params = {
            # optimization parameters for parts of the networks - see
            # https://pytorch.org/docs/stable/optim.html#per-parameter-options
            "feature": {  # optimizer parameters for feature extraction layers
                "params": self.network.feature.parameters(),
                "lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "classifier": {  # optimizer parameters for classification layers
                "params": self.network.classifier.parameters(),
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
        }
        # lr_scheduler
        self.lr_update_interval = 10  # update learning rates every # epochs
        self.lr_cooling_factor = 0.7  # multiply learning rates by # on each update

        ### metrics ###
        self.metrics_fn = multiclass_metrics  # or binary_metrics
        self.single_target = False  # if True: predict only class w max score
        self.prediction_threshold = 0.25
        # dictionaries to store accuracy metrics & loss for each epoch
        self.train_metrics = {}
        self.valid_metrics = {}
        self.loss = {}  # could add TensorBoard tracking

    def _init_optimizer(self):
        """initialize an instance of self.optimizer

        This function is called at during .train()

        To modify the optimizer, change the value of
        self.optimizer_cls and/or self.optimizer_params
        prior to calling .train().
        """
        return self.optimizer_cls(self.optimizer_params.values())

    def _set_train(self, batch_size, num_workers):
        """Prepare network for training on train_dataset

        Args:
            batch_size: number of training files to load/process before
                        re-calculating the loss function and backpropagation
            num_workers: parallelization (number of cores or cpus)

        Effects:
            Sets up the optimization, loss function, and network.
            Creates self.train_loader
        """

        ###########################
        # Setup cuda and networks #
        ###########################
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.network.to(self.device)

        ######################
        # Optimization setup #
        ######################

        # Setup optimizer parameters for each network component
        # If optimizer already exists, keep the same state dict
        # (for instance, user may be resuming training)
        # we re-create it bc the user may have changed self.optimizer_cls
        if self.opt_net is not None:
            optim_state_dict = self.opt_net.state_dict()
            self.opt_net = self._init_optimizer()
            self.opt_net.load_state_dict(optim_state_dict)
        else:
            self.opt_net = self._init_optimizer()

        # Update loss function in case it required knowledge of class_freq
        self.network.class_freq = np.sum(self.train_dataset.df.values, 0)
        self.network.setup_loss()

        # Set up learning rate cooling schedule
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt_net,
            step_size=self.lr_update_interval,
            gamma=self.lr_cooling_factor,
            last_epoch=self.current_epoch - 1,
        )

        ######################
        # Dataloader setup #
        ######################

        # train_loader samples batches of images + labels from train_dataset
        self.train_loader = get_dataloader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            sampler=self.sampler,
        )

        # valid_loader samples batches of images + labels from valid_dataset
        # self.valid_loader = get_dataloader(
        #     self.valid_dataset,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     shuffle=False,
        #     sampler=None,
        # )

    def train_epoch(self):
        """perform forward pass, loss, backpropagation for one epoch

        Returns: (targets, predictions, scores) on training files
        """
        self.network.train()

        total_tgts = []
        total_preds = []
        total_scores = []

        for batch_idx, item in enumerate(self.train_loader):
            # load a batch of images and labels from the train loader
            # all augmentation occurs in the Preprocessor (train_loader)
            data, labels = item["X"].to(self.device), item["y"].to(self.device)
            labels = labels.squeeze(1)

            ####################
            # Forward and loss #
            ####################

            # forward pass: feature extractor and classifier
            feats = self.network.feature(data)  # feature extraction
            logits = self.network.classifier(feats)  # classification

            # save targets and predictions
            total_scores.append(logits.detach().cpu().numpy())
            total_tgts.append(labels.detach().cpu().numpy())
            total_preds.append(  # TODO: should this be allowed to single-target?
                (
                    (torch.sigmoid(logits) >= self.prediction_threshold)
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                )
            )
            # calculate loss #may be able to move loss fn outside of network
            loss = self.network.criterion_cls(logits, labels)
            self.loss[self.current_epoch] = float(loss)

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_net.zero_grad()
            # backward pass: calculate the gradients
            loss.backward()
            # update the network using the gradients*lr
            self.opt_net.step()

            ################
            # Save weights #
            ################
            # if batch_idx % self.save_interval == 0:
            #     self.save(self.save_path)

            ###########
            # Logging #
            ###########
            # log basic train info (used to print every batch)
            if batch_idx % self.log_interval == 0:
                """show some basic progress metrics during the epoch"""
                N = len(self.train_loader)
                print(
                    "Epoch: {} [batch {}/{} ({:.2f}%)] ".format(
                        self.current_epoch, batch_idx, N, 100 * batch_idx / N
                    )
                )

                tgts = labels.int().detach().cpu().numpy()

                # Threshold prediction #TODO: single_target option?
                preds = (
                    (torch.sigmoid(logits) >= self.prediction_threshold)
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Log the Jaccard score and Hamming loss
                jac = jaccard_score(tgts, preds, average="macro")
                ham = hamming_loss(tgts, preds)
                print(f"\tJacc: {jac:0.3f} Hamm: {ham:0.3f} DistLoss: {loss:.3f}")

        # update learning parameters each epoch
        self.scheduler.step()

        # return targets, preds, scores
        total_tgts = np.concatenate(total_tgts, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_scores = np.concatenate(total_scores, axis=0)

        return total_tgts, total_preds, total_scores

    def train(
        self,
        train_dataset,
        valid_dataset,
        epochs=1,
        batch_size=1,
        num_workers=0,
        save_path=".",
        save_interval=1,  # save weights every n epochs
        log_interval=10,  # print metrics every n batches
    ):
        """train the model on samples from train_dataset

        If customized loss functions, networks, optimizers, or schedulers
        are desired, modify the respective attributes before calling .train().

        Args:
            train_dataset: a Preprocessor that loads sample (audio file + label)
                           to Tensor in batches (see docs/tutorials for details)
            valid_dataset: a Preprocessor for evaluating performance
            epochs: number of epochs to train for [default=1]
                    (1 epoch constitutes 1 view of each training sample)
            batch_size: number of training files to load/process before
                        re-calculating the loss function and backpropagation
            num_workers: parallelization (ie, cores or cpus)
                        Note: use 0 for single (root) process (not 1)
            save_path: location to save intermediate and best model objects
                        [default=".", ie current location of script]
            save_interval: interval in epochs to save model object with weights
                            [default:1] Note: the best model is always saved to
                            best.model in addition to other saved epochs.
            log_interval: interval in epochs to evaluate model with validation
                          dataset and print metrics to the log

        #TODO: verbose switch
        """

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.save_path = save_path

        self._set_train(batch_size, num_workers)

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            # 1 epoch = 1 view of each training file
            # loss fn & backpropogation occurs after each batch

            ### Training ###
            train_targets, train_preds, train_scores = self.train_epoch()

            #### Validation ###
            print("\nValidation.")
            valid_scores, valid_preds, valid_targets = [
                df.values
                for df in self.predict(
                    self.valid_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    activation_layer="softmax_and_logit"
                    if self.single_target
                    else "raw",
                    binary_preds="single_target"
                    if self.single_target
                    else "multi_target",
                    threshold=self.prediction_threshold,
                )
            ]

            ### Metrics ###
            self.valid_metrics[self.current_epoch] = self.metrics_fn(
                valid_targets, valid_preds, self.classes
            )
            self.train_metrics[self.current_epoch] = self.metrics_fn(
                train_targets, train_preds, self.classes
            )
            # print basic metrics (this could  break if metrics_fn changes)
            print(
                f"\t Precision: {self.valid_metrics[self.current_epoch]['precision']}"
            )
            print(f"\t Recall: {self.valid_metrics[self.current_epoch]['recall']}")
            print(f"\t F1: {self.valid_metrics[self.current_epoch]['f1']}")

            ### Save ###
            if (
                self.current_epoch + 1
            ) % self.save_interval == 0 or epoch >= epochs - 1:
                print("Saving weights, metrics, and train/valid scores.")

                self.save(
                    extras={
                        "train_scores": train_scores,
                        "train_targets": train_targets,
                        "validation_scores": valid_scores,
                        "validation_targets": valid_targets,
                    }
                )

            # if best model (by F1 score), update & save weights to best.model
            f1 = self.valid_metrics[self.current_epoch]["f1"]
            if f1 > best_f1:
                self.network.update_best()
                best_f1 = f1
                best_epoch = self.current_epoch
                print("Updating best model")
                self.save(
                    f"{self.save_path}/best.model",
                    extras={
                        "train_scores": train_scores,
                        "train_targets": train_targets,
                        "validation_scores": valid_scores,
                        "validation_targets": valid_targets,
                    },
                )

            self.current_epoch += 1

        print(f"\nBest Model Appears at Epoch {best_epoch} with F1 {best_f1:.3f}.")

    def evaluate(self, loader, single_target=False, threshold=0.5):
        """Predict on data from DataLoader

        this function is called with valid_dataset during training

        Args:
            loader:
                a DataLoader that supplies {"X":tensor,"y":list of labels}
            single_target (bool):
                predict exactly one class per sample
            threshold (float or list):
                prediction threshold for post-sigmoid scores,
                or a list of thresholds, one per class
                (only relevant for single_target=False) [default=0.5]

        Returns:
            targets, binary predictions, scores
        """
        # TODO: should (?) call predict() with flag to return targets & preds

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.network.to(self.device)

        self.network.eval()

        total_tgts = []
        total_preds = []
        total_scores = []

        with torch.set_grad_enabled(False):

            for batch in loader:  # one batch of X,y samples
                # setup data
                data = batch["X"].to(self.device)
                labels = batch["y"].to(self.device)
                data.requires_grad = False
                labels.requires_grad = False

                # forward
                feats = self.network.feature(data)
                logits = self.network.classifier(feats)
                sigmoids = torch.sigmoid(logits)

                # Binary predictions
                if single_target:  # predict highest scoring class only
                    # TODO: check if this is working correctly
                    batch_preds = np.zeros(np.shape(logits)).astype(int)
                    for i, class_scores in enumerate(logits):
                        batch_preds[i, np.argmax(class_scores)] = 1
                else:  # predict 0 or 1 based on a fixed threshold
                    batch_preds = (sigmoids >= threshold).int().detach().cpu().numpy()
                total_preds.append(batch_preds)
                total_tgts.append(labels.int().detach().cpu().numpy())
                total_scores.append(sigmoids.detach().cpu().numpy())

        total_tgts = np.concatenate(total_tgts, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_scores = np.concatenate(total_scores, axis=0)

        return total_tgts, total_preds, total_scores

    def save(
        self,
        path=None,
        save_weights=True,
        save_optimizer=True,  # TODO: do we need to save the scheduler?
        extras={},
    ):
        """save model with weights (default location is self.save_path)

        if save_weights is False: only save metadata/metrics
        if save_optimizer is False: don't save self.optim.state_dict()
        extras: arbitrary dictionary of things to save, eg valid-preds
        """

        if path is None:
            path = f"{self.save_path}/epoch-{self.current_epoch}.model"
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)

        # add items to save into a dictionary
        model_dict = {
            "model": self.name,
            "classes": self.classes,
            "epoch": self.current_epoch,
            "valid_metrics": self.valid_metrics,
            "train_metrics": self.valid_metrics,
            "loss": self.loss,
            "lr_update_interval": self.lr_update_interval,
            "lr_cooling_factor": self.lr_cooling_factor,
            "optimizer_params": self.optimizer_params,
            "single_target": self.single_target,
        }
        if save_weights:
            model_dict.update({"model_state_dict": self.network.state_dict()})
        if save_optimizer:
            if self.opt_net is None:
                self.opt_net = self._init_optimizer()
            model_dict.update({"optimizer_state_dict": self.opt_net.state_dict()})

        # user can provide an arbitrary dictionary of extra things to save
        model_dict.update(extras)

        print(f"Saving to {path}")
        torch.save(model_dict, path)

    def load(
        self,
        path,
        load_weights=True,
        load_classifier_weights=True,
        load_optimizer_state_dict=True,
        verbose=False,
    ):
        """load model and optimizer state_dict from disk

        the object should be saved with model.save()
        which uses torch.save with keys for 'model_state_dict' and 'optimizer_state_dict'

        verbose: if True, print missing/unused keys for model weights
        """
        try:
            model_dict = torch.load(path)
        except RuntimeError:  # model was saved on GPU and now on CPU
            model_dict = torch.load(path, map_location=torch.device("cpu"))

        # load misc saved items
        self.current_epoch = model_dict["epoch"]
        self.train_metrics = model_dict["train_metrics"]
        self.valid_metrics = model_dict["valid_metrics"]
        self.loss = model_dict["loss"]
        self.lr_update_interval = model_dict["lr_update_interval"]
        self.lr_cooling_factor = model_dict["lr_cooling_factor"]
        self.optimizer_params = model_dict["optimizer_params"]
        self.single_target = model_dict["single_target"]

        # load the nn feature/classifier weights from the checkpoint
        if load_weights and "model_state_dict" in model_dict:
            print("loading weights from saved object")
            init_weights = model_dict["model_state_dict"]
            # init_weights = OrderedDict({'network.'+k: init_weights[k]
            #                         for k in init_weights})
            if load_classifier_weights:
                self.network.load_state_dict(init_weights, strict=False)
                load_keys = set(init_weights.keys())
                self_keys = set(self.network.state_dict().keys())
            else:  # load only the feature weights
                init_weights = OrderedDict(
                    {k.replace("feature.", ""): init_weights[k] for k in init_weights}
                )
                self.network.feature.load_state_dict(init_weights, strict=False)
                load_keys = set(init_weights.keys())
                self_keys = set(self.network.feature.state_dict().keys())

            if verbose:
                # check if some weight_dict keys were missing or unused
                missing_keys = self_keys - load_keys
                unused_keys = load_keys - self_keys
                print("missing keys: {}".format(sorted(list(missing_keys))))
                print("unused_keys: {}".format(sorted(list(unused_keys))))

        # create an optimizer then load the checkpoint state dict
        self.opt_net = self._init_optimizer()
        if load_optimizer_state_dict and "optimizer_state_dict" in model_dict:
            self.opt_net.load_state_dict(model_dict["optimizer_state_dict"])

    def predict(  # TODO: check that columns match self.classes
        self,
        prediction_dataset,
        batch_size=1,
        num_workers=0,
        activation_layer=None,  # softmax','sigmoid','softmax_and_logit', None
        binary_preds=None,  #'single_target','multi_target', None
        threshold=0.5,
    ):
        """Generate predictions on a dataset

        Choose to return any combination of scores, labels, and single-target or
        multi-target binary predictions. Also choose activation layer for scores
        (softmax, sigmoid, softmax then logit, or None).

        Note: the order of returned dataframes is scores, preds, labels

        Args:
            prediction_dataset:
                a pytorch dataset object that returns tensors,
                such as opensoundscape.datasets.AudioToImagePreprocessor
            batch_size:
                Number of files to load simultaneously [default: 1]
            num_workers:
                parallelization (ie cpus or cores), use 0 for current proess
                [default: 0]
            activation_layer:
                Optionally apply an activation layer such as sigmoid or
                softmax to the raw outputs of the model.
                options:
                - None: no activation, return raw scores (ie logit, [-inf:inf])
                - 'softmax': scores all classes sum to 1
                - 'sigmoid': all scores in [0,1] but don't sum to 1
                - 'softmax_and_logit': applies softmax first then logit
                [default: None]
            binary_preds:
                Optionally return binary (thresholded 0/1) predictions
                options:
                - 'single_target': max scoring class = 1, others = 0
                - 'multi_target': scores above threshold = 1, others = 0
                - None: do not create or return binary predictions
                [default: None]
            threshold:
                prediction threshold for sigmoid scores. Only relevant when
                binary_preds == 'multi_target'

        Returns: 3 DataFrames (or Nones), w/index matching prediciton_dataset.df
            scores: post-activation_layer scores
            predictions: 0/1 preds for each class
            labels: labels from dataset (if available)

        Note: if no return type selected for labels/scores/preds, returns None
        instead of a DataFrame
        """

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.network.to(self.device)

        self.network.eval()

        dataloader = torch.utils.data.DataLoader(
            prediction_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            # pin_memory=True,#TODO: what does pin_memory=True do?
        )

        ### Prediction ###
        total_scores = []
        total_preds = []
        total_tgts = []

        has_labels = False

        # disable gradient updates during inference
        with torch.set_grad_enabled(False):

            for batch in dataloader:
                # get batch of Tensors
                batch_tensors = batch["X"].to(self.device)
                batch_tensors.requires_grad = False
                # get batch's labels if available
                batch_targets = []
                if "y" in batch.keys():
                    batch_targets = batch["y"].to(self.device)
                    batch_targets.requires_grad = False
                    has_labels = True

                # forward pass of network: feature extractor + classifier
                feats = self.network.feature(batch_tensors)
                logits = self.network.classifier(feats)

                ### Activation layer ###
                if activation_layer == None:
                    scores = logits
                elif activation_layer == "softmax":
                    # "softmax" activation: preds across all classes sum to 1
                    scores = softmax(logits, 1)
                elif activation_layer == "sigmoid":
                    scores = torch.sigmoid(logits)
                elif activation_layer == "softmax_and_logit":
                    scores = torch.logit(softmax(logits, 1))
                else:
                    raise ValueError(
                        f"invalid option for activation_layer: {activation_layer}"
                    )

                ### Binary predictions ###
                if (
                    binary_preds == "single_target"
                ):  # predict highest scoring class only
                    batch_preds = np.zeros(np.shape(logits)).astype(int)
                    for i, class_scores in enumerate(logits.detach().cpu().numpy()):
                        batch_preds[
                            i, np.argmax(class_scores)
                        ] = 1  # TODO: can we do this on the tensor instead?
                elif (
                    binary_preds == "multi_target"
                ):  # predict 0 or 1 based on a fixed threshold
                    batch_preds = (
                        (torch.sigmoid(logits) >= threshold)
                        .int()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                elif binary_preds is None:
                    batch_preds = []
                else:
                    raise ValueError(f"invalid option for binary_preds: {binary_preds}")

                # detach the returned values: currently tethered to gradients
                # and updates via optimizer/backprop. detach() returns
                # just numeric values.
                total_scores.append(scores.detach().cpu().numpy())
                total_preds.append(batch_preds)
                total_tgts.append(batch_targets.int().detach().cpu().numpy())

        # aggregate across all batches
        total_tgts = np.concatenate(total_tgts, axis=0)
        total_scores = np.concatenate(total_scores, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)

        # return 3 DataFrames with same index/columns as prediction_dataset's df
        # use None for placeholder if no preds / labels
        samples = prediction_dataset.df.index.values
        score_df = pd.DataFrame(index=samples, data=total_scores, columns=self.classes)
        pred_df = (
            None
            if binary_preds is None
            else pd.DataFrame(index=samples, data=total_preds, columns=self.classes)
        )
        label_df = (
            None
            if not has_labels
            else pd.DataFrame(index=samples, data=total_tgts, columns=self.classes)
        )

        return score_df, pred_df, label_df

    @classmethod
    def from_checkpoint(cls, path):
        """create model instance from saved model"""
        model_obj = cls()
        model_obj.load(path)
        return model_obj


class Resnet18Multilabel(PytorchModel):
    def __init__(self, classes):
        """if you want to change other parameters,
        simply create the object then modify them
        """
        # TODO: check that df columns match self.classes

        self.classes = classes
        self.weights_init = "ImageNet"

        # initialize the model architecture without an optimizer
        # since we dont know the train class counts to give the optimizer
        architecture = DistRegResNetClassifier(
            num_cls=len(self.classes),
            weights_init=self.weights_init,
            num_layers=18,
            class_freq=None,
        )

        super(Resnet18Multilabel, self).__init__(architecture, self.classes)
        self.name = "Resnet18Multilabel"

    @classmethod
    def from_checkpoint(cls, path):
        classes = torch.load(path)["classes"]
        model_obj = cls(classes)
        model_obj.load(path)
        return model_obj


class Resnet18Binary(
    PytorchModel
):  # TODO: binary model should not make [1,1] prediction (need softmax)
    # TODO: binary model should only accept train/test dfs w/1 column
    # TODO: validate that index of df is path and labels are one-hot
    # TODO: make a single-target class, this is just a special case w 2 classes
    def __init__(self):
        """if you want to change parameters, create the object then modify them"""
        self.weights_init = "ImageNet"
        self.classes = ["negative", "positive"]

        architecture = PlainResNetClassifier(  # pass architecture as argument
            num_cls=2, weights_init=self.weights_init, num_layers=18
        )

        super(Resnet18Binary, self).__init__(architecture, self.classes)
        self.name = "Resnet18Binary"

        self.metrics_fn = binary_metrics
        self.single_target = True
