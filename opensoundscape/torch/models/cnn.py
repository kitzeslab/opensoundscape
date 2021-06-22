"""classes for pytorch machine learning models in opensoundscape

For tutorials, see notebooks on opensoundscape.org
"""
# originally based on zhmiao's BirdMultiLabel

import os
import numpy as np
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict
import warnings
import random

import torch
import torch.optim as optim
from torch.nn.functional import softmax
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, hamming_loss, precision_recall_fscore_support

from opensoundscape.torch.architectures.resnet import ResNetArchitecture
from opensoundscape.torch.models.utils import BaseModule, get_dataloader
from opensoundscape.metrics import multiclass_metrics, binary_metrics
from opensoundscape.torch.loss import (
    BCEWithLogitsLoss_hot,
    CrossEntropyLoss_hot,
    ResampleLoss,
)
from opensoundscape.torch.safe_dataset import SafeDataset


class PytorchModel(BaseModule):
    """
    Generic Pytorch Model with .train() and .predict()

    flexible architecture, optimizer, loss function, parameters

    for tutorials and examples see opensoundscape.org

    methods include train(), predict(), save(), and load()

    Args:
        architecture:
            a model architecture object, for example one generated
            with the torch.architectures.cnn_architectures module
        classes:
            list of class names. Must match with training dataset classes.
        single_target:
            - True: model expects exactly one positive class per sample
            - False: samples can have an number of positive classes
            [default: False]
    """

    def __init__(self, architecture, classes, single_target=False):

        super(PytorchModel, self).__init__()

        self.name = "PytorchModel"

        # model characteristics
        self.current_epoch = 0
        self.classes = classes  # train_dataset.labels
        self.single_target = single_target  # if True: predict only class w max score
        print(f"created {self.name} model object with {len(self.classes)} classes")

        ### data loading parameters ###
        self.sampler = None  # can be "imbalanced" for ImbalancedDatasetSmpler

        ### architecture ###
        # (feature extraction + classifier + loss fn)
        # can by a pytorch CNN such as Resnet18, or RNN, etc
        # must have .forward(), .train(), .eval(), .to(), .state_dict()
        self.network = architecture

        ### loss function ###
        if self.single_target:  # use cross entropy loss by default
            self.loss_cls = CrossEntropyLoss_hot
        else:  # for multi-target, use binary cross entropy
            self.loss_cls = BCEWithLogitsLoss_hot

        ### training parameters ###
        # defaults partially from  zhmiao's BirdMultiLabel
        # optimizer
        self.opt_net = None  # don't set directly. initialized during training
        self.optimizer_cls = optim.SGD  # or torch.optim.Adam, etc

        self.optimizer_params = {  # optimizer parameters for classification layers
            "params": self.network.parameters(),
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        }

        # lr_scheduler
        self.lr_update_interval = 10  # update learning rates every # epochs
        self.lr_cooling_factor = 0.7  # multiply learning rates by # on each update

        ### metrics ###
        self.metrics_fn = multiclass_metrics  # or binary_metrics
        self.prediction_threshold = 0.25
        # dictionaries to store accuracy metrics & loss for each epoch
        self.train_metrics = {}
        self.valid_metrics = {}
        self.loss_hist = {}  # could add TensorBoard tracking

    def _init_optimizer(self):
        """initialize an instance of self.optimizer

        This function is called during .train() so that the user
        has a chance to swap/modify the optimizer before training.

        To modify the optimizer, change the value of
        self.optimizer_cls and/or self.optimizer_params
        prior to calling .train().
        """
        return self.optimizer_cls([self.optimizer_params])

    def _init_loss_fn(self):
        """initialize an instance of self.loss_cls

        This function is called during .train() so that the user
        has a chance to change the loss function before training.
        """
        self.loss_fn = self.loss_cls()

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

        ###########################
        # Setup loss function     #
        ###########################
        self._init_loss_fn()

        ######################
        # Optimization setup #
        ######################

        # Setup optimizer parameters for each network component
        # Note: we re-create bc the user may have changed self.optimizer_cls
        # If optimizer already exists, keep the same state dict
        # (for instance, user may be resuming training w/saved state dict)
        if self.opt_net is not None:
            optim_state_dict = self.opt_net.state_dict()
            self.opt_net = self._init_optimizer()
            self.opt_net.load_state_dict(optim_state_dict)
        else:
            self.opt_net = self._init_optimizer()

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

        # SafeDataset loads a new sample if loading a sample throws an error
        # indices of bad samples are appended to ._unsafe_indices
        self.train_safe_dataset = SafeDataset(self.train_dataset)

        # train_loader samples batches of images + labels from train_dataset
        self.train_loader = get_dataloader(
            self.train_safe_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            sampler=self.sampler,
        )

    def train_epoch(self):
        """perform forward pass, loss, backpropagation for one epoch

        Returns: (targets, predictions, scores) on training files
        """
        self.network.train()

        total_tgts = []
        total_preds = []
        total_scores = []

        for batch_idx, batch_data in enumerate(self.train_loader):
            # load a batch of images and labels from the train loader
            # all augmentation occurs in the Preprocessor (train_loader)
            batch_tensors = batch_data["X"].to(self.device)
            batch_labels = batch_data["y"].to(self.device)
            batch_labels = batch_labels.squeeze(1)

            ####################
            # Forward and loss #
            ####################

            # forward pass: feature extractor and classifier
            logits = self.network.forward(batch_tensors)

            # save targets and predictions
            total_scores.append(logits.detach().cpu().numpy())
            total_tgts.append(batch_labels.detach().cpu().numpy())

            # generate binary predictions
            if self.single_target:  # predict highest scoring class only
                batch_preds = F.one_hot(logits.argmax(1), len(logits[0]))
            else:  # multi-target: predict 0 or 1 based on a fixed threshold
                batch_preds = torch.sigmoid(logits) >= self.prediction_threshold
            total_preds.append(batch_preds.int().detach().cpu().numpy())

            # calculate loss
            loss = self.loss_fn(logits, batch_labels)
            self.loss_hist[self.current_epoch] = loss.detach().cpu().numpy()

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_net.zero_grad()
            # backward pass: calculate the gradients
            loss.backward()
            # update the network using the gradients*lr
            self.opt_net.step()

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

                # Log the Jaccard score and Hamming loss
                tgts = batch_labels.int().detach().cpu().numpy()
                preds = batch_preds.int().detach().cpu().numpy()
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
        unsafe_sample_log="./unsafe_samples.log",
    ):
        """train the model on samples from train_dataset

        If customized loss functions, networks, optimizers, or schedulers
        are desired, modify the respective attributes before calling .train().

        Args:
            train_dataset:
                a Preprocessor that loads sample (audio file + label)
                to Tensor in batches (see docs/tutorials for details)
            valid_dataset:
                a Preprocessor for evaluating performance
            epochs:
                number of epochs to train for [default=1]
                (1 epoch constitutes 1 view of each training sample)
            batch_size:
                number of training files to load/process before
                re-calculating the loss function and backpropagation
            num_workers:
                parallelization (ie, cores or cpus)
                Note: use 0 for single (root) process (not 1)
            save_path:
                location to save intermediate and best model objects
                [default=".", ie current location of script]
            save_interval:
                interval in epochs to save model object with weights [default:1]
                Note: the best model is always saved to best.model
                in addition to other saved epochs.
            log_interval:
                interval in epochs to evaluate model with validation
                dataset and print metrics to the log
            unsafe_sample_log:
                file path: log all samples that failed in preprocessing
                (file written when training completes)
                - if None,  does not write a file

        """
        class_err = (
            "Train and validation datasets must have same classes "
            "and class order as model object."
        )
        assert list(self.classes) == list(train_dataset.df.columns), class_err
        assert list(self.classes) == list(valid_dataset.df.columns), class_err

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.save_path = save_path

        self._set_train(batch_size, num_workers)

        self.best_f1 = 0.0
        self.best_epoch = 0

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
                    else None,
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
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = self.current_epoch
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

        print(
            f"\nBest Model Appears at Epoch {self.best_epoch} with F1 {self.best_f1:.3f}."
        )

        # warn the user if there were unsafe samples (failed to preprocess)
        if len(self.train_safe_dataset._unsafe_indices) > 0:
            bad_paths = self.train_safe_dataset.df.index[
                self.train_safe_dataset._unsafe_indices
            ].values
            msg = (
                f"There were {len(bad_paths)} "
                "samples that raised errors during preprocessing. "
            )
            if unsafe_sample_log is not None:
                msg += f"Their file paths are logged in {unsafe_sample_log}"
                with open(unsafe_sample_log, "w") as f:
                    [f.write(p + "\n") for p in bad_paths]
            warnings.warn(msg)

    def save(self, path=None, save_weights=True, save_optimizer=True, extras={}):
        """save model with weights (default location is self.save_path)

        Args:
            path: destination for saved model. if None, uses self.save_path
            save_weights: if False, only save metadata/metrics [default: True]
            save_optimizer: if False, don't save self.optim.state_dict()
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
            "loss_hist": self.loss_hist,
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

        Args:
            path: where the file is saved
            load_weights: if False, ignore network weights [default:True]
            load_classifier_weights: if False, ignore classifier layer weights
                Use False to only load feature weights, eg to re-use
                trained cnn's feature extractor for new class [default: True]
            load_optimizer_state_dict: if False, ignore saved parameters
                for optimizer's state [default: True]
            verbose: if True, print missing and unused keys for model weights
        """
        try:
            model_dict = torch.load(path)
        except RuntimeError:  # model was saved on GPU and now on CPU
            model_dict = torch.load(path, map_location=torch.device("cpu"))

        # load misc saved items
        self.current_epoch = model_dict["epoch"]
        self.train_metrics = model_dict["train_metrics"]
        self.valid_metrics = model_dict["valid_metrics"]
        self.loss_hist = model_dict["loss_hist"]
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

    def predict(
        self,
        prediction_dataset,
        batch_size=1,
        num_workers=0,
        activation_layer=None,  # softmax','sigmoid','softmax_and_logit', None
        binary_preds=None,  #'single_target','multi_target', None
        threshold=0.5,
        error_log=None,
    ):
        """Generate predictions on a dataset

        Choose to return any combination of scores, labels, and single-target or
        multi-target binary predictions. Also choose activation layer for scores
        (softmax, sigmoid, softmax then logit, or None).

        Note: the order of returned dataframes is (scores, preds, labels)

        Args:
            prediction_dataset:
                a Preprocessor or DataSset object that returns tensors,
                such as AudioToSpectrogramPreprocessor (no augmentation)
                or CnnPreprocessor (w/augmentation) from opensoundscape.datasets
            batch_size:
                Number of files to load simultaneously [default: 1]
            num_workers:
                parallelization (ie cpus or cores), use 0 for current process
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
            error_log:
                if not None, saves a list of files that raised errors to
                the specified file location [default: None]

        Returns: 3 DataFrames (or Nones), w/index matching prediciton_dataset.df
            scores: post-activation_layer scores
            predictions: 0/1 preds for each class
            labels: labels from dataset (if available)

        Note: if loading an audio file raises a PreprocessingError, the scores
            and predictions for that sample will be np.nan

        Note: if no return type selected for labels/scores/preds, returns None
        instead of a DataFrame in the returned tuple
        """
        err_msg = (
            "Prediction dataset must have same classes"
            "and class order as model object, or no classes."
        )
        if len(prediction_dataset.df.columns) > 0:
            assert list(self.classes) == list(prediction_dataset.df.columns), err_msg

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.network.to(self.device)

        self.network.eval()

        # SafeDataset will not fail on bad files,
        # but will provide a different sample! Later we go back and replace scores
        # with np.nan for the bad samples (using safe_dataset._unsafe_indices)
        # this approach to error handling feels hacky
        safe_dataset = SafeDataset(prediction_dataset)

        dataloader = torch.utils.data.DataLoader(
            safe_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            # use pin_memory=True when loading files on CPU and training on GPU
            pin_memory=torch.cuda.is_available(),
        )

        ### Prediction ###
        total_scores = []
        total_preds = []
        total_tgts = []

        failed_files = []  # keep list of any samples that raise errors

        has_labels = False

        # disable gradient updates during inference
        with torch.set_grad_enabled(False):

            for batch in dataloader:
                # get batch of Tensors
                batch_tensors = batch["X"].to(self.device)
                batch_tensors.requires_grad = False
                # get batch's labels if available
                batch_targets = torch.Tensor([]).to(self.device)
                if "y" in batch.keys():
                    batch_targets = batch["y"].to(self.device)
                    batch_targets.requires_grad = False
                    has_labels = True

                # forward pass of network: feature extractor + classifier
                logits = self.network.forward(batch_tensors)

                ### Activation layer ###
                if activation_layer == None:  # scores [-inf,inf]
                    scores = logits
                elif activation_layer == "softmax":
                    # "softmax" activation: preds across all classes sum to 1
                    scores = softmax(logits, 1)
                elif activation_layer == "sigmoid":  # map [-inf,inf] to [0,1]
                    scores = torch.sigmoid(logits)
                elif activation_layer == "softmax_and_logit":  # scores [-inf,inf]
                    scores = torch.logit(softmax(logits, 1))
                else:
                    raise ValueError(
                        f"invalid option for activation_layer: {activation_layer}"
                    )

                ### Binary predictions ###
                # generate binary predictions
                if binary_preds == "single_target":
                    # predict highest scoring class only
                    batch_preds = F.one_hot(logits.argmax(1), len(logits[0]))
                elif binary_preds == "multi_target":
                    # predict 0 or 1 based on a fixed threshold
                    batch_preds = torch.sigmoid(logits) >= threshold
                elif binary_preds is None:
                    batch_preds = torch.Tensor([])
                else:
                    raise ValueError(f"invalid option for binary_preds: {binary_preds}")

                # detach the returned values: currently tethered to gradients
                # and updates via optimizer/backprop. detach() returns
                # just numeric values.
                total_scores.append(scores.detach().cpu().numpy())
                total_preds.append(batch_preds.float().detach().cpu().numpy())
                total_tgts.append(batch_targets.int().detach().cpu().numpy())

        # aggregate across all batches
        total_tgts = np.concatenate(total_tgts, axis=0)
        total_scores = np.concatenate(total_scores, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)

        print(np.shape(total_scores))

        # replace scores/preds with nan for samples that failed in preprocessing
        # this feels hacky (we predicted on substitute-samples rather than
        # skipping the samples that failed preprocessing)
        total_scores[safe_dataset._unsafe_indices, :] = np.nan
        if binary_preds is not None:
            total_preds[safe_dataset._unsafe_indices, :] = np.nan

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


class CnnResampleLoss(PytorchModel):
    """Subclass of PytorchModel with ResampleLoss.

    ResampleLoss may perform better than BCE Loss for multitarget problems
    in some scenarios.

    Args:
        architecture:
            a model architecture object, for example one generated
            with the torch.architectures.cnn_architectures module
        classes:
            list of class names. Must match with training dataset classes.
        single_target:
            - True: model expects exactly one positive class per sample
            - False: samples can have an number of positive classes
            [default: False]
    """

    def __init__(self, architecture, classes, single_target=False):

        self.classes = classes

        super(CnnResampleLoss, self).__init__(architecture, self.classes, single_target)
        self.name = "CnnResampleLoss"
        self.loss_cls = ResampleLoss

    def _init_loss_fn(self):
        """initialize an instance of self.loss_cls

        We override the parent method because we need to pass class frequency
        to the ResampleLoss constructor

        This function is called during .train() so that the user
        has a chance to change the loss function before training.

        Note: if you change the loss function, you may need to override this
        to correctly initialize self.loss_cls
        """
        class_frequency = np.sum(self.train_dataset.df.values, 0)
        # initializing ResampleLoss requires us to pass class_frequency
        self.loss_fn = self.loss_cls(class_frequency)

    @classmethod
    def from_checkpoint(cls, path):
        # need to get classes first to initialize the model object
        try:
            classes = torch.load(path)["classes"]
        except RuntimeError:  # model was saved on GPU and now on CPU
            classes = torch.load(path, map_location=torch.device("cpu"))["classes"]
        model_obj = cls(classes)
        model_obj.load(path)
        return model_obj


class Resnet18Multiclass(CnnResampleLoss):
    """Multi-class model with resnet18 architecture and ResampleLoss.

    Can be single or multi-target.

    Args:
        classes:
            list of class names. Must match with training dataset classes.
        single_target:
            - True: model expects exactly one positive class per sample
            - False: samples can have an number of positive classes
            [default: False]

    Notes
    - Allows separate parameters for feature & classifier blocks
        via self.optimizer_params's keys: "feature" and "classifier"
        (by using hand-built architecture)
    - Uses ResampleLoss which requires class counts as an input.
    """

    def __init__(self, classes, single_target=False):

        self.classes = classes
        self.weights_init = "ImageNet"

        # initialize the model architecture without an optimizer
        # since we dont know the train class counts to give the optimizer
        architecture = ResNetArchitecture(
            num_cls=len(self.classes), weights_init=self.weights_init, num_layers=18
        )

        super(Resnet18Multiclass, self).__init__(
            architecture, self.classes, single_target
        )
        self.name = "Resnet18Multiclass"

        # optimization parameters for parts of the networks - see
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options
        self.optimizer_params = {
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

    def _init_optimizer(self):
        """initialize an instance of self.optimizer

        We override the parent method because we need to pass a list of
        separate optimizer_params for different parts of the network
        - ie we now have a dictionary of param dictionaries instead of just a
        param dictionary.

        This function is called during .train() so that the user
        has a chance to swap/modify the optimizer before training.

        To modify the optimizer, change the value of
        self.optimizer_cls and/or self.optimizer_params
        prior to calling .train().
        """
        return self.optimizer_cls(self.optimizer_params.values())


class Resnet18Binary(PytorchModel):
    """Subclass of PytorchModel with Resnet18 architecture

    This subclass allows separate training parameters
    for the feature extractor and classifier

    Args:
        classes:
            list of class names. Must match with training dataset classes.
        single_target:
            - True: model expects exactly one positive class per sample
            - False: samples can have an number of positive classes
            [default: False]

    """

    def __init__(self, classes):

        self.weights_init = "ImageNet"
        self.classes = classes

        architecture = ResNetArchitecture(
            num_cls=2, weights_init=self.weights_init, num_layers=18
        )

        super(Resnet18Binary, self).__init__(
            architecture, self.classes, single_target=True
        )
        self.name = "Resnet18Binary"

        self.metrics_fn = binary_metrics
        self.single_target = True

        # optimization parameters for parts of the networks - see
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options
        self.optimizer_params = {
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

    def _init_optimizer(self):
        """initialize an instance of self.optimizer

        We override the parent method because we need to pass a list of
        separate optimizer_params for different parts of the network
         - ie we now have a dictionary of param dictionaries instead of just a
         param dictionary.

        This function is called during .train() so that the user
        has a chance to swap/modify the optimizer before training.

        To modify the optimizer, change the value of
        self.optimizer_cls and/or self.optimizer_params
        prior to calling .train().
        """
        return self.optimizer_cls(self.optimizer_params.values())

    @classmethod
    def from_checkpoint(cls, path):
        # need to get classes first to initialize the model object
        try:
            classes = torch.load(path)["classes"]
        except RuntimeError:  # model was saved on GPU and now on CPU
            classes = torch.load(path, map_location=torch.device("cpu"))["classes"]
        model_obj = cls(classes)
        model_obj.load(path)
        return model_obj


class InceptionV3(PytorchModel):
    def __init__(
        self,
        classes,
        freeze_feature_extractor=False,
        use_pretrained=True,
        single_target=False,
    ):
        """Model object for InceptionV3 architecture.

        See opensoundscape.org for exaple use.

        Args:
            classes:
                list of output classes (usually strings)
            freeze-feature_extractor:
                if True, feature weights don't have
                gradient, and only final classification layer is trained
            use_pretrained:
                if True, use pre-trained InceptionV3 Imagenet weights
            single_target:
                if True, predict exactly one class per sample

        """
        from opensoundscape.torch.architectures.cnn_architectures import inception_v3

        self.classes = classes

        architecture = inception_v3(
            len(self.classes),
            freeze_feature_extractor=freeze_feature_extractor,
            use_pretrained=use_pretrained,
        )

        super(InceptionV3, self).__init__(architecture, self.classes, single_target)
        self.name = "InceptionV3"

    def train_epoch(self):
        """perform forward pass, loss, backpropagation for one epoch

        need to override parent because Inception returns different outputs
        from the forward pass (final and auxiliary layers)

        Returns: (targets, predictions, scores) on training files
        """
        self.network.train()

        total_tgts = []
        total_preds = []
        total_scores = []

        for batch_idx, batch_data in enumerate(self.train_loader):
            # load a batch of images and labels from the train loader
            # all augmentation occurs in the Preprocessor (train_loader)
            batch_tensors = batch_data["X"].to(self.device)
            batch_labels = batch_data["y"].to(self.device)
            batch_labels = batch_labels.squeeze(1)

            ####################
            # Forward and loss #
            ####################

            # forward pass: feature extractor and classifier
            # inception returns two sets of outputs
            inception_outputs = self.network.forward(batch_tensors)
            logits = inception_outputs.logits
            aux_logits = inception_outputs.aux_logits

            # save targets and predictions
            total_scores.append(logits.detach().cpu().numpy())
            total_tgts.append(batch_labels.detach().cpu().numpy())

            # generate binary predictions
            if self.single_target:  # predict highest scoring class only
                batch_preds = F.one_hot(logits.argmax(1), len(logits[0]))
            else:  # multi-target: predict 0 or 1 based on a fixed threshold
                batch_preds = torch.sigmoid(logits) >= self.prediction_threshold
            total_preds.append(batch_preds.int().detach().cpu().numpy())

            # calculate loss
            loss1 = self.loss_fn(logits, batch_labels)
            loss2 = self.loss_fn(aux_logits, batch_labels)
            loss = loss1 + 0.4 * loss2
            self.loss_hist[self.current_epoch] = loss.detach().cpu().numpy()

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_net.zero_grad()
            # backward pass: calculate the gradients
            loss.backward()
            # update the network using the gradients*lr
            self.opt_net.step()

            ###########
            # Logging #
            ###########
            # log basic train info
            if batch_idx % self.log_interval == 0:
                """show some basic progress metrics during the epoch"""
                N = len(self.train_loader)
                print(
                    "Epoch: {} [batch {}/{} ({:.2f}%)] ".format(
                        self.current_epoch, batch_idx, N, 100 * batch_idx / N
                    )
                )

                # Log the Jaccard score and Hamming loss
                tgts = batch_labels.int().detach().cpu().numpy()
                preds = batch_preds.int().detach().cpu().numpy()
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

    @classmethod
    def from_checkpoint(cls, path):
        # need to get classes first to initialize the model object
        try:
            classes = torch.load(path)["classes"]
        except RuntimeError:  # model was saved on GPU and now on CPU
            classes = torch.load(path, map_location=torch.device("cpu"))["classes"]
        model_obj = cls(classes=classes, use_pretrained=False)
        model_obj.load(path)
        return model_obj


class InceptionV3ResampleLoss(InceptionV3):
    def __init__(
        self,
        classes,
        freeze_feature_extractor=False,
        use_pretrained=True,
        single_target=False,
    ):
        """Subclass of InceptionV3 with ResampleLoss.

        May perform better than BCE for multitarget problems.
        """
        self.classes = classes

        super(InceptionV3ResampleLoss, self).__init__(
            classes,
            freeze_feature_extractor=False,
            use_pretrained=True,
            single_target=single_target,
        )
        self.name = "InceptionV3ResampleLoss"
        self.loss_cls = ResampleLoss

    def _init_loss_fn(self):
        """initialize an instance of self.loss_cls

        We override the parent method because we need to pass class frequency
        to the ResampleLoss constructor

        This function is called during .train() so that the user
        has a chance to change the loss function before training.

        Note: if you change the loss function, you may need to override this
        to correctly initialize self.loss_cls
        """
        class_frequency = np.sum(self.train_dataset.df.values, 0)
        # initializing ResampleLoss requires us to pass class_frequency
        self.loss_fn = self.loss_cls(class_frequency)
