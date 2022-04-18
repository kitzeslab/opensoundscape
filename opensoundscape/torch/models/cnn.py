"""classes for pytorch machine learning models in opensoundscape

For tutorials, see notebooks on opensoundscape.org
"""

import os
import numpy as np
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict
import warnings
import random
from deprecated import deprecated

import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, hamming_loss, precision_recall_fscore_support

from opensoundscape.torch.architectures import cnn_architectures
from opensoundscape.torch.models.utils import (
    BaseModule,
    get_dataloader,
    get_batch,
    apply_activation_layer,
    tensor_binary_predictions,
    collate_lists_of_audio_clips,
)
from opensoundscape.preprocess.preprocessors import SpecPreprocessor
from opensoundscape.helpers import make_clip_df

from opensoundscape.metrics import multiclass_metrics, binary_metrics
from opensoundscape.torch.loss import (
    BCEWithLogitsLoss_hot,
    CrossEntropyLoss_hot,
    ResampleLoss,
)
from opensoundscape.torch.safe_dataset import SafeDataset
import opensoundscape


class CNN(BaseModule):
    """
    Generic Pytorch Model with .train(), .predict(), and .save()

    flexible architecture, optimizer, loss function, parameters

    for tutorials and examples see opensoundscape.org

    Args:
        architecture:
            *EITHER* a pytorch model object (subclass of torch.nn.Module),
            for example one generated with the `cnn_architectures` module
            *OR* a string matching one of the architectures listed by
            cnn_architectures.list_architectures(), eg 'resnet18'.
            - If a string is provided, uses default parameters
                (including use_pretrained=True)
        classes:
            list of class names. Must match with training dataset classes if training.
        single_target:
            - True: model expects exactly one positive class per sample
            - False: samples can have an number of positive classes
            [default: False]
    """

    def __init__(
        self,
        architecture,
        classes,
        sample_duration,
        single_target=False,
        preprocessor_class=SpecPreprocessor,
        sample_shape=[224, 224],  # TODO: maybe [1,224,224]
    ):

        super(CNN, self).__init__()

        self.name = "CNN"

        # model characteristics
        self.current_epoch = 0
        self.classes = classes
        self.single_target = single_target  # if True: predict only class w max score
        self.opensoundscape_version = opensoundscape.__version__
        print(f"created {self.name} model object with {len(self.classes)} classes")

        ### data loading parameters ###
        self.sampler = None  # can be "imbalanced" for ImbalancedDatasetSmpler

        ### architecture ###
        # (feature extraction + classifier + loss fn)
        # can be a pytorch CNN such as Resnet18, or RNN, etc
        # must have .forward(), .train(), .eval(), .to(), .state_dict()
        # for convenience, allow user to provide string matching
        # a key from cnn_architectures.ARCH_DICT
        if type(architecture) == str:
            assert architecture in cnn_architectures.list_architectures(), (
                f"architecture must be a pytorch model object or string matching "
                f"one of cnn_architectures.list_architectures() options. Got {architecture}"
            )
            architecture = cnn_architectures.ARCH_DICT[architecture](len(classes))
        else:
            assert issubclass(
                type(architecture), torch.nn.Module
            ), "architecture must be a string or an instance of a subclass of torch.nn.Module"
        self.network = architecture

        ### network device ###
        # automatically gpu (default is 'cuda:0') if available
        # can override after init, eg model.device='cuda:1'
        # network and samples are moved to gpu during training/inference
        # devices could be 'cuda:0', torch.device('cuda'), torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ### sample loading/preprocessing ###
        self.preprocessor = preprocessor_class(
            label_df=pd.DataFrame(columns=self.classes),
            sample_duration=sample_duration,
            out_shape=sample_shape,
        )

        ### loss function ###
        if self.single_target:  # use cross entropy loss by default
            self.loss_cls = CrossEntropyLoss_hot
        else:  # for multi-target, use binary cross entropy
            self.loss_cls = BCEWithLogitsLoss_hot

        ### training parameters ###
        # optimizer
        self.opt_net = None  # don't set directly. initialized during training
        self.optimizer_cls = optim.SGD  # or torch.optim.Adam, etc

        # instead of putting "params" key here, we only add it during
        # _init_optimizer, just before initializing the optimizers
        # this avoids an issue when re-loading a model of
        # having the wrong .parameters() list
        self.optimizer_params = {
            # "params": self.network.parameters(),
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        }

        # lr_scheduler
        self.lr_update_interval = 10  # update learning rates every # epochs
        self.lr_cooling_factor = 0.7  # multiply learning rates by # on each update

        ### metrics ###
        self.metrics_fn = multiclass_metrics  # or binary_metrics #TODO: remove?
        self.prediction_threshold = 0.5

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
        param_dict = self.optimizer_params
        param_dict["params"] = self.network.parameters()
        return self.optimizer_cls([param_dict])

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
            Sets up the optimization, loss function, and network
        """

        ###########################
        # Move network to device  #
        ###########################
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

    def _train_epoch(self, train_loader):
        """perform forward pass, loss, backpropagation for one epoch

        Returns: (targets, predictions, scores) on training files
        """
        self.network.train()

        total_tgts = []
        total_preds = []
        total_scores = []
        batch_loss = []

        for batch_idx, batch_data in enumerate(train_loader):
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
            loss = self.loss_fn(
                logits, batch_labels
            )  # TODO: fails when there's only 1 class

            # save loss for each batch; later take average for epoch

            batch_loss.append(loss.detach().cpu().numpy())

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
                N = len(train_loader)
                print(
                    "Epoch: {} [batch {}/{} ({:.2f}%)] ".format(
                        self.current_epoch, batch_idx, N, 100 * batch_idx / N
                    )
                )

                # Log the Jaccard score and Hamming loss, and Loss function
                tgts = batch_labels.int().detach().cpu().numpy()
                preds = batch_preds.int().detach().cpu().numpy()
                jac = jaccard_score(tgts, preds, average="macro")
                ham = hamming_loss(tgts, preds)
                epoch_loss_avg = np.mean(batch_loss)
                print(
                    f"\tJacc: {jac:0.3f} Hamm: {ham:0.3f} DistLoss: {epoch_loss_avg:.3f}"
                )

        # update learning parameters each epoch
        self.scheduler.step()

        # save the loss averaged over all batches
        self.loss_hist[self.current_epoch] = np.mean(batch_loss)

        # return targets, preds, scores
        total_tgts = np.concatenate(total_tgts, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_scores = np.concatenate(total_scores, axis=0)

        return total_tgts, total_preds, total_scores

    def _validation(self, validation_df):  # TODO: this is out of date
        pass

    def train(
        self,
        train_df,
        validation_df,
        epochs=1,
        batch_size=1,
        num_workers=0,
        save_path=".",
        save_interval=1,  # save weights every n epochs
        log_interval=10,  # print metrics every n batches
        unsafe_samples_log="./unsafe_samples.log",
    ):
        """train the model on samples from train_dataset

        If customized loss functions, networks, optimizers, or schedulers
        are desired, modify the respective attributes before calling .train().

        Args:
            train_df:
                a dataframe of files and labels for training the model
            validation_df:
                a dataframe of files and labels for evaluating the model
            epochs:
                number of epochs to train for [default=1]
                (1 epoch constitutes 1 view of each training sample)
            batch_size:
                number of training files simultaneously passed through
                forward pass, loss function, and backpropagation
            num_workers:
                number of parallel CPU tasks for preprocessing
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
            _unsafe_samples_log:
                file path: log all samples that failed in preprocessing
                (file written when training completes)
                - if None,  does not write a file

        """
        class_err = (
            "Train and validation datasets must have same classes "
            "and class order as model object."
        )
        assert list(self.classes) == list(train_df.columns), class_err
        assert list(self.classes) == list(validation_df.columns), class_err

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_path = save_path

        self._set_train(batch_size, num_workers)

        ######################
        # Dataloader setup #
        ######################

        train_dataset = self.preprocessor.sample(n=0)  # TODO: helper fn?
        train_dataset.label_df = train_df

        # SafeDataset loads a new sample if loading a sample throws an error
        # indices of bad samples are appended to ._unsafe_indices
        train_safe_dataset = SafeDataset(train_dataset, unsafe_behavior="substitute")

        # train_loader samples batches of images + labels from training set
        self.train_loader = get_dataloader(
            train_safe_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            sampler=self.sampler,
        )

        self.best_validation_score = 0.0  # TODO: should allow any metric
        self.best_epoch = 0

        for epoch in range(epochs):
            # 1 epoch = 1 view of each training file
            # loss fn & backpropogation occurs after each batch

            ### Training ###
            train_targets, train_preds, train_scores = self._train_epoch(
                self.train_loader
            )

            #### Validation ### #TODO what should be moved to a function?
            print("\nValidation.")
            validation_scores, validation_preds, unsafe_val_samples = self.predict(
                validation_df,
                batch_size=batch_size,
                num_workers=num_workers,  # TODO: should we make these attributes?
                activation_layer="softmax_and_logit" if self.single_target else None,
                binary_preds="single_target" if self.single_target else "multi_target",
                threshold=self.prediction_threshold,
            )
            validation_targets = validation_df.values
            validation_scores = validation_scores.values
            validation_preds = validation_preds.values

            ### Metrics ###
            self.valid_metrics[self.current_epoch] = self.metrics_fn(
                validation_targets, validation_preds, self.classes
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

            validation_score = self.valid_metrics[self.current_epoch]["f1"]

            ### Save ###
            if (
                self.current_epoch + 1
            ) % self.save_interval == 0 or epoch >= epochs - 1:
                print("Saving weights, metrics, and train/valid scores.")

                self.save(f"{self.save_path}/epoch-{self.current_epoch}.model")

            # if best model (by F1 score), update & save weights to best.model
            if validation_score > self.best_validation_score:
                self.best_validation_score = validation_score
                self.best_epoch = self.current_epoch
                print("Updating best model")
                self.save(f"{self.save_path}/best.model")

            self.current_epoch += 1

        print(
            f"\nBest Model Appears at Epoch {self.best_epoch} with Validation score {self.best_validation_score:.3f}."
        )

        # warn the user if there were unsafe samples (failed to preprocess)
        _ = train_safe_dataset.report(log=unsafe_samples_log)

    def save(self, path, save_datasets=True):
        import copy

        """save model with weights using torch.save()

        load from saved file with torch.load(path) or cnn.load_model(path)

        Args:
            path: file path for saved model object
        """
        os.makedirs(Path(path).parent, exist_ok=True)
        model_copy = copy.deepcopy(self)
        if not save_datasets:
            for atr in [  # TODO update
                "train_dataset",
                "train_loader",
                "train_safe_dataset",
                "valid_dataset",
            ]:
                try:
                    delattr(model_copy, atr)
                except AttributeError:
                    pass
        torch.save(model_copy, path)

    def predict(
        self,
        samples,
        batch_size=1,
        num_workers=0,
        activation_layer=None,  # softmax','sigmoid','softmax_and_logit', None
        binary_preds=None,  #'single_target','multi_target', None
        threshold=0.5,
        error_log=None,
        split_files_into_clips=True,
        overlap_fraction=0,  # overlap between consecutive clips, eg 0.5
        final_clip=None,
        augmentation_on=False,
        unsafe_samples_log=None,
    ):
        """Generate predictions on a dataset

        Choose to return any combination of scores, labels, and single-target or
        multi-target binary predictions. Also choose activation layer for scores
        (softmax, sigmoid, softmax then logit, or None).

        Note: the order of returned dataframes is (scores, preds, labels)

        Args:
            samples:
                the files to generate predictions for. Can be:
                - a dataframe with index containing audio paths, OR
                - a list (or np.ndarray) of audio file paths
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
                prediction threshold(s) for sigmoid scores. Only relevant when
                binary_preds == 'multi_target'
            error_log:
                if not None, saves a list of files that raised errors to
                the specified file location [default: None]
            overlap_fraction: fraction of overlap between consecutive clips when
                predicting on clips of longer audio files. For instance, 0.5
                gives 50% overlap between consecutive clips.
            final_clip: see [#TODO?]
            augmentation_on: default False. preprocessor.augmentation_on
                will be set to this value. If True, any Augmentations in
                preprocessor will be performed. If False, they will be skipped.
            unsafe_samples_log: if not None, samples that failed to preprocess
                will be listed in this text file.

        Returns:
            scores: df of post-activation_layer scores
            predictions: df of 0/1 preds for each class
            unsafe_samples: list of samples that failed to preprocess

        Note: if loading an audio file raises a PreprocessingError, the scores
            and predictions for that sample will be np.nan

        Note: if no return type is selected for `binary_preds`, returns None
        instead of a DataFrame for `predictions`
        """
        # validate type of samples: list, np array, or df
        if type(samples) == list or type(samples) == np.ndarray:
            prediction_df = pd.DataFrame(index=samples)
        elif type(samples) == pd.DataFrame:
            prediction_df = samples
        else:
            raise ValueError(
                f"samples must be type list, np.ndarray, or pd.DataFrame, was {type(samples)}."
            )
        # TODO: write test for type handling
        if len(prediction_df.columns) > 0:
            if not list(self.classes) == list(prediction_df.columns):
                warnings.warn(
                    "The columns of input samples df differ from `model.classes`."
                )

        if len(prediction_df) < 1:
            warnings.warn("prediction_df has zero samples. Returning None.")
            return None, None, None
        # TODO: write test for zero length warning

        prediction_dataset = self.preprocessor.sample(n=0)  # TODO make/use helper
        prediction_dataset.label_df = prediction_df
        prediction_dataset.augmentation_on = augmentation_on
        if split_files_into_clips:  # TODO: handle missing files?
            prediction_dataset.clip_times_df = make_clip_df(
                prediction_df.index.values,
                prediction_dataset.sample_duration,
                overlap_fraction * prediction_dataset.sample_duration,
                final_clip,
            )
            # update "label_df" so that index matches clip_times_df
            prediction_dataset.label_df = prediction_dataset.clip_times_df[[]]
        # TODO: add method 'eval()': runs predict then compares to labels

        # move network to device
        self.network.to(self.device)

        self.network.eval()

        # SafeDataset will not fail on bad files,
        # but will provide a different sample! Later we go back and replace scores
        # with np.nan for the bad samples (using safe_dataset._unsafe_indices)
        # this approach to error handling feels hacky
        # however, returning None would break the batching of samples
        safe_dataset = SafeDataset(prediction_dataset, unsafe_behavior="substitute")

        self.dataloader = torch.utils.data.DataLoader(
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

        # disable gradient updates during inference
        with torch.set_grad_enabled(False):

            for batch in self.dataloader:
                # get batch of Tensors
                batch_tensors = batch["X"].to(self.device)
                batch_tensors.requires_grad = False

                # forward pass of network: feature extractor + classifier
                logits = self.network.forward(batch_tensors)

                ### Activation layer ###
                scores = apply_activation_layer(logits, activation_layer)

                ### Binary predictions ###
                batch_preds = tensor_binary_predictions(
                    scores=logits, mode=binary_preds, threshold=threshold
                )

                # disable gradients on returned values
                total_scores.append(scores.detach().cpu().numpy())
                total_preds.append(batch_preds.float().detach().cpu().numpy())

        # aggregate across all batches
        total_scores = np.concatenate(total_scores, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)

        # replace scores/preds with nan for samples that failed in preprocessing
        # this feels hacky (we predicted on substitute-samples rather than
        # skipping the samples that failed preprocessing)
        total_scores[safe_dataset._unsafe_indices, :] = np.nan
        if binary_preds is not None:
            total_preds[safe_dataset._unsafe_indices, :] = np.nan

        # return 2 DataFrames with same index/columns as prediction_dataset's df
        # use None for placeholder if no preds
        samples = prediction_dataset.label_df.index.values
        score_df = pd.DataFrame(index=samples, data=total_scores, columns=self.classes)
        if split_files_into_clips:  # return a multi-index
            score_df.index = pd.MultiIndex.from_frame(
                prediction_dataset.clip_times_df.reset_index()
            )
        # binary 0/1 predictions
        if binary_preds is None:
            pred_df = None
        else:
            pred_df = pd.DataFrame(
                index=samples, data=total_preds, columns=self.classes
            )
            if split_files_into_clips:  # return a multi-index
                pred_df.index = pd.MultiIndex.from_frame(
                    prediction_dataset.clip_times_df.reset_index()
                )

        # warn the user if there were unsafe samples (failed to preprocess)
        unsafe_samples = safe_dataset.report(log=unsafe_samples_log)

        return score_df, pred_df, unsafe_samples


class CnnResampleLoss(CNN):
    """Subclass of CNN with ResampleLoss.

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
        class_frequency = (
            torch.tensor(self.train_dataset.label_df.values).sum(0).to(self.device)
        )

        # initializing ResampleLoss requires us to pass class_frequency
        self.loss_fn = self.loss_cls(class_frequency)


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
    - Uses ResampleLoss
    """

    def __init__(self, classes, single_target=False, use_pretrained=True):

        self.classes = classes
        architecture = cnn_architectures.resnet18(
            num_classes=len(self.classes), use_pretrained=use_pretrained
        )
        super(Resnet18Multiclass, self).__init__(
            architecture, self.classes, single_target
        )
        self.name = "Resnet18Multiclass"

        # optimization parameters for parts of the networks - see
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options
        self.optimizer_params = {
            "feature": {  # optimizer parameters for feature extraction layers
                # "params": self.network.feature.parameters(),
                "lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "classifier": {  # optimizer parameters for classification layers
                # "params": self.network.classifier.parameters(),
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
        param_dict = self.optimizer_params
        # in torch's resnet18, the classifier layer is called "fc"
        feature_extractor_params_list = [
            param
            for name, param in self.network.named_parameters()
            if not name.split(".")[0] == "fc"
        ]
        classifier_params_list = [
            param
            for name, param in self.network.named_parameters()
            if name.split(".")[0] == "fc"
        ]
        param_dict["feature"]["params"] = feature_extractor_params_list
        param_dict["classifier"]["params"] = classifier_params_list
        return self.optimizer_cls(param_dict.values())


class Resnet18Binary(CNN):
    """Subclass of CNN with Resnet18 architecture

    This subclass allows separate training parameters
    for the feature extractor and classifier via optimizer_params

    Args:
        classes:
            list of class names. Must match with training dataset classes.
        single_target:
            - True: model expects exactly one positive class per sample
            - False: samples can have an number of positive classes
            [default: False]

    """

    def __init__(self, classes, use_pretrained=True):
        assert len(classes) == 2, "binary model must have 2 classes"

        single_target = True  # binary model is always single-target
        self.classes = classes
        architecture = cnn_architectures.resnet18(
            num_classes=len(self.classes), use_pretrained=use_pretrained
        )
        super(Resnet18Binary, self).__init__(architecture, self.classes, single_target)
        self.name = "Resnet18Binary"

        # optimization parameters for separate feature extractor and classifier
        # see: https://pytorch.org/docs/stable/optim.html#per-parameter-options
        self.optimizer_params = {
            "feature": {  # optimizer parameters for feature extraction layers
                # "params": self.network.feature.parameters(),
                "lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "classifier": {  # optimizer parameters for classification layers
                # "params": self.network.classifier.parameters(),
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
        param_dict = self.optimizer_params
        feature_extractor_params_list = [
            param
            for name, param in self.network.named_parameters()
            if not name.split(".")[0] == "fc"
        ]
        classifier_params_list = [
            param
            for name, param in self.network.named_parameters()
            if name.split(".")[0] == "fc"
        ]
        param_dict["feature"]["params"] = feature_extractor_params_list
        param_dict["classifier"]["params"] = classifier_params_list
        return self.optimizer_cls(param_dict.values())


class InceptionV3(CNN):
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

    def _train_epoch(self, train_loader):
        """perform forward pass, loss, backpropagation for one epoch

        need to override parent because Inception returns different outputs
        from the forward pass (final and auxiliary layers)

        Returns: (targets, predictions, scores) on training files
        """
        self.network.train()

        total_tgts = []
        total_preds = []
        total_scores = []

        for batch_idx, batch_data in enumerate(train_loader):
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
                N = len(train_loader)
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


def load_model(path, device=None):
    """load a saved model object

    Args:
        path: file path of saved model
        device: which device to load into, eg 'cuda:1'
        [default: None] will choose first gpu if available, otherwise cpu

    Returns:
        a model object with loaded weights
    """
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    model = torch.load(path, map_location=device)
    model.device = device
    return model


def load_outdated_model(path, model_class, architecture_constructor=None, device=None):
    """load a CNN saved with a previous version of OpenSoundscape

    This function enables you to load models saved with opso 0.4.x, 0.5.x, and 0.6.0 when using >=0.6.1.
    For models created with 0.6.1 and above, use load_model(path) which is more robust.

    Note: If you are loading a model created with opensoundscape 0.4.x, you most likely want to specify
    `model_class = opensoundscape.torch.models.CnnResnet18Binary`. If your model was created with
    opensoundscape 0.5.x or 0.6.0, you need to choose the appropriate class.

    Note: for future use of the loaded model, you can simply call
    `model.save(path)` after creating it, then reload it with
    `model = load_model(path)`. The saved model will be fully compatible with opensoundscape >=0.6.1.

    Examples:
    ```
    #load a binary resnet18 model from opso 0.4.x, 0.5.x, or 0.6.0
    from opensoundscape.torch.models.cnn import Resnet18Binary
    model = load_outdated_model('old_model.tar',model_class=Resnet18Binary)

    #load a resnet50 model of class CNN created with opso 0.5.0
    from opensoundscape.torch.models.cnn import CNN
    from opensoundscape.torch.architectures.cnn_architectures import resnet50
    model_050 = load_outdated_model('opso050_pytorch_model_r50.model',model_class=CNN,architecture_constructor=resnet50)
    ```

    Args:
        path: path to model file, ie .model or .tar file
        model_class: the opensoundscape class to create,
            eg CNN, CnnResampleLoss, or Resnet18Binary from opensoundscape.torch.models.cnn
        architecture_constructor: the *function* that creates desired cnn architecture
            eg opensoundscape.torch.architectures.cnn_architectures.resnet18
            Note: this is only required for classes that take the architecture as an input, for instance
            CNN or CnnResampleLoss. It's not required for e.g. Resnet18Binary or InceptionV3 which
            internally create a specific architecture.
        device: optionally specify a device to map tensors onto, eg 'cpu', 'cuda:0', 'cuda:1'[default: None]
            - if None, will choose cuda:0 if cuda is available, otherwise chooses cpu

    Returns:
        a cnn model object with the weights loaded from the saved model
    """
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    # use torch to load the saved model object
    model_dict = torch.load(path, map_location=device)

    if type(model_dict) != dict:
        raise ValueError(
            "This model was saved with a version of opensoundscape >=0.6.1. "
            "Use opensoundcape.torch.models.cnn.load_model() instead of this function."
        )

    # get the list of classes
    if "classes" in model_dict:
        classes = model_dict["classes"]
    elif "labels_yaml" in model_dict:
        import yaml

        classes = list(yaml.safe_load(model_dict["labels_yaml"]).values())
    else:
        raise ValueError("Could not get a list of classes from the saved model.")

    # try to construct a model object
    try:
        model = model_class(classes=classes)
    except TypeError:  # may require us to specify the architecture
        architecture = architecture_constructor(
            num_classes=len(classes), use_pretrained=False
        )
        model = model_class(architecture=architecture, classes=classes)

    # rename keys of resnet18 architecture from 0.4.x-0.6.0 to match pytorch resnet18 keys
    model_dict["model_state_dict"] = {
        k.replace("classifier.", "fc.").replace("feature.", ""): v
        for k, v in model_dict["model_state_dict"].items()
    }

    # load the state dictionary of the network, allowing mismatches
    mismatched_keys = model.network.load_state_dict(
        model_dict["model_state_dict"], strict=False
    )
    print(mismatched_keys)

    # if there's no record of single-tartet vs multitarget, it' single target from opso 0.4.x
    try:
        single_target = model_dict["single_target"]
    except KeyError:
        single_target = True

    model.single_target = single_target

    return model
