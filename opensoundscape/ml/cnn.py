"""classes for pytorch machine learning models in opensoundscape

For tutorials, see notebooks on opensoundscape.org
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import os
import copy

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

import opensoundscape
from opensoundscape.ml import cnn_architectures
from opensoundscape.ml.utils import apply_activation_layer, check_labels
from opensoundscape.preprocess.preprocessors import (
    SpectrogramPreprocessor,
    BasePreprocessor,
    preprocessor_from_dict,
)
from opensoundscape.preprocess import io
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.ml.cnn_architectures import inception_v3
from opensoundscape.sample import collate_audio_samples
from opensoundscape.utils import identity
from opensoundscape.logging import wandb_table

from opensoundscape.ml.cam import CAM
import pytorch_grad_cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MulticlassAveragePrecision,
    MulticlassAUROC,
)
from torchmetrics import Accuracy

import opensoundscape
from opensoundscape.ml import cnn_architectures
from opensoundscape.ml.loss import (
    BCEWithLogitsLoss_hot,
    CrossEntropyLoss_hot,
    ResampleLoss,
)

from opensoundscape.ml.dataloaders import SafeAudioDataloader
from opensoundscape.sample import collate_audio_samples

import warnings


MODEL_CLS_DICT = dict()


def list_model_classes():
    """return list of available action function keyword strings
    (can be used to initialize Action class)
    """
    return list(MODEL_CLS_DICT.keys())


def register_model_cls(model_cls):
    """add class to MODEL_CLS_DICT

    this allows us to recreate the class when loading saved model file with load_model()

    """
    # register the model in dictionary
    MODEL_CLS_DICT[io.build_name(model_cls)] = model_cls
    # return the function
    return model_cls


class BaseModule:
    def __init__(self):
        """base class for pytorch and lightning models in opensoundscape

        This class is intended to be subclassed by classes with more customized functionality.
        For example, see SpectrogramModule, SpectrogramClassifier, and LightningSpectrogramModule.
        """
        super().__init__()

        self.name = "BaseModule"
        self.opensoundscape_version = opensoundscape.__version__

        # TODO: set up logging of hyperparameters to arbitary logger

        # model characteristics # TODO: maybe group into self.training_state dictionary
        self.scheduler = None
        """torch.optim.lr_scheduler object for learning rate scheduling"""

        self.torch_metrics = {"accuracy": Accuracy("binary")}
        """specify torchmetrics "name":object pairs to compute metrics during training/validation"""

        self.score_metric = "accuracy"
        """choose one of the keys in self.torch_metrics to use as the overall score metric
        
        this metric will be used to determine the best model during training
        """

        self.preprocessor = BasePreprocessor()
        """an instance of BasePreprocessor or subclass that preprocesses audio samples into tensors

        The preprocessor contains .pipline, and ordered set of Actions to run

        preprocessor will have attributes .sample_duration (seconds)
        and .height, .width, .channels for output shape (input shape to self.network)
        
        The pipeline can be modified by adding or removing actions, and by modifying parameters:
        ```python
            my_obj.preprocessor.remove_action('add_noise')
            my_obj.preprocessor.insert_action('add_noise',Action(my_function),after_key='frequency_mask')
        ```

        Or, the preprocessor can be replaced with a different or custom preprocessor, for instance:
        ```python
        from opensoundscape.preprocess import AudioPreprocessor
        my_obj.preprocessor = AudioPreprocessor(
            sample_duration=5, 
            sample_rate=22050
        )
        # this preprocessor returns 1d arrays of the audio signal
        ```
        """

        # to use a custom DataLoader or Sampler, change these attributes
        # to the custom class (init must take same arguments)
        # or override .train_dataloader(), .predict_dataloader()
        self.train_dataloader_cls = SafeAudioDataloader
        """a DataLoader class to use for training, defaults to SafeAudioDataloader"""
        self.inference_dataloader_cls = SafeAudioDataloader
        """a DataLoader class to use for inference, defaults to SafeAudioDataloader"""

        self.network = torch.nn.Module()
        """a pytorch Module such as Resnet18 or a custom object"""

        ### loss function ###
        self.loss_fn = BCEWithLogitsLoss_hot()
        """specify a loss function to use for training, eg BCEWithLogitsLoss_hot,
        
        by initializing a callable object or passing a function
        """

        self.optimizer_params = {
            "class": torch.optim.SGD,
            "kwargs": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "classifier_lr": None,  # optionally specify different lr for classifier layer
        }
        """optimizer settings: dictionary with "class" and "kwargs" to class.__init__
        
        for example, to use Adam optimizer set:
        ```python
        my_instance.optimizer_params = {
            "class": torch.optim.Adam,
            "kwargs": {
                "lr": 0.001,
                "weight_decay": 0.0005,
            },
        }
        ```
        """

        self.lr_scheduler_params = {
            "class": torch.optim.lr_scheduler.StepLR,
            "kwargs": {
                "step_size": 10,
                "gamma": 0.7,
            },
        }
        """learning rate schedule: dictionary with "class" and "kwargs" to class.__init__
        
        for example, to use Cosine Annealing, set:
        ```python
        model.lr_scheduler_params = {
            "class": torch.optim.lr_scheduler.CosineAnnealingLR,
            "kwargs":{
                "T_max": n_epochs,
                "eta_min": 1e-7,
                "last_epoch":self.current_epoch-1
        }
        ```
        """

        self.use_amp = False
        """if True, uses automatic mixed precision for training"""

    def training_step(self, samples, batch_idx):
        """a standard Lightning method used within the training loop, acting on each batch

        returns loss

        Effects:
            logs metrics and loss to the current logger
        """
        batch_tensors, batch_labels = samples
        batch_tensors = batch_tensors.to(self.device)
        batch_labels = batch_labels.to(self.device)

        batch_size = len(batch_tensors)

        # automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if "cuda" in str(self.device):
            device_type = "cuda"
            dtype = torch.float16
        else:
            device_type = "cpu"
            dtype = torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype):
            output = self.network(batch_tensors)
            loss = self.loss_fn(output, batch_labels)
        if not self.lightning_mode:
            # if not using Lightning, we manually call
            # loss.backward() and optimizer.step()
            # Lightning does this behind the scenes
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()  # set_to_none=True here can modestly improve performance
        # else:
        #     output = self.network(batch_tensors)
        #     loss = self.loss_fn(output, batch_labels)
        #     if not self.lightning_mode:
        #         # if not using Lightning, we manually call
        #         # loss.backward() and optimizer.step()
        #         # Lightning does this behind the scenes
        #         loss.backward()
        #         self.optimizer.step()
        #         self.optimizer.zero_grad()

        # single-target torchmetrics expect labels as integer class indices rather than one-hot
        y = batch_labels.argmax(dim=1) if self.single_target else batch_labels
        # TODO: does not allow soft labels, but some torchmetrics expect long type?
        batch_metrics = {
            f"train_{name}": metric.to(self.device)(
                output.detach(), y.detach().long()
            ).cpu()
            for name, metric in self.torch_metrics.items()
        }

        if self.lightning_mode:
            self.log(
                f"train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch_tensors),
            )
            self.log_dict(
                batch_metrics, on_epoch=True, on_step=False, batch_size=batch_size
            )
            # when on_epoch=True, compute() is called to reset the metric at epoch end

        return loss

    # def predict_step(self, batch): #runs forward() if we don't override default

    @property
    def classifier_params(self):
        """return the parameters of the classifier layer of the network

        override this method if the classifier parameters should be retrieved in a different way
        """
        return self.classifier.parameters()

    def configure_optimizers(
        self,
        reset_optimizer=False,
        restart_scheduler=False,
    ):
        """standard Lightning method to initialize an optimizer and learning rate scheduler

        Lightning uses this function at the start of training. Weirdly it needs to
        return {"optimizer": optimizer, "scheduler": scheduler}.

        Initializes the optimizer and  learning rate scheduler using the parameters
        self.optimizer_params and self.scheduler_params, which are dictionaries with a key
        "class" and a key "kwargs" (containing a dictionary of keyword arguments to initialize
        the class with). We initialize the class with the kwargs and the appropriate
        first argument: optimizer=opt_cls(self.parameters(), **opt_kwargs) and
        scheduler=scheduler_cls(optimizer, **scheduler_kwargs)

        You can also override this method and write one that returns
        {"optimizer": optimizer, "scheduler": scheduler}

        Uses the attributes:
        - self.optimizer_params: dictionary with "class" key such as torch.optim.Adam,
            and "kwargs", dict of keyword args for class's init
        - self.scheduler_params: dictionary with "class" key such as
            torch.optim.lr_scheduler.StepLR, and and "kwargs", dict of keyword args for class's init
        - self.lr_scheduler_step: int, number of times lr_scheduler.step() has been called
            - can set to -1 to restart learning rate schedule
            - can set to another value to start lr scheduler from an arbitrary position

        Note: when used by lightning, self.optimizer and self.scheduler should not be modified
        directly, lightning handles these internally. Lightning will call the method without
        passing reset_optimizer or restart_scheduler, so default=False results in not modifying .optimizer or .scheduler

        Documentation:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        Args:
            reset_optimizer: if True, initializes the optimizer from scratch even if self.optimizer is not None
            reset_scheduler: if True, initializes the scheduler from scratch even if self.scheduler is not None
        Returns:
            dictionary with keys "optimizer" and "scheduler" containing the
            optimizer and learning rate scheduler objects to use during training
        """

        if reset_optimizer:
            self.optimizer = None
        if restart_scheduler:
            self.scheduler = None
            self.lr_scheduler_step = -1

        # create optimizer, if it doesn't exist yet
        # self.optimizer_params dictionary has "class" and "kwargs" keys
        # copy the kwargs dict to avoid modifying the original values
        # when the optimizer is stepped
        optimizer = self.optimizer_params["class"](
            self.network.parameters(), **self.optimizer_params["kwargs"].copy()
        )

        if self.optimizer_params["classifier_lr"] is not None:
            # customize the learning rate of the classifier layer
            try:
                # Cannot check `param in param_list`. Instead, compare the objects' ids.
                # see https://discuss.pytorch.org/t/confused-by-runtimeerror-when-checking-for-parameter-in-list/211308
                classifier_param_ids = {id(p) for p in self.classifier_params}
                # remove these parameters from their current group
                for param_group in optimizer.param_groups:
                    param_group["params"] = [
                        p
                        for p in param_group["params"]
                        if id(p) not in classifier_param_ids
                    ]
            except Exception as e:
                raise ValueError(
                    "Could not access self.classifier.parameters(). "
                    "Make sure self.classifier propoerty returns a torch.nn.Module object."
                ) from e

            # add them to a new group with custom learning rate
            optimizer.add_param_group(
                {
                    "params": self.classifier_params,
                    "lr": self.optimizer_params["classifier_lr"],
                }
            )

        if hasattr(self, "optimizer") and self.optimizer is not None:
            # load the state dict of the previously existing optimizer,
            # updating the params references to match current instance of self.network
            try:
                opt_state_dict = self.optimizer.state_dict().copy()
                opt_state_dict["params"] = self.network.parameters()
                optimizer.load_state_dict(opt_state_dict)
            except:
                warnings.warn(
                    "attempt to load state dict of existing self.optimizer failed. "
                    "Optimizer will be initialized from scratch"
                )
            # TODO: write tests for lightning to check behavior of continued training

        # create learning rate scheduler
        # self.scheduler_params dictionary has "class" key and kwargs for init
        # additionally use self.lr_scheduler_step to initialize the scheduler's "last_epoch"
        # which determines the starting point of the learning rate schedule
        # (-1 restarts the lr schedule from the initial lr)
        args = self.lr_scheduler_params["kwargs"].copy()
        args.update({"last_epoch": self.lr_scheduler_step})
        scheduler = self.lr_scheduler_params["class"](optimizer, **args)

        if self.scheduler is not None:
            # load the state dict of the previously existing scheduler,
            # updating the params references to match current instance of self.network
            try:
                scheduler_state_dict = self.scheduler.state_dict().copy()
                # scheduler_state_dict["params"] = self.network.parameters()
                scheduler.load_state_dict(scheduler_state_dict)
            except:
                warnings.warn(
                    "attempt to load state dict of existing self.scheduler failed. "
                    "Scheduler will be initialized from scratch"
                )

        return {"optimizer": optimizer, "scheduler": scheduler}

    def validation_step(self, samples, batch_idx, dataloader_idx=0):
        """currently only used for lightning

        not used by SpectrogramClassifier"""
        batch_tensors, batch_labels = samples
        batch_tensors = batch_tensors.to(self.device)
        batch_labels = batch_labels.to(self.device)

        batch_size = len(batch_tensors)
        logits = self.network(batch_tensors)
        loss = self.loss_fn(logits, batch_labels)

        # compute and log any metrics in self.torch_metrics
        # TODO: consider using validation set names rather than integer index
        # (would have to store a set of names for the validation set)
        batch_metrics = {
            f"val{dataloader_idx}_{name}": metric.to(self.device)(
                logits.detach(), batch_labels.detach().int()
            ).cpu()
            for name, metric in self.torch_metrics.items()
        }
        if self.lightning_mode:
            self.log_dict(
                batch_metrics, on_epoch=True, on_step=False, batch_size=batch_size
            )
            # when on_epoch=True, compute() is called to reset the metric at epoch end
            self.log(
                f"val{dataloader_idx}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch_tensors),
            )
        return loss

    def train_dataloader(
        self,
        samples,
        bypass_augmentations=False,
        collate_fn=collate_audio_samples,
        **kwargs,
    ):
        """generate dataloader for training

        train_loader samples batches of images + labels from training set

        Args: see self.train_dataloader_cls docstring for arguments
            **kwargs: any arguments to pass to the DataLoader __init__
            Note: some arguments are fixed and should not be passed in kwargs:
            - shuffle=True: shuffle samples for training
            - bypass_augmentations=False: apply augmentations to training samples

        """
        return self.train_dataloader_cls(
            samples=samples,
            preprocessor=self.preprocessor,
            split_files_into_clips=True,
            clip_overlap=0,
            final_clip=None,
            bypass_augmentations=bypass_augmentations,
            shuffle=True,  # SHUFFLE SAMPLES because we are training
            # use pin_memory=True when loading files on CPU and training on GPU
            pin_memory=False if self.device == torch.device("cpu") else True,
            collate_fn=collate_fn,
            **kwargs,
        )

    def predict_dataloader(self, samples, collate_fn=collate_audio_samples, **kwargs):
        """generate dataloader for inference (predict/validate/test)

        Args: see self.inference_dataloader_cls docstring for arguments
            **kwargs: any arguments to pass to the DataLoader __init__
            Note: these arguments are fixed and should not be passed in kwargs:
            - shuffle=False: retain original sample order
        """
        # for convenience, convert str/pathlib.Path to list of length 1
        if isinstance(samples, (str, Path)):
            samples = [samples]

        return self.inference_dataloader_cls(
            samples=samples,
            preprocessor=self.preprocessor,
            shuffle=False,  # keep original order
            pin_memory=False if self.device == torch.device("cpu") else True,
            collate_fn=collate_fn,
            **kwargs,
        )


class ChannelDimCheckError(Exception):
    pass


def get_channel_dim(model):

    # Get the first layer
    first_layer = list(model.children())[0]

    # If the first layer is a Sequential container, get its first module
    while isinstance(first_layer, torch.nn.Sequential):
        first_layer = list(first_layer.children())[0]

    # Get the number of input channels
    # try checking first layer's .in_channelss then .in_featuers
    try:
        return first_layer.in_channels
    except:
        raise ChannelDimCheckError(
            "Couldn't access .in_channels or .in_features of first layer"
        )


class SpectrogramModule(BaseModule):
    """Parent class for both SpectrogramClassifier (pytorch) and LightningSpectrogramModule (lightning)

    implements functionality that is shared between both pure PyTorch and Lightning classes/workflows

    Args:
        architecture: a pytorch Module such as Resnet18 or a custom object
        classes: list of class names
        sample_duration: duration of audio samples in seconds
        single_target: if True, predict only class with max score
        channels: number of channels in input data
        sample_height: height of input data
        sample_width: width of input data
        preprocessor_dict: dictionary defining preprocessor and parameters,
            can be generated with preprocessor.to_dict()
            if not None, will override other preprocessor arguments
            (sample_duration, sample_height, sample_width, channels)
        preprocessor_cls:
            a class object that inherits from BasePreprocessor
            if preprocessor_dict is None, this class will be instantiated to set self.preprocessor
        **preprocessor_kwargs: additional arguments to pass to the initialization of the preprocessor class
            this is ignored if preprocessor_dict is not None
    """

    def __init__(
        self,
        architecture,
        classes,
        sample_duration,
        single_target=False,
        preprocessor_dict=None,
        preprocessor_cls=SpectrogramPreprocessor,
        **preprocessor_kwargs,
    ):
        super().__init__()
        self.classes = classes
        self._single_target = single_target
        self.name = "SpectrogramModule"

        self.use_amp = False  # use automatic mixed precision
        self.lightning_mode = False  # True: skip things done automatically by Lightning

        self.lr_scheduler_step = -1
        """track number of calls to lr_scheduler.step()
        
        set to -1 to restart learning rate schedule from initial lr
        
        this value is used to initialize the lr_scheduler's `last_epoch` parameter
        it is tracked separately from self.current_epoch because the lr_scheduler
        might be stepped more or less than 1 time per epoch

        Note that the initial learning rate is set via self.optimizer_params['kwargs']['lr']
        """

        ### PREPROCESSOR ###
        preprocessor_kwargs.update(sample_duration=sample_duration)

        if preprocessor_dict is None:
            self.preprocessor = preprocessor_cls(**preprocessor_kwargs)
        else:
            # reload a preprocessor serialized to a dictionary
            # finds the class using preprocessors.PREPROCESSOR_DICT lookup
            # note that some preprocessor settings may not be saved in the dictionary
            # in particular, Overlay augmentation's overlay_df and criterion_fn
            self.preprocessor = preprocessor_from_dict(preprocessor_dict)

        ### ARCHITECTURE ###
        # allow user to pass a string, in which case we look up the architecture
        # in cnn_architectures.ARCH_DICT and instantiate it
        if type(architecture) == str:
            assert architecture in cnn_architectures.list_architectures(), (
                f"architecture must be a pytorch model object or string matching "
                f"one of cnn_architectures.list_architectures() options. Got {architecture}"
            )
            architecture = cnn_architectures.ARCH_DICT[architecture](
                len(classes), num_channels=self.preprocessor.channels
            )
        else:
            assert issubclass(
                type(architecture), torch.nn.Module
            ), "architecture must be a string or an instance of a subclass of torch.nn.Module"

            # warn user if this architecture is not "registered", since we won't be able to reload it
            if (
                not hasattr(architecture, "constructor_name")
                or architecture.constructor_name
                not in cnn_architectures.ARCH_DICT.keys()
            ):
                warnings.warn(
                    """
                    This architecture is not listed in opensoundscape.ml.cnn_architectures.ARCH_DICT.
                    It will not be available for loading after saving the model with .save() (unless using pickle=True). 
                    To make it re-loadable, define a function that generates the architecture from arguments: (n_classes, n_channels) 
                    then use opensoundscape.ml.cnn_architectures.register_architecture() to register the generating function.

                    The function can also set the returned object's .constructor_name to the registered string key in ARCH_DICT
                    to avoid this warning and ensure it is reloaded correctly by opensoundscape.ml.load_model().

                    See opensoundscape.ml.cnn_architectures module for examples of constructor functions
                    """
                )
            # try to update channels arg to match architecture
            try:
                arch_channels = get_channel_dim(architecture)
                if self.preprocessor.channels != arch_channels:
                    warnings.warn(
                        f"Modifying .preprocessor to match architecture's expected number of channels ({arch_channels}) "
                        f"(originally {self.preprocessor.channels})."
                    )
                    self.preprocessor.channels = arch_channels
            except:
                # can we try to check if first layer expects input with channels=channels?
                warnings.warn(
                    f"Failed to detect expected # input channels of this architecture."
                    "Make sure your architecture expects the number of channels "
                    f"equal to `channels` argument {self.preprocessor.channels}). "
                    f"Pytorch architectures generally expect 3 channels by default."
                )

        self.network = architecture
        """a pytorch Module such as Resnet18 or a custom object

        for convenience, __init__ also allows user to provide string matching
        a key from opensoundscape.ml.cnn_architectures.ARCH_DICT.
        
        List options: `opensoundscape.ml.cnn_architectures.list_architectures()`
        """

        ### LOSS FUNCTION ###
        # choose canonical loss for single or multi-target classification
        # can override by setting `self.loss_fn=...`
        if self.single_target:
            self.loss_fn = CrossEntropyLoss_hot()
        else:
            self.loss_fn = BCEWithLogitsLoss_hot()

        ### EVALUATION METRICS ###
        # These metrics are a good starting point
        # for single and multi-target classification
        # User can add/remove metrics as desired.
        self._init_torch_metrics()

        ### Logging ###
        self.wandb_logging = dict(
            n_preview_samples=8,  # before train/predict, log n random samples
            top_samples_classes=None,  # specify list of classes to see top samples from
            n_top_samples=3,  # after prediction, log n top scoring samples per class
            # logs histograms of params & grads every n batches;
            watch_freq=10,  # use  None for no logging of params & grads
            gradcam=True,  # if True, logs GradCAMs for top scoring samples during predict()
            # log the model graph to wandb - seems to cause issues when attempting to
            # continue training the model, so True is not recommended
            log_graph=False,
        )

    def change_classes(self, new_classes):
        """change the classes that the model predicts

        replaces the network's final linear classifier layer with a new layer
        with random weights and the correct number of output features

        will raise an error if self.network.classifier_layer is not the name of
        a torch.nn.Linear layer, since we don't know how to replace it otherwise

        Args:
            new_classes: list of class names
        """
        assert len(new_classes) > 0, "new_classes must have >0 elements"

        assert isinstance(self.classifier, torch.nn.Linear), (
            f"Expected self.classifier to be a torch.nn.Linear layer, "
            f"but found {type(self.classifier)}. Cannot automatically replace this layer to "
            "achieve desired number of output features."
        )

        # replace fully-connected final classifier layer
        clf_layer_name = self.network.classifier_layer
        new_layer = cnn_architectures.change_fc_output_size(
            self.classifier, len(new_classes)
        )
        cnn_architectures.set_layer_from_name(self.network, clf_layer_name, new_layer)

        # update class list
        self.classes = new_classes

        # re-initialize metrics, using the new number of classes
        self._init_torch_metrics()

    @property
    def classifier(self):
        """return the classifier layer of the network, based on .network.classifier_layer string"""
        return self.network.get_submodule(self.network.classifier_layer)

    @property
    def single_target(self):
        return self._single_target

    @single_target.setter
    def single_target(self, st):
        """Set single_target to True or False. If changed, re-initialize torch metrics

        Args:
            st: (bool) if True, uses softmax activation for evaluation, predicting only
                class with max score. If False, uses sigmoid activation, predicting all classes
                independently.
        """
        if not self._single_target == st:
            self._log(f"Updating torchmetrics and loss_fn to match single_target={st}")
            self._single_target = st
            self._init_torch_metrics()
            self.loss_fn = CrossEntropyLoss_hot() if st else BCEWithLogitsLoss_hot()

    def _init_torch_metrics(self):
        if self.single_target:
            self.torch_metrics = {
                "map": MulticlassAveragePrecision(
                    len(self.classes), validate_args=False, thresholds=50
                ),
                # .log() doesn't allow logging lists of values, so we don't
                # log class-by-class metrics.
                # "class_ap": MulticlassAveragePrecision(
                #     len(self.classes),
                #     validate_args=False,
                #     average=None,
                #     thresholds=50,
                # ),
                "auroc": MulticlassAUROC(
                    len(self.classes),
                    validate_args=False,
                    thresholds=50,  # speeds up computation
                    average="macro",
                ),
                # "class_auroc": MulticlassAUROC(
                #     len(self.classes),
                #     validate_args=False,
                #     average=None,
                #     thresholds=50,
                # ),
            }
            self.score_metric = "auroc"
        else:
            self.torch_metrics = {
                "map": MultilabelAveragePrecision(
                    len(self.classes), validate_args=False, thresholds=50
                ),
                # "class_ap": MultilabelAveragePrecision(
                #     len(self.classes),
                #     validate_args=False,
                #     average=None,
                #     thresholds=50,
                # ),
                "auroc": MultilabelAUROC(
                    len(self.classes),
                    validate_args=False,
                    thresholds=50,
                    average="macro",
                ),
                # "class_auroc": MultilabelAUROC(
                #     len(self.classes), validate_args=False, thresholds=50, average=None
                # ),
            }
            self.score_metric = "auroc"

    def freeze_layers_except(self, train_layers=None):
        """Freeze all parameters of a model except the parameters in the target_layer(s)

        Freezing parameters means that the optimizer will not update the weights

        Modifies the model in place!

        Args:
            model: the model to freeze the parameters of
            train_layers: layer or list/iterable of the layers whose parameters should not be frozen
                For example: pass `model.classifier` to train only the classifier

        Example 1:
        ```
        freeze_all_layers_except(model, model.classifier)
        ```

        Example 2: freeze all but 2 layers
        ```
        freeze_all_layers_except(model, [model.layer1, model.layer2])
        ```
        """
        # handle single layer or list of layers
        if isinstance(train_layers, torch.nn.Module):
            train_layers = [train_layers]
        elif train_layers is None:
            train_layers = []

        for train_layer in train_layers:
            assert isinstance(
                train_layer, torch.nn.Module
            ), f"model attribute {train_layer} was not a torch.nn.Module"

        # first, disable gradients for all layers
        self.network.requires_grad_(False)

        # then, enable gradient updates for the target layers
        for train_layer in train_layers:
            train_layer.requires_grad_(True)

    def freeze_feature_extractor(self):
        """freeze all layers except self.classifier

        prepares the model for transfer learning where only the classifier is trained

        uses the attribute self.network.classifier_layer (via the .classifier attribute)
        to identify the classifier layer

        if this is not set will raise Exception - use freeze_layers_except() instead
        """
        try:
            clf_layer = self.classifier
            assert isinstance(clf_layer, torch.nn.Module)
        except Exception as e:
            raise ValueError(
                "freeze_feature_extractor() requires self.network.classifier_layer to be a string defining the classifier layer."
                "Consider using freeze_layers_except() and specifying layers to leave unfrozen."
            ) from e
        self.freeze_layers_except(train_layers=self.classifier)

    def unfreeze(self):
        """Unfreeze all layers & parameters of self.network

        Enables gradient updates for all layers & parameters

        Modifies the object in place
        """
        self.network.requires_grad_(True)


@register_model_cls
class SpectrogramClassifier(SpectrogramModule, torch.nn.Module):
    name = "SpectrogramClassifier"

    def __init__(self, *args, **kwargs):
        """defines pure pytorch train, predict, and eval methods for a spectrogram classifier

        subclasses SpectrogramModule, defines methods that are used for pure PyTorch workflow. To
        use lightning, see ml.lightning.LightningSpectrogramModule.

        Args:
            see SpectrogramModule for arguments

        Methods:
            predict: generate predictions across a set of audio files or a dataframe defining audio
            files and start/end clip times

            train: fit the machine learning model using training data and evaluate with validation
            data

            save: save the model to a file load: load the model from a file

            embed: generate embeddings for a set of audio files

            generate_samples: creates preprocessed sample tensors, same arguments as predict()

            generate_cams: generate gradient activation maps for a set of audio files

            eval: evaluate performance by applying self.torch_metrics to predictions and labels

            run_validation: test accuracy by running inference on a validation set and computing

            metrics change_classes: change the classes that the model predicts

            freeze_feature_extractor: freeze all layers except the classifier

            freeze_layers_except: freeze all parameters of a model, optionally exluding some layers

            train_dataloader: create dataloader for training predict_dataloader: create dataloader
            for inference (predict/validate/test)

            save_weights: save just the self.network state dict to a file

            load_weights: load just the self.network state dict from a file

        Editable Attributes & Properties:
            single_target: (bool) if True, predict only class with max score

            device: (torch.device or str) device to use for training and inference preprocessor:
            object defining preprocessing and augmentation operations, e.g. SpectrogramPreprocessor

            network: pytorch model object, e.g. Resnet18

            loss_fn: callable object to use for calculating loss during training, e.g.
            BCEWithLogitsLoss_hot()

            optimizer_params: (dict) with "class" and "kwargs" keys for class.__init__(**kwargs)

            lr_scheduler_params: (dict) with "class" and "kwargs" for class.__init__(**kwargs)
            use_amp: (bool) if True, uses automatic mixed precision for training wandb_logging:
            (dict) settings for logging to Weights and Biases

            score_metric: (str) name of the metric for overall evaluation - one of the keys in
            self.torch_metrics

            log_file: (str) path to save output to a text file logging_level: (int) amt of logging
            to log file. 0 for nothing, 1,2,3 for increasing logged info

            verbose: (int) amt of logging to stdout. 0 for nothing, 1,2,3 for increasing printed
            output

        Other attributes:
            torch_metrics: dictionary of torchmetrics name:object pairs to use for calculating
            metrics
                - override _init_torch_metrics() method in a subclass rather than modifying directly
                - in general, if self.single_target is True, metrics will be called with
                  metric(predictions, labels) where predictions is shape (n_samples,n_classes) and
                  labels has integer labels with shape (n_samples,). If single_target is False,
                  instead labels are multi-hot encoded and have shape (n_samples,n_classes)
            classes: list of class names
                - set the class list with __init__() or change_classes(), rather than modifying
                  directly
                This ensures that other parameters like self.torch_metrics are updated accordingly
        """
        super().__init__(*args, **kwargs)

        self.log_file = None
        """specify a path to save output to a text file"""
        self.logging_level = 1
        """amount of logging to self.log_file. 0 for nothing, 1,2,3 for increasing logged info"""
        self.verbose = 1
        """amount of logging to stdout. 0 for nothing, 1,2,3 for increasing printed output"""
        self.scheduler = None  # learning rate scheduler, initialized during configure_optimizers() call
        self.optimizer = None  # optimizer during training , initialized during configure_optimizers() call
        self.current_epoch = 0
        """track number of trained epochs"""

        ### metrics ###
        self.loss_hist = {}
        """dictionary of epoch:mean batch loss during training"""
        self.train_metrics = {}
        self.valid_metrics = {}

        self.device = _gpu_if_available()  # device to use for training and inference

    def _log(self, message, level=1):
        txt = str(message)
        if self.logging_level >= level and self.log_file is not None:
            with open(self.log_file, "a") as logfile:
                logfile.write(txt + "\n")
        if self.verbose >= level:
            print(txt)

    def predict(
        self,
        samples,
        batch_size=1,
        num_workers=0,
        activation_layer=None,
        split_files_into_clips=True,
        clip_overlap=None,
        clip_overlap_fraction=None,
        clip_step=None,
        overlap_fraction=None,
        final_clip=None,
        bypass_augmentations=True,
        invalid_samples_log=None,
        raise_errors=False,
        wandb_session=None,
        return_invalid_samples=False,
        progress_bar=True,
        audio_root=None,
        **dataloader_kwargs,
    ):
        """Generate predictions on a set of samples

        Return dataframe of model output scores for each sample.
        Optional activation layer for scores
        (softmax, sigmoid, softmax then logit, or None)

        Args:
            samples:
                the files to generate predictions for. Can be:
                - a dataframe with index containing audio paths, OR
                - a dataframe with multi-index (file, start_time, end_time), OR
                - a list (or np.ndarray) of audio file paths
                - a single file path (str or pathlib.Path)
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
            split_files_into_clips:
                If True, internally splits and predicts on clips from longer audio files
                Otherwise, assumes each row of `samples` corresponds to one complete sample
            clip_overlap_fraction, clip_overlap, clip_step, final_clip:
                see `opensoundscape.utils.generate_clip_times_df`
            overlap_fraction: deprecated alias for clip_overlap_fraction
            bypass_augmentations: If False, Actions with
                is_augmentation==True are performed. Default True.
            invalid_samples_log: if not None, samples that failed to preprocess
                will be listed in this text file.
            raise_errors:
                if True, raise errors when preprocessing fails
                if False, just log the errors to unsafe_samples_log
            wandb_session: a wandb session to log to
                - pass the value returned by wandb.init() to progress log to a
                Weights and Biases run
                - if None, does not log to wandb
            return_invalid_samples: bool, if True, returns second argument, a set
                containing file paths of samples that caused errors during preprocessing
                [default: False]
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            audio_root: optionally pass a root directory (pathlib.Path or str)
                - `audio_root` is prepended to each file path
                - if None (default), samples must contain full paths to files
            **dataloader_kwargs: additional arguments to self.predict_dataloader()

        Returns:
            df of post-activation_layer scores
            - if return_invalid_samples is True, returns (df,invalid_samples)
            where invalid_samples is a set of file paths that failed to preprocess

        Effects:
            (1) wandb logging
            If wandb_session is provided, logs progress and samples to Weights
            and Biases. A random set of samples is preprocessed and logged to
            a table. Progress over all batches is logged. Afte prediction,
            top scoring samples are logged.
            Use self.wandb_logging dictionary to change the number of samples
            logged or which classes have top-scoring samples logged.

            (2) unsafe sample logging
            If unsafe_samples_log is not None, saves a list of all file paths that
            failed to preprocess in unsafe_samples_log as a text file

        Note: if loading an audio file raises a PreprocessingError, the scores
            for that sample will be np.nan

        """
        if audio_root is not None:  # add this to dataloader keyword arguments
            dataloader_kwargs.update(dict(audio_root=audio_root))
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(
            samples,
            bypass_augmentations=bypass_augmentations,
            split_files_into_clips=split_files_into_clips,
            overlap_fraction=overlap_fraction,
            clip_overlap=clip_overlap,
            clip_overlap_fraction=clip_overlap_fraction,
            clip_step=clip_step,
            final_clip=final_clip,
            batch_size=batch_size,
            num_workers=num_workers,
            raise_errors=raise_errors,
            **dataloader_kwargs,
        )

        # check for matching class list
        if len(dataloader.dataset.dataset.classes) > 0 and list(self.classes) != list(
            dataloader.dataset.dataset.classes
        ):
            warnings.warn(
                "The columns of input samples df differ from `model.classes`."
            )

        # Initialize Weights and Biases (wandb) logging
        if wandb_session is not None:
            # update the run config with information about the model
            wandb_session.config.update(self._generate_wandb_config())

            # update the run config with prediction parameters
            wandb_session.config.update(
                dict(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    activation_layer=activation_layer,
                )
            )

            # Log a table of preprocessed samples to wandb
            wandb_session.log(
                {
                    "Peprocessed_samples": wandb_table(
                        dataloader.dataset.dataset,
                        self.wandb_logging["n_preview_samples"],
                    )
                }
            )

        ### Prediction/Inference ###
        # iterate dataloader and run inference (forward pass) to generate scores
        pred_scores = self.__call__(
            dataloader=dataloader,
            wandb_session=wandb_session,
            progress_bar=progress_bar,
        )

        ### Apply activation layer ###
        pred_scores = apply_activation_layer(pred_scores, activation_layer)

        # return DataFrame with same index/columns as prediction_dataset's df
        df_index = dataloader.dataset.dataset.label_df.index
        score_df = pd.DataFrame(index=df_index, data=pred_scores, columns=self.classes)

        # warn the user if there were invalid samples (failed to preprocess)
        # and log them to a file
        invalid_samples = dataloader.dataset.report(log=invalid_samples_log)

        # log top-scoring samples per class to wandb table
        if wandb_session is not None:
            classes_to_log = self.wandb_logging["top_samples_classes"]
            if classes_to_log is None:  # pick the first few classes if none specified
                classes_to_log = self.classes
                if len(classes_to_log) > 5:  # don't accidentally log hundreds of tables
                    classes_to_log = classes_to_log[0:5]

            for i, c in enumerate(classes_to_log):
                top_samples = score_df.nlargest(
                    n=self.wandb_logging["n_top_samples"], columns=[c]
                )
                # note: the "labels" of these samples are actually prediction scores
                dataset = AudioFileDataset(
                    samples=top_samples,
                    preprocessor=self.preprocessor,
                    bypass_augmentations=True,
                )
                table = wandb_table(
                    dataset=dataset,
                    classes_to_extract=[c],
                    drop_labels=True,
                    gradcam_model=self if self.wandb_logging["gradcam"] else None,
                )
                wandb_session.log({f"Top_scoring_{c.replace(' ','_')}": table})

        if return_invalid_samples:
            return score_df, invalid_samples
        else:
            return score_df

    def generate_samples(
        self,
        samples,
        invalid_samples_log=None,
        return_invalid_samples=False,
        audio_root=None,
        **dataloader_kwargs,
    ):
        """
        Generate AudioSample objects. Input options same as .predict()

        Args:
            samples: (same as CNN.predict())
                the files to generate predictions for. Can be:
                - a dataframe with index containing audio paths, OR
                - a dataframe with multi-index (file, start_time, end_time), OR
                - a list (or np.ndarray) of audio file paths
                - a single file path as str or pathlib.Path
            see .predict() documentation for other args
            **dataloader_kwargs: any arguments to inference_dataloader_cls.__init__
                except samples (uses `samples`) and collate_fn (uses `identity`)
                (Note: default class is SafeAudioDataloader)

        Returns:
            a list of AudioSample objects
            - if return_invalid_samples is True, returns second value: list of paths to
            samples that failed to preprocess

        Example:
        ```
        from opensoundscappe.preprocess.utils import show_tensor_grid
        samples = generate_samples(['/path/file1.wav','/path/file2.wav'])
        tensors = [s.data for s in samples]
        show_tensor_grid(tensors,columns=3)
        ```
        """
        if audio_root is not None:  # add this to dataloader keyword arguments
            dataloader_kwargs.update(dict(audio_root=audio_root))
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(
            samples, collate_fn=identity, **dataloader_kwargs
        )

        # move model to device
        try:
            self.network.to(self.device)
            self.network.eval()
        except AttributeError:
            pass  # not a PyTorch model object

        # generate samples in batches
        generated_samples = []
        for batch in dataloader:
            generated_samples.extend(batch)
        # get & log list of any sampls that failed to preprocess
        invalid_samples = dataloader.dataset.report(log=invalid_samples_log)

        if return_invalid_samples:
            return generated_samples, invalid_samples
        else:
            return generated_samples

    def eval(self, targets=None, scores=None, reset_metrics=True):
        """compute single-target or multi-target metrics from targets and scores

        Or, compute metrics on accumulated values in the TorchMetrics if targets is None

        By default, the overall model score is "map" (mean average precision)
        for multi-target models (self.single_target=False) and "f1" (average
        of f1 score across classes) for single-target models).

        update self.torch_metrics to include the desired metrics

        Args:
            targets: 0/1 for each sample and each class
            - if None, runs metric.compute() on each of self.torch_metrics
                (using accumulated values)
            scores: continuous values in 0/1 for each sample and class
            - if targets is None, this is ignored
            reset_metrics: if True, resets the metrics after computing them
                [default: True]

        Returns:
            dictionary of `metrics` (name: value)

        Raises:
            AssertionError: if targets are outside of range [0,1]
        """
        metrics = {}
        if targets is not None:
            # move tensors to device; avoid error float64 not supported on mps
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            scores = torch.tensor(scores, dtype=torch.float32).to(self.device)

            # check for invalid label values outside range of [0,1]
            assert (
                targets.max() <= 1 and targets.min() >= 0
            ), "Labels must in range [0,1], but found values outside range"

            # remove all samples with NaN for a prediction
            targets = targets[~torch.isnan(scores).any(dim=1), :]
            scores = scores[~torch.isnan(scores).any(dim=1), :]

            if len(scores) < 1:
                warnings.warn("Recieved empty list of predictions (or all nan)")
                return np.nan, np.nan

            # map is failing with memory limit on MPS, use CPU instead
            # TODO: reconsider casting labels to int (support soft labels)
            # if self.single_target, use argmax to get predicted class
            # because torchmetrics Multiclass metrics expect class indices
            y = targets.argmax(dim=1) if self.single_target else targets
            for name, metric in self.torch_metrics.items():
                device = (
                    self.device if self.device.type != "mps" else torch.device("cpu")
                )
                metrics[name] = metric.to(device)(
                    scores.detach().to(device), y.detach().long().to(device)
                ).cpu()

                if reset_metrics:
                    metric.reset()
        else:
            # compute each TorchMetrics overal value from accumulated values
            # since .reset() was last called
            # for instance, over all batches in an epoch
            for name, metric in self.torch_metrics.items():
                metrics[name] = metric.compute().cpu()
                if reset_metrics:
                    metric.reset()

        return metrics

    def run_validation(self, validation_df, progress_bar=True, **kwargs):
        """run validation on a validation set

        override this to customize the validation step
        eg, could run validation on multiple datasets and save performance of each
        in self.valid_metrics[current_epoch][validation_dataset_name]

        Args:
            validation_df: dataframe of validation samples
            progress_bar: if True, show a progress bar with tqdm
            **kwargs: passed to self.predict_dataloader()

        Returns:
            metrics: dictionary of evaluation metrics calculated with self.torch_metrics

        Effects:
            updates self.valid_metrics[current_epoch] with metrics for the current epoch
        """
        # run inference
        validation_scores = self.predict(
            validation_df,
            activation_layer=("softmax" if self.single_target else "sigmoid"),
            progress_bar=progress_bar,
            **kwargs,
        )

        # if validation_df index is file paths, we need to generate clip-df with labels
        # to evaluate the scores. Easiest to do this with self.predict_dataloader()
        dl = self.predict_dataloader(validation_df, **kwargs)
        val_labels = dl.dataset.dataset.label_df.values
        return self.eval(val_labels, validation_scores.values)

    def _train_epoch(self, train_loader, wandb_session=None, progress_bar=True):
        """perform forward pass, loss, and backpropagation for one epoch

        If wandb_session is passed, logs progress to wandb run

        Args:
            train_loader: DataLoader object to create samples
            wandb_session: a wandb session to log to
                - pass the value returned by wandb.init() to progress log to a
                Weights and Biases run
                - if None, does not log to wandb

        Returns:
            dictionary of evaluation metrics calculated with self.torch_metrics
        """
        self.network.train()
        batch_loss = []

        for batch_idx, (batch_tensors, batch_labels) in enumerate(
            tqdm(train_loader, disable=not progress_bar)
        ):
            # save loss for each batch; later take average for epoch
            loss = self.training_step((batch_tensors, batch_labels), batch_idx)
            batch_loss.append(loss.detach().cpu().numpy())
            ###########
            # Logging #
            ###########
            # log basic train info (used to print every batch)
            if batch_idx % self.log_interval == 0:
                # show some basic progress metrics during the epoch
                N = len(train_loader)
                self._log(
                    f"Epoch: {self.current_epoch} "
                    f"[batch {batch_idx}/{N}, {100 * batch_idx / N :.2f}%] "
                )

                # Log the Loss function
                epoch_loss_avg = np.mean(batch_loss)
                self._log(f"\tEpoch Running Average Loss: {epoch_loss_avg:.3f}")
                self._log(f"\tMost Recent Batch Loss: {batch_loss[-1]:.3f}")

        # update learning parameters each epoch
        self.scheduler.step()
        self.lr_scheduler_step += 1

        # compute and reset TorchMetrics
        self.train_metrics[self.current_epoch] = self.eval()

        # save the loss averaged over all batches
        self.loss_hist[self.current_epoch] = np.mean(batch_loss)

        if wandb_session is not None:
            wandb_session.log({"loss": np.mean(batch_loss)})
            wandb_session.log({"training": self.train_metrics[self.current_epoch]})

        # return a single overall score for the epoch
        return self.train_metrics[self.current_epoch]

    def _generate_wandb_config(self):
        # create a dictionary of parameters to save for this run
        wandb_config = dict(
            architecture=io.build_name(self.network),
            sample_duration=self.preprocessor.sample_duration,
            cuda_device_count=torch.cuda.device_count(),
            mps_available=torch.backends.mps.is_available(),
            classes=self.classes,
            single_target=self.single_target,
            opensoundscape_version=self.opensoundscape_version,
        )
        if "weight_decay" in self.optimizer_params:
            wandb_config["l2_regularization"] = self.optimizer_params["weight_decay"]
        else:
            wandb_config["l2_regularization"] = "n/a"

        if "lr" in self.optimizer_params:
            wandb_config["learning_rate"] = self.optimizer_params["lr"]
        else:
            wandb_config["learning_rate"] = "n/a"

        try:
            wandb_config["sample_shape"] = [
                self.preprocessor.height,
                self.preprocessor.width,
                self.preprocessor.channels,
            ]
        except:
            wandb_config["sample_shape"] = "n/a"

        return wandb_config

    def train(
        self,
        train_df,
        validation_df=None,
        epochs=1,
        batch_size=1,
        num_workers=0,
        save_path=".",
        save_interval=1,  # save weights every n epochs
        log_interval=10,  # print metrics every n batches
        validation_interval=1,  # compute validation metrics every n epochs
        reset_optimizer=False,
        restart_scheduler=False,
        invalid_samples_log="./invalid_training_samples.log",
        raise_errors=False,
        wandb_session=None,
        progress_bar=True,
        audio_root=None,
        **dataloader_kwargs,
    ):
        """train the model on samples from train_dataset

        If customized loss functions, networks, optimizers, or schedulers
        are desired, modify the respective attributes before calling .train().

        Args:
            train_df:
                a dataframe of files and labels for training the model
                - either has index `file` or multi-index (file,start_time,end_time)
            validation_df:
                a dataframe of files and labels for evaluating the model
                [default: None means no validation is performed]
            epochs:
                number of epochs to train for
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
                interval in batches to print training loss/metrics
            validation_interval:
                interval in epochs to test the model on the validation set
                Note that model will only update it's best score and save best.model
                file on epochs that it performs validation.
            reset_optimizer:
                if True, resets the optimizer rather than retaining state_dict
                of self.optimizer [default: False]
            restart_scheduler:
                if True, resets the learning rate scheduler rather than retaining
                state_dict of self.scheduler [default: False]
            invalid_samples_log:
                file path: log all samples that failed in preprocessing
                (file written when training completes)
                - if None,  does not write a file
            raise_errors:
                if True, raise errors when preprocessing fails
                if False, just log the errors to unsafe_samples_log
            wandb_session: a wandb session to log to
                - pass the value returned by wandb.init() to progress log to a
                Weights and Biases run
                - if None, does not log to wandb
                For example:
                ```
                import wandb
                wandb.login(key=api_key) #find your api_key at https://wandb.ai/settings
                session = wandb.init(enitity='mygroup',project='project1',name='first_run')
                ...
                model.train(...,wandb_session=session)
                session.finish()
                ```
            audio_root: optionally pass a root directory (pathlib.Path or str)
                - `audio_root` is prepended to each file path
                - if None (default), samples must contain full paths to files
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            **dataloader_kwargs: additional arguments passed to train_dataloader()
        Effects:
            If wandb_session is provided, logs progress and samples to Weights
            and Biases. A random set of training and validation samples
            are preprocessed and logged to a table. Training progress, loss,
            and metrics are also logged.
            Use self.wandb_logging dictionary to change the number of samples
            logged.
        """

        ### Input Validation ###
        # Validation of class list
        check_labels(train_df, self.classes)
        if validation_df is not None:
            check_labels(validation_df, self.classes)

        # Validation: warn user if no validation set
        if validation_df is None:
            warnings.warn(
                "No validation set was provided. Model will be "
                "evaluated using the performance on the training set."
            )

        if audio_root is not None:  # add this to dataloader keyword arguments
            dataloader_kwargs.update(dict(audio_root=audio_root))

        ## Initialization ##

        # Initialize attributes
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_path = save_path

        # Initialize Weights and Biases (wandb) logging ###
        if wandb_session is not None:
            # update the run config with information about the model
            wandb_session.config.update(self._generate_wandb_config())

            # update the run config with training parameters
            wandb_session.config.update(
                dict(
                    epochs=epochs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    lr_sheculer_params=self.lr_scheduler_params,
                    optimizer_params=self.optimizer_params,
                    model_save_path=Path(save_path).resolve(),
                )
            )

            # use wandb.watch to log histograms of parameter and gradient values
            # value of None for log_freq means do not use wandb.watch()
            log_freq = self.wandb_logging["watch_freq"]
            if log_freq is not None:
                wandb_session.watch(
                    self.network,
                    log="all",
                    log_freq=log_freq,
                    log_graph=(self.wandb_logging["log_graph"]),
                )

            # log tables of preprocessed samples
            wandb_session.log(
                {
                    "training_samples": wandb_table(
                        AudioFileDataset(
                            train_df, self.preprocessor, bypass_augmentations=False
                        ),
                        self.wandb_logging["n_preview_samples"],
                    ),
                    "training_samples_no_aug": wandb_table(
                        AudioFileDataset(
                            train_df, self.preprocessor, bypass_augmentations=True
                        ),
                        self.wandb_logging["n_preview_samples"],
                    ),
                    "validation_samples": wandb_table(
                        AudioFileDataset(
                            validation_df,
                            self.preprocessor,
                            bypass_augmentations=True,
                        ),
                        self.wandb_logging["n_preview_samples"],
                    ),
                }
            )

        # Move network to device
        self.network.to(self.device)

        ### Set Up DataLoader, Loss and Optimization ###
        train_loader = self.train_dataloader(
            train_df,
            batch_size=batch_size,
            num_workers=num_workers,
            raise_errors=raise_errors,
            **dataloader_kwargs,
        )

        ######################
        # Optimization setup #
        ######################
        # TODO: check if resuming training is working properly for optimizer and scheduler

        # Set up optimizer parameters for each network component
        # Note: we re-create bc the user may have changed self.optimizer_params, or
        # if we re-created the objects, the IDs of the params list have changed.
        # if self.optimizer/self.scheduler are not None, re-loads their state_dicts
        optimizer_and_scheduler = self.configure_optimizers(
            reset_optimizer=reset_optimizer, restart_scheduler=restart_scheduler
        )
        self.scheduler = optimizer_and_scheduler["scheduler"]
        self.optimizer = optimizer_and_scheduler["optimizer"]

        # Note: loss function (self.loss_fn) was initialized at __init__
        # can override like model.loss_fn = SomeLossCls()

        self.best_score = 0.0
        self.best_epoch = 0

        ### Train ###

        for epoch in range(epochs):
            # 1 epoch = 1 view of each training sample
            # loss fn, backpropogation, and optimizer step generally occur after each batch
            # validation generally occurs after validation_interval epochs

            ### Training ###
            self._log(f"\nTraining Epoch {self.current_epoch}")
            train_metrics = self._train_epoch(
                train_loader, wandb_session, progress_bar=progress_bar
            )

            #### Validation ###
            if epoch % validation_interval == 0:
                if validation_df is not None:
                    self._log("\nValidation.")
                    val_metrics = self.run_validation(
                        validation_df,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        raise_errors=raise_errors,
                        **dataloader_kwargs,
                    )
                    self.valid_metrics[self.current_epoch] = val_metrics
                    score = val_metrics[self.score_metric]  # overall score
                    if wandb_session is not None:
                        wandb_session.log({"validation": val_metrics})
                else:
                    # Get overall score from training metrics if no validation_df given
                    score = train_metrics[self.score_metric]

                # if this is the best score, update & save weights to best.model
                # we save both the pickled and unpickled formats here
                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = self.current_epoch
                    save_path = f"{self.save_path}/best.model"
                    pickle_path = f"{self.save_path}/best.pickle"
                    self._log(f"New best model saved to {save_path}", level=2)
                    self.save(save_path, pickle=False)
                    self.save(pickle_path, pickle=True)

            # save pickled model every n epochs
            # pickled model file allows us to resume training
            if (
                self.current_epoch + 1
            ) % self.save_interval == 0 or epoch == epochs - 1:
                save_path = f"{self.save_path}/epoch-{self.current_epoch}.model"
                self._log(f"Saving model to {save_path}", level=2)
                try:
                    self.save(save_path, pickle=True)
                except Exception as e:
                    self._log(
                        "Saving pickled model failed. This may be beacuse the model is not picklable "
                        "e.g. if it contains a lambda function, generator, or other non-picklable object."
                    )

            if wandb_session is not None:
                wandb_session.log({"epoch": epoch})
            self.current_epoch += 1

        ### Logging ###
        self._log("Training complete", level=2)
        self._log(
            f"\nBest Model Appears at Epoch {self.best_epoch} "
            f"with Validation score {self.best_score:.3f}."
        )

        # warn the user if there were invalid samples (samples that failed to preprocess)
        invalid_samples = train_loader.dataset.report(log=invalid_samples_log)
        self._log(
            f"{len(invalid_samples)} of {len(train_df)} total training "
            f"samples failed to preprocess",
            level=2,
        )
        self._log(f"List of invalid samples: {invalid_samples}", level=3)

    def save(self, path, save_hooks=False, pickle=False):
        """save model with weights using torch.save()

        load from saved file with cnn.load_model(path)


        Args:
            path: file path for saved model object
            save_hooks: retain forward and backward hooks on modules
                [default: False] Note: True can cause issues when using
                wandb.watch()
            pickle: if True, saves the entire model object using torch.save()
                Note: if using pickle=True, entire object is pickled, which means that
                saving and loading model objects across OpenSoundscape versions
                might not work properly. pickle=True is useful for resuming training,
                because it retains the state of the optimizer, scheduler, loss function, etc
                pickle=False is recommended for saving models for inference/deployment/sharing
                [default: False]
        """
        os.makedirs(Path(path).parent, exist_ok=True)

        model_copy = copy.deepcopy(self)

        if not save_hooks:
            # remove all forward and backward hooks on network.modules()
            from collections import OrderedDict

            for m in model_copy.network.modules():
                m._forward_hooks = OrderedDict()
                m._backward_hooks = OrderedDict()

        if pickle:
            # save a pickled model object; may not work across opso versions
            torch.save(model_copy, path)
        else:
            # save dictionary of separate components
            # better for cross-version compatability
            # dictionary can be loaded with torch.load() to inspect individual components
            try:
                arch_name = self.network.constructor_name
            except AttributeError:
                arch_name = "unknown"
                warnings.warn(
                    "Could not determine architecture constructor name for saved model"
                )
            torch.save(
                {
                    "weights": self.network.state_dict(),
                    "class": io.build_name(self),
                    "classes": self.classes,
                    "sample_duration": self.preprocessor.sample_duration,
                    "architecture": arch_name,
                    "preprocessor_dict": self.preprocessor.to_dict(),
                    "opensoundscape_version": opensoundscape.__version__,
                    # doesn't support resuming training across with optimizer/scheduler states
                    # because there are many other parameters that would need to be saved
                    # instead, use pickle=True for resuming training
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    # "scheduler_state_dict": self.scheduler.state_dict(),
                },
                path,
            )

    @classmethod
    def load(cls, path, unpickle=True):
        """load a model saved using CNN.save()

        Args:
            path: path to file saved using CNN.save()
            unpickle: if True, passes `weights_only=False` to
                torch.load(). This is necessary if the model was saved with
                pickle=True, which saves the entire model object. If
                `unpickle=False`, this function will work if the model was saved
                with pickle=False, but will raise an error if the model was saved
                with pickle=True. [default: True]

        Returns:
            new CNN instance

        Note: Note that if you used pickle=True when saving, the model object might not load properly
        across different versions of OpenSoundscape.
        """
        model_dict = torch.load(path, weights_only=not unpickle)

        opso_version = (
            model_dict.pop("opensoundscape_version")
            if isinstance(model_dict, dict)
            else model_dict.opensoundscape_version
        )
        if opso_version != opensoundscape.__version__:
            warnings.warn(
                f"Model was saved with OpenSoundscape version {opso_version}, "
                f"but you are currently using version {opensoundscape.__version__}. "
                "This might not be an issue but you should confirm that the model behaves as expected."
            )

        if isinstance(model_dict, dict):
            # load up the weights and instantiate from dictionary keys
            # includes preprocessing parameters and settings
            state_dict = model_dict.pop("weights")
            class_name = model_dict.pop("class")
            model = cls(**model_dict)
            model.network.load_state_dict(state_dict)
        else:
            model = model_dict  # entire pickled object, not dictionary
            opso_version = model.opensoundscape_version

        return model

    def save_weights(self, path):
        """save just the weights of the network

        This allows the saved weights to be used more flexibly than model.save()
        which will pickle the entire object. The weights are saved in a pickled
        dictionary using torch.save(self.network.state_dict())

        Args:
            path: location to save weights file
        """
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path, strict=True):
        """load network weights state dict from a file

        For instance, load weights saved with .save_weights()
        in-place operation

        Args:
            path: file path with saved weights
            strict: (bool) see torch.load()
        """
        self.network.load_state_dict(torch.load(path), strict=strict)

    def __call__(
        self,
        dataloader,
        wandb_session=None,
        progress_bar=True,
        intermediate_layers=None,
        avgpool_intermediates=True,
    ):
        """Run inference on a dataloader, generating scores for each sample

        Optionally also return outputs from intermediate layers

        Args:
            dataloader: DataLoader object to create samples, e.g. from .predict_dataloader()
            wandb_session: a wandb session to log progress to (e.g. return value of wandb.init())
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            intermediate_layers: list of layers to return outputs from
                [default: None] if None, only returns final layer outputs
                if a list of layers is provided, returns a second value
                with outputs from each layer. Example: [self.model.layer1]
            avgpool_intermediates: bool, if True, applies global average pooling to intermediate outputs
                i.e. averages across all dimensions except first to get a 1D vector per sample
                [default: True] (note that False may results in large memory usage)

        Returns:
            if intermediate outputs is None, returns
            `scores`: np.array of scores for each sample

            if intermediate_outputs is not None, returns a tuple:
            `(scores, intermediate_outputs)` where intermediate_outputs is
            a list of tensors, the outputs from each layer in intermediate_layers
        """

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            warnings.warn(
                "dataloader is not an instance of torch.utils.data.DataLoader!"
            )

        # move network to device
        self.network.to(self.device)
        self.network.eval()

        # initialize scores
        pred_scores = []

        # init a variable to save outputs of each batch for each target layer
        intermediate_layers = intermediate_layers or []
        intermediate_outputs = [[] for _ in intermediate_layers]

        # define a function that will be used to save the output of each target layer
        # during inference
        def forward_hook_to_save_output(layer_name, idx):
            def hook(module, input, output):
                if avgpool_intermediates:
                    # apply global average pooling to intermediate outputs
                    # (average across all dimensions except first to get a 1D vector per sample)
                    # (also skip batch dimension: so start averaging from dim 2)
                    if output.dim() > 2:
                        output = output.mean(list(range(2, output.dim())))
                intermediate_outputs[idx].append(output)

            return hook

        # initialize forward hooks to save intermediate outputs
        fhooks = []  # keep the handles so we can remove the hooks later
        for idx, l in enumerate(intermediate_layers):
            fhooks.append(l.register_forward_hook(forward_hook_to_save_output(l, idx)))

        # disable gradient updates during inference
        with torch.set_grad_enabled(False):
            for i, (batch_tensors, _) in enumerate(
                tqdm(dataloader, disable=not progress_bar)
            ):
                batch_tensors = batch_tensors.to(self.device)
                batch_tensors.requires_grad = False

                # forward pass of network: feature extractor + classifier
                logits = self.network(batch_tensors)

                # disable gradients on returned values
                pred_scores.extend(list(logits.detach().cpu().numpy()))

                if wandb_session is not None:
                    wandb_session.log(
                        {
                            "progress": i / len(dataloader),
                            "completed_batches": i,
                            "total_batches": len(dataloader),
                        }
                    )

        # clean up by removing forward hooks
        for fh in fhooks:
            fh.remove()

        # aggregate across all batches
        if len(pred_scores) > 0:
            pred_scores = np.array(pred_scores)

            # aggregate across batches
            # note that shapes of elements in intermediate_outputs may vary
            # (so we don't make one combined np.array)
            # careful with squeezing: if we have a batch size of 1, we don't want to squeeze out the batch dimension
            intermediate_outputs = [
                torch.vstack(x).detach().cpu().numpy() for x in intermediate_outputs
            ]

            # replace scores with nan for samples that failed in preprocessing
            # (we predicted on substitute-samples rather than
            # skipping the samples that failed preprocessing)
            pred_scores[dataloader.dataset._invalid_indices, :] = np.nan
            for i in range(len(intermediate_outputs)):
                intermediate_outputs[i][dataloader.dataset._invalid_indices, :] = np.nan
        else:
            pred_scores = None

        if len(intermediate_layers) > 0:
            return pred_scores, intermediate_outputs
        return pred_scores

    def generate_cams(
        self,
        samples,
        method="gradcam",
        classes=None,
        target_layers=None,
        guided_backprop=False,
        progress_bar=True,
        **kwargs,
    ):
        """
        Generate a activation and/or backprop heatmaps for each sample

        Args:
            samples: (same as CNN.predict())
                the files to generate predictions for. Can be:
                - a dataframe with index containing audio paths, OR
                - a dataframe with multi-index (file, start_time, end_time), OR
                - a list (or np.ndarray) of audio file paths
            method: method to use for activation map. Can be str (choose from below)
                or a class of pytorch_grad_cam (any subclass of BaseCAM), or None
                if None, activation maps will not be created [default:'gradcam']

                str can be any of the following:
                    "gradcam": pytorch_grad_cam.GradCAM,
                    "hirescam": pytorch_grad_cam.HiResCAM,
                    "scorecam": pytorch_grad_cam.ScoreCAM,
                    "gradcam++": pytorch_grad_cam.GradCAMPlusPlus,
                    "ablationcam": pytorch_grad_cam.AblationCAM,
                    "xgradcam": pytorch_grad_cam.XGradCAM,
                    "eigencam": pytorch_grad_cam.EigenCAM,
                    "eigengradcam": pytorch_grad_cam.EigenGradCAM,
                    "layercam": pytorch_grad_cam.LayerCAM,
                    "fullgrad": pytorch_grad_cam.FullGrad,
                    "gradcamelementwise": pytorch_grad_cam.GradCAMElementWise,

            classes (list): list of classes, will create maps for each class
                [default: None] if None, creates an activation map for the highest
                scoring class on a sample-by-sample basis
            target_layers (list): list of target layers for GradCAM
                - if None [default] attempts to use architecture's default target_layer
                Note: only architectures created with opensoundscape 0.9.0+ will
                have a default target layer. See pytorch_grad_cam docs for suggestions.
                Note: if multiple layers are provided, the activations are merged across
                    layers (rather than returning separate activations per layer)
            guided_backprop: bool [default: False] if True, performs guided backpropagation
                for each class in classes. AudioSamples will have attribute .gbp_maps,
                a pd.Series indexed by class name
            **kwargs are passed to SafeAudioDataloader
                (incl: batch_size, num_workers, split_file_into_clips, bypass_augmentations,
                raise_errors, overlap_fraction, final_clip, other DataLoader args)

        Returns:
            a list of AudioSample objects with .cam attribute, an instance of the CAM class (
            visualize with `sample.cam.plot()`). See the CAM class for more details

        See pytorch_grad_cam documentation for references to the source of each method.
        """

        ## INPUT VALIDATION ##

        if classes is not None:  # check that classes are in model.classes
            assert np.all(
                [c in self.classes for c in classes]
            ), "`classes` must be in self.classes"

        # if target_layer is None, attempt to retrieve default target layers of network
        if target_layers is None:
            try:
                # get default layer to use for outputs to CAMs
                target_layers = [self.network.get_submodule(self.network.cam_layer)]
            except AttributeError as exc:
                raise AttributeError(
                    "Please specify target_layers. Models trained with older versions of Opensoundscape "
                    "and user-specified models do not have default target layers for the cam. "
                    "For example, for a ResNET model, try target_layers=[model.network.layer4]"
                ) from exc
        else:  # check that target_layers are modules of self.network
            for tl in target_layers:
                assert (
                    tl in self.network.modules()
                ), f"target_layers must be in self.network.modules(), but {tl} is not."

        ## INITIALIZE CAMS AND DATALOADER ##
        # move model to device
        self.network.to(self.device)
        self.network.eval()

        # initialize cam object: `method` is either str in methods_dict keys, or the class
        methods_dict = {
            "gradcam": pytorch_grad_cam.GradCAM,
            "hirescam": pytorch_grad_cam.HiResCAM,
            "scorecam": pytorch_grad_cam.ScoreCAM,
            "gradcam++": pytorch_grad_cam.GradCAMPlusPlus,
            "ablationcam": pytorch_grad_cam.AblationCAM,
            "xgradcam": pytorch_grad_cam.XGradCAM,
            "eigencam": pytorch_grad_cam.EigenCAM,
            "eigengradcam": pytorch_grad_cam.EigenGradCAM,
            "layercam": pytorch_grad_cam.LayerCAM,
            "fullgrad": pytorch_grad_cam.FullGrad,
            "gradcamelementwise": pytorch_grad_cam.GradCAMElementWise,
        }
        if isinstance(method, str) and method in methods_dict:
            # get cam clsas based on string name and create instance
            cam = methods_dict[method](model=self.network, target_layers=target_layers)
            cam.device = self.device
        elif method is None:
            cam = None
        elif issubclass(method, pytorch_grad_cam.base_cam.BaseCAM):
            # generate instance of cam from class
            cam = method(model=self.network, target_layers=target_layers)
            cam.device = self.device
        else:
            raise ValueError(
                f"`method` {method} not supported. "
                f"Must be str from list of supported methods or a subclass of "
                f"pytorch_grad_cam.base_cam.BaseCAM. See docstring for details. "
            )

        # initialize guided back propagation object
        if guided_backprop:
            gb_model = pytorch_grad_cam.GuidedBackpropReLUModel(
                model=self.network, device=self.device
            )

        # create dataloader, collate using `identity` to return list of AudioSample
        # rather than (samples, labels) tensors
        dataloader = self.inference_dataloader_cls(
            samples, self.preprocessor, shuffle=False, collate_fn=identity, **kwargs
        )

        ## GENERATE SAMPLES ##

        generated_samples = []
        for i, samples in enumerate(
            tqdm(
                dataloader,
                disable=not progress_bar,
            )
        ):
            # load a batch of images and labels from the dataloader
            # we collate here rather than in the DataLoader so that
            # we can still access the AudioSamples and thier information
            batch_tensors, batch_labels = collate_audio_samples(samples)
            batch_tensors = batch_tensors.to(self.device)
            batch_tensors.requires_grad = False

            # generate logits with forward pass
            logits = self.network(batch_tensors)

            # generate class activation maps using cam object
            def target(class_name):
                """helper for pytorch_grad_cam class syntax"""
                # first get integet position of class name in self.classes
                class_idx = list(self.classes).index(class_name)
                # then create list of class required by pytorch_grad_cam
                return [ClassifierOutputTarget(class_idx)]

            if cam is not None:
                if classes is None:  # selects highest scoring class per sample
                    batch_maps = pd.Series({None: cam(batch_tensors)})
                else:  # one activation map per class
                    batch_maps = pd.Series(
                        {c: cam(batch_tensors, targets=target(c)) for c in classes}
                    )

            # update the AudioSample objects to include the activation maps
            # and create guided backprop maps, one sample at a time
            for i, sample in enumerate(samples):
                # add the scores (logits) from the network to AudioSample as dictionary
                sample.scores = dict(
                    zip(self.classes, logits[i].detach().cpu().numpy())
                )

                # add the cams as a dictionary keyed by class
                if cam is None:
                    activation_maps = None
                else:
                    # extract this sample's activation maps from batch (all classes)
                    activation_maps = pd.Series(
                        {c: batch_maps[c][i] for c in batch_maps.index}
                    )

                # if requested, calculate the ReLU backpropogation, which creates
                # high resolution pixel-activation levels for specific classes
                # GuidedBackpropReLUasModule does not support batching
                if guided_backprop:
                    # create "batch" with one sample
                    t = batch_tensors[i].unsqueeze(0)
                    # target_category expects the index position of the class eg 0 for
                    # first class, rather than the class name
                    # note: t.detach() to avoid bug,
                    # see https://github.com/jacobgil/pytorch-grad-cam/issues/401
                    if classes is None:  # defaults to highest scoring class
                        gbp_maps = pd.Series({None: gb_model(t.detach())})
                    else:  # one for each class
                        cls_list = list(self.classes)
                        gbp_maps = pd.Series(
                            {
                                c: gb_model(
                                    t.detach(), target_category=cls_list.index(c)
                                )
                                for c in classes
                            }
                        )

                    # average the guided backprop map over the channel dimension
                    def avg_over_channels(img):
                        return img.mean(axis=-1)

                    gbp_maps = gbp_maps.apply(avg_over_channels)

                else:  # no guided backprop requested
                    gbp_maps = None

                # add CAM object as sample.cam (includes activation_map and gbp_maps)
                sample.cam = CAM(
                    base_image=batch_tensors[i],
                    activation_maps=activation_maps,
                    gbp_maps=gbp_maps,
                )

                # add sample to list of outputs to return
                generated_samples.append(sample)

        # return list of AudioSamples containing .cam attributes
        return generated_samples

    def embed(
        self,
        samples,
        target_layer=None,
        progress_bar=True,
        return_preds=False,
        avgpool=True,
        return_dfs=True,
        audio_root=None,
        **dataloader_kwargs,
    ):
        """
        Generate embeddings (intermediate layer outputs) for audio files/clips

        Note: to capture embeddings on multiple layers, use self.__call__ with intermediate_layers
        argument directly. This wrapper only allows one target_layer.

        Note: Output can be n-dimensional array (return_dfs=False) or pd.DataFrame with multi-index
        like .predict() (return_dfs=True). If avgpool=False, return_dfs is forced to False since we
        can't create a DataFrame with >2 dimensions.

        Args:
            samples: same as CNN.predict(): list of file paths, OR pd.DataFrame with index
                containing audio file paths, OR a pd.DataFrame with multi-index (file, start_time,
                end_time)
            target_layers: layers from self.model._modules to
                extract outputs from - if None, attempts to use self.model.embedding_layer as
                default
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_preds: bool, if True, returns two outputs (embeddings, logits)
            avgpool: bool, if True, applies global average pooling to intermediate outputs
                i.e. averages across all dimensions except first to get a 1D vector per sample
            return_dfs: bool, if True, returns embeddings as pd.DataFrame with multi-index like
                .predict(). if False, returns np.array of embeddings [default: True]. If
                avg_pool=False, overrides to return np.array since we can't have a df with >2
                dimensions
            audio_root: optionally pass a root directory (pathlib.Path or str)
                - `audio_root` is prepended to each file path
                - if None (default), samples must contain full paths to files
            dataloader_kwargs are passed to self.predict_dataloader()

        Returns: (embeddings, preds) if return_preds=True or embeddings if return_preds=False
            types are pd.DataFrame if return_dfs=True, or np.array if return_dfs=False

        """
        if audio_root is not None:
            dataloader_kwargs.update(dict(audio_root=audio_root))
        if not avgpool:  # cannot create a DataFrame with >2 dimensions
            return_dfs = False

        # if target_layer is None, attempt to retrieve default target layers of network
        if target_layer is None:
            try:
                target_layer = self.network.get_submodule(self.network.embedding_layer)
            except (AttributeError, KeyError) as exc:
                raise AttributeError(
                    "Please specify target_layer. Models trained with older versions of Opensoundscape, "
                    "or custom architectures, do not have default `.network.embedding_layer`. "
                    "e.g. For a ResNET model, try target_layers=[self.model.layer4]"
                ) from exc
        else:  # check that target_layers are modules of self.model
            assert (
                target_layer in self.network.modules()
            ), f"target_layers must be in self.model.modules(), but {target_layer} is not."

        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples, **dataloader_kwargs)

        # run inference, returns (scores, intermediate_outputs)
        preds, embeddings = self(
            dataloader=dataloader,
            progress_bar=progress_bar,
            intermediate_layers=[target_layer],
            avgpool_intermediates=avgpool,
        )

        if return_dfs:
            # put embeddings in DataFrame with multi-index like .predict()
            embeddings = pd.DataFrame(
                data=embeddings[0], index=dataloader.dataset.dataset.label_df.index
            )
        else:
            embeddings = embeddings[0]

        if return_preds:
            if return_dfs:
                # put predictions in a DataFrame with same index as embeddings
                preds = pd.DataFrame(
                    data=preds, index=dataloader.dataset.dataset.label_df.index
                )
            return embeddings, preds
        return embeddings

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        """
        Set the device to use in train/predict, casting strings to torch.device datatype

        Automatically gpu (default is 'cuda:0' or 'mps') if available. Can set after init, eg
        model.device='cuda:1'. Network and samples are moved to device during training/inference.
        Devices could be 'cuda:0', torch.device('cuda'), torch.device('cpu'), torch.device('mps')
        etc

        Args:
            device: a torch.device object or str such as 'cuda:0', 'mps', 'cpu'
        """
        self._device = torch.device(device)


@register_model_cls
class CNN(SpectrogramClassifier):
    """alias for SpectrogramClassifier

    improves comaptibility with older code / previous opso versions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaseClassifier(SpectrogramClassifier):
    """alias for SpectrogramClassifier

    improves compatibility with older code / previous opso versions,
    which had a BaseClassifier class as a parent to the CNN class
    """


def use_resample_loss(model, train_df):
    """Modify a model to use ResampleLoss for multi-target training

    ResampleLoss may perform better than BCE Loss for multitarget problems
    in some scenarios.

    Args:
        model: CNN object
        train_df: dataframe of labels, used to calculate class frequency
    """
    class_frequency = torch.tensor(train_df.values).sum(0).to(model.device)
    model.loss_fn = ResampleLoss(class_frequency)


@register_model_cls
class InceptionV3(SpectrogramClassifier):
    """Child of SpectrogramClassifier class for InceptionV3 architecture"""

    def __init__(
        self,
        classes,
        sample_duration,
        single_target=False,
        freeze_feature_extractor=False,
        weights="DEFAULT",
        sample_width=299,
        sample_height=299,
        **kwargs,
    ):
        """Model object for InceptionV3 architecture subclassing CNN

        See opensoundscape.org for exaple use.

        Args:
            classes:
                list of output classes (usually strings)
            sample_duration: duration in seconds of one audio sample
            single_target: if True, predict exactly one class per sample
                [default:False]
            freeze-feature_extractor:
                if True, feature weights don't have
                gradient, and only final classification layer is trained
            weights:
                string containing version name of the pre-trained classification weights to use for
                this architecture. if 'DEFAULT', model is loaded with best available weights (note
                that these may change across versions). Pre-trained weights available for each
                architecture are listed at https://pytorch.org/vision/stable/models.html
            sample_height: height of input image in pixels
            sample_width: width of input image in pixels
            **kwargs passed to SpectrogramClassifier.__init__()

        Note: InceptionV3 architecture implementation assumes channels=3
        """

        self.classes = classes

        architecture = inception_v3(
            len(self.classes),
            freeze_feature_extractor=freeze_feature_extractor,
            weights=weights,
        )
        architecture.constructor_name = "inception_v3"

        if "architecture" in kwargs:
            kwargs.pop("architecture")

        super().__init__(
            architecture=architecture,
            classes=classes,
            sample_duration=sample_duration,
            single_target=single_target,
            height=sample_height,
            width=sample_width,
            channels=3,
            **kwargs,
        )
        self.name = "InceptionV3"

    def training_step(self, samples, batch_idx):
        """Training step for pytorch lightning

        Args:
            batch: a batch of data from the DataLoader
            batch_idx: index of the batch

        Returns:
            loss: loss value for the batch
        """
        batch_tensors, batch_labels = samples
        batch_tensors = batch_tensors.to(self.device)
        batch_labels = batch_labels.to(self.device)

        batch_size = len(batch_tensors)

        # automatic mixed precision
        # can get rid of if/else blocks and use enabled=true
        # once mps is supported https://github.com/pytorch/pytorch/pull/99272
        # but right now, raises error if enabled=True and device is mps

        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # with torch.autocast(
        #     device_type=self.device, dtype=torch.float16, enabled=self.use_amp
        # ):
        #     output = self.network(input)
        #     loss = self.loss_fn(output, batch_labels)

        # if not self.lightning_mode:
        #     # if not using Lightning, we manually call
        #     # loss.backward() and optimizer.step()
        #     # Lightning does this behind the scenes
        #     self.scaler.scale(loss).backward()
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        #     self.optimizer.zero_grad()  # set_to_none=True here can modestly improve performance

        # if self.use_amp is False, GradScaler with enabled=False should have no effect
        if "mps" in str(self.device):
            use_amp = False  # Not using amp: not implemented for mps as of 2024-07-11
        else:
            use_amp = self.use_amp

        if use_amp:  # as of 7/11/24, torch.autocast is not supported for mps
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if "cuda" in str(self.device):
                device_type = "cuda"
                dtype = torch.float16
            else:
                device_type = "cpu"
                dtype = torch.bfloat16
            with torch.autocast(
                device_type=device_type, dtype=dtype
            ):  # , enabled=self.use_amp
                # ):
                inception_outs = self.network(batch_tensors)
                logits = inception_outs.logits
                aux_logits = inception_outs.aux_logits

                loss1 = self.loss_fn(logits, batch_labels)
                loss2 = self.loss_fn(aux_logits, batch_labels)
                loss = loss1 + 0.4 * loss2
            if not self.lightning_mode:
                # if not using Lightning, we manually call
                # loss.backward() and optimizer.step()
                # Lightning does this behind the scenes
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()  # set_to_none=True here can modestly improve performance
        else:
            output = self.network(batch_tensors)

            # calculate loss
            inception_outs = self.network(batch_tensors)
            logits = inception_outs.logits
            aux_logits = inception_outs.aux_logits

            loss1 = self.loss_fn(logits, batch_labels)
            loss2 = self.loss_fn(aux_logits, batch_labels)
            loss = loss1 + 0.4 * loss2

            if not self.lightning_mode:
                # if not using Lightning, we manually call
                # loss.backward() and optimizer.step()
                # Lightning does this behind the scenes
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        # compute and log any metrics in self.torch_metrics
        batch_metrics = {
            f"train_{name}": metric.to(self.device)(
                logits.detach(), batch_labels.detach().int()
            ).cpu()
            for name, metric in self.torch_metrics.items()
        }

        if self.lightning_mode:
            self.log(
                f"train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch_tensors),
            )
            self.log_dict(
                batch_metrics, on_epoch=True, on_step=False, batch_size=batch_size
            )
        return loss

    @classmethod
    def from_torch_dict(self):
        raise NotImplementedError(
            "Creating InceptionV3 from torch dict is not implemented."
        )


def load_model(path, device=None, unpickle=True):
    """load a saved model object

    This function handles models saved either as pickled objects or as a dictionary
    including weights, preprocessing parameters, architecture name, etc.

    Note that pickled objects may not load properly across different versions of
    OpenSoundscape, while the dictionary format does not retain the full training state
    for resuming model training.

    Args:
        path: file path of saved model
        device: which device to load into, eg 'cuda:1'
            [default: None] will choose first gpu if available, otherwise cpu
        unpickle: if True, passes `weights_only=False` to torch.load(). This is necessary if the
        model was saved with`pickle=True`, which saves the entire model object.
            If `unpickle=False`, this function will work if the model was saved with pickle=False,
            but will raise an error if the model was saved with pickle=True. [default: True]
    Returns:
        a model object with loaded weights
    """
    try:
        # load the entire pickled model object from a file and
        # move the model to the desired torch "device" (eg cpu or cuda for gpu)
        # by default, will choose cuda:0 if cuda is available,
        # otherwise mps (Apple Silicon) if available, otherwise cpu
        if device is None:
            device = _gpu_if_available()
        loaded_content = torch.load(
            path,
            map_location=device,
            weights_only=not unpickle,
        )

        if isinstance(loaded_content, dict):
            model_cls = MODEL_CLS_DICT[loaded_content.pop("class")]
            model = model_cls(**loaded_content)
            model.network.load_state_dict(loaded_content["weights"])
        else:
            model = loaded_content

        # warn the user if loaded model's opso version doesn't match the current one
        if model.opensoundscape_version != opensoundscape.__version__:
            warnings.warn(
                f"This model was saved with an earlier version of opensoundscape "
                f"({model.opensoundscape_version}) and will not work properly in "
                f"the current opensoundscape version ({opensoundscape.__version__}). "
                f"To use models across package versions use .save_torch_dict and "
                f".load_torch_dict"
            )

        model.device = device
        return model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            """
            This model file could not be loaded in this version of
            OpenSoundscape. You may need to load the model with the version
            of OpenSoundscape that created it and torch.save() the
            model.network.state_dict(), then load the weights with model.load_weights
            in the current OpenSoundscape version (where model is a new instance of 
            this class). If you do this, make sure to
            re-create any specific preprocessing steps that were used in the
            original model. See the `Predict with pre-trained CNN` tutorial for details.
            """
        ) from e


def _gpu_if_available():
    """
    Return a torch.device, chosing cuda:0 or mps if available

    Returns the first available GPU device (torch.device('cuda:0')) if cuda is available,
    otherwise returns torch.device('mps') if MPS is available,
    otherwise returns the CPU device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
