import warnings
import torch
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MulticlassAveragePrecision,
    MulticlassAUROC,
)

import opensoundscape
from opensoundscape.ml import cnn_architectures
from opensoundscape.ml.loss import (
    BCEWithLogitsLoss_hot,
    CrossEntropyLoss_hot,
    ResampleLoss,
)
from opensoundscape.preprocess.preprocessors import (
    BasePreprocessor,
    SpectrogramPreprocessor,
)
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from opensoundscape.sample import collate_audio_samples

from lightning.pytorch.callbacks import ModelCheckpoint

import warnings

import lightning as L


class OpenSoundscapeLightningModule(L.LightningModule):
    def __init__(self):
        super(OpenSoundscapeLightningModule, self).__init__()

        self.name = "OPSOLightningModule"
        self.opensoundscape_version = opensoundscape.__version__

        # TODO: set up logging of hyperparameters to arbitary logger

        # model characteristics
        self.scheduler = None
        """torch.optim.lr_scheduler object for learning rate scheduling"""

        self.torch_metrics = {}
        """add torchmetrics "name":object pairs to compute metrics during training"""

        self.preprocessor = BasePreprocessor()
        """an instance of BasePreprocessor or subclass that preprocesses audio samples

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
        """a pytorch Module such as Resnet18 or a custom object

        for convenience, __init__ also allows user to provide string matching
        a key from opensoundscape.ml.cnn_architectures.ARCH_DICT.
        
        List options: `opensoundscape.ml.cnn_architectures.list_architectures()`
        """

        ### loss function ###
        self.loss_fn = BCEWithLogitsLoss_hot()
        """specify a loss function to use for training, eg BCEWithLogitsLoss_hot
        
        by initializing a callable object or passing a function
        """

        self.optimizer_params = {
            "class": torch.optim.SGD,
            "kwargs": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
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
        self.scheduler_params = {
            "class": torch.optim.lr_scheduler.CosineAnnealingLR,
            "kwargs":{
                "T_max": n_epochs,
                "eta_min": 1e-7,
                "last_epoch":self.current_epoch-1
        }
        ```
        """

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
        logits = self.network(batch_tensors)
        loss = self.loss_fn(logits, batch_labels)
        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=len(batch_tensors),
        )

        # compute and log any metrics in self.torch_metrics
        # TODO: consider using validation set names rather than integer index
        # (would have to store a set of names for the validation set)
        batch_metrics = {
            f"train_{name}": metric.to(self.device)(
                logits.detach(), batch_labels.detach().int()
            ).cpu()
            for name, metric in self.torch_metrics.items()
        }
        self.log_dict(
            batch_metrics, on_epoch=True, on_step=False, batch_size=batch_size
        )
        # when on_epoch=True, compute() is called to reset the metric at epoch end

        return loss

    def forward(self, samples):
        """standard Lightning method defining action to take on each batch for inference

        typically returns logits (raw, untransformed model outputs)
        """
        batch_tensors, _ = samples
        batch_tensors = batch_tensors.to(self.device)
        return self.network(batch_tensors)

    # def predict_step(self, batch): #runs forward() if we don't override default

    def configure_optimizers(self):
        """standard Lightning method to initialize an optimizer and learning rate scheduler

        Lightning uses this function at the start of training

        Here, we initialize the optimizer and lr_scheduler using the parameters
        self.optimizer_params and self.scheduler_params, which are dictionaries with a key
        "class" and a key "kwargs" (containing a dictionary of keyword arguments to initialize
        the class with). We initialize the class with the kwargs and the appropriate
        first argument: optimizer=opt_cls(self.parameters(), **opt_kwargs) and
        scheduler=scheduler_cls(optimizer, **scheduler_kwargs)

        You can also override this method and write one that returns
        {"optimizer": optimizer, "lr_scheduler": scheduler}

        Documentation:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        Uses the attributes:
        - self.optimizer_params: dictionary with "class" key such as torch.optim.Adam,
            and "kwargs", dict of keyword args for class's init
        - self.scheduler_params: dictionary with "class" key such as
            torch.optim.lr_scheduler.StepLR, and and "kwargs", dict of keyword args for class's init
        """

        # self.optimizer_params dictionary has "class" and "kwargs" keys
        # copy the kwargs dict to avoid modifying the original values
        optimizer = self.optimizer_params["class"](
            self.network.parameters(), **self.optimizer_params["kwargs"].copy()
        )

        # self.scheduler_params dictionary has "class" key and kwargs for init
        scheduler = self.lr_scheduler_params["class"](
            optimizer, **self.lr_scheduler_params["kwargs"].copy()
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def validation_step(self, samples, batch_idx, dataloader_idx=0):
        batch_tensors, batch_labels = samples
        batch_tensors = batch_tensors.to(self.device)
        batch_labels = batch_labels.to(self.device)

        batch_size = len(batch_tensors)
        logits = self.network(batch_tensors)
        loss = self.loss_fn(logits, batch_labels)
        self.log(
            f"val{dataloader_idx}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch_tensors),
        )

        # compute and log any metrics in self.torch_metrics
        # TODO: consider using validation set names rather than integer index
        # (would have to store a set of names for the validation set)
        batch_metrics = {
            f"val{dataloader_idx}_{name}": metric.to(self.device)(
                logits.detach(), batch_labels.detach().int()
            ).cpu()
            for name, metric in self.torch_metrics.items()
        }
        self.log_dict(
            batch_metrics, on_epoch=True, on_step=False, batch_size=batch_size
        )
        # when on_epoch=True, compute() is called to reset the metric at epoch end

        return loss

    # reloading models from checkpoints:
    # model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt") # can override hyperparams
    # # disable randomness, dropout, etc...
    # model.eval()
    # # predict with the model
    # y_hat = model(x)
    # hyperparameters passed to init are saved and reloaded
    # resume training
    # model = MyLitModel()
    # trainer = Trainer()
    # # automatically restores model, epoch, step, LR schedulers, etc...
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
    # can also call trainer.validate()

    # self.save_hyperparameters() create .hparams dictionary?
    # load:
    # model = MyClass.load_from_checkpoint(PATH, override_hparam=new_value)

    # previously used in eval(): #TODO add this validation
    # maybe using at_train_start hook
    # # check for invalid label values
    # assert (
    #     targets.max(axis=None) <= 1 and targets.min(axis=None) >= 0
    # ), "Labels must in range [0,1], but found values outside range"

    # # remove all samples with NaN for a prediction before evaluating
    # targets = targets[~np.isnan(scores).any(axis=1), :]
    # scores = scores[~np.isnan(scores).any(axis=1), :]

    def train_dataloader(
        self,
        samples,
        bypass_augmentations=False,
        collate_fn=collate_audio_samples,
        **kwargs,
    ):
        """generate dataloader for training

        train_loader samples batches of images + labels from training set

        Args:
            samples: list of files or pd.DataFrame with multi-index ['file','start_time','end_time']
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
        """generate dataloader for inference (predict/validate/test)"""
        return self.inference_dataloader_cls(
            samples=samples,
            preprocessor=self.preprocessor,
            shuffle=False,  # keep original order
            pin_memory=False if self.device == torch.device("cpu") else True,
            collate_fn=collate_fn,
            **kwargs,
        )

    # def save_torch_dict(self, path):
    #     """save model to file for use in other opso versions

    #      WARNING: this does not save any preprocessing or augmentation
    #         settings or parameters, or other attributes such as the training
    #         parameters or loss function. It only saves architecture, weights,
    #         classes, sample shape, sample duration, and single_target.

    #     To save the entire pickled model object (recover all parameters and
    #     settings), use model.save() instead. Note that models saved with
    #     model.save() will not work across different versions of OpenSoundscape.

    #     To recreate the model after saving with this function, use CNN.from_torch_dict(path)

    #     Args:
    #         path: file path for saved model object

    #     Effects:
    #         saves a file using torch.save() containing model weights and other information
    #     """

    #     # warn the user if the achirecture can't be re-created from the
    #     # string name (self.architecture_name)
    #     if not self.architecture_name in cnn_architectures.list_architectures():
    #         warnings.warn(
    #             f"""\n The value of `self.architecture_name` ({self.architecture_name})
    #                 is not an architecture that can be generated in OpenSoundscape. Using
    #                 CNN.from_torch_dict on the saved file will cause an error. To fix this,
    #                 you can use .save() instead of .save_torch_model, or change
    #                 `self.architecture_name` to one of the architecture name strings listed by
    #                 opensoundscape.ml.cnn_architectures.list_architectures()
    #                 if this architecture is supported."""
    #         )

    #     os.makedirs(Path(path).parent, exist_ok=True)

    #     # save just the basics, loses preprocessing/other settings
    #     torch.save(
    #         {
    #             "weights": self.network.state_dict(),
    #             "classes": self.classes,
    #             "architecture": self.architecture_name,
    #             "sample_duration": self.preprocessor.sample_duration,
    #             "single_target": self.single_target,
    #             "sample_shape": [
    #                 self.preprocessor.height,
    #                 self.preprocessor.width,
    #                 self.preprocessor.channels,
    #             ],
    #         },
    #         path,
    #     )

    # @classmethod
    # def from_torch_dict(cls, path):
    #     """load a model saved using CNN.save_torch_dict()

    #     Args:
    #         path: path to file saved using CNN.save_torch_dict()

    #     Returns:
    #         new CNN instance

    #     Note: if you used .save() instead of .save_torch_dict(), load
    #     the model using cnn.load_model(). Note that the model object will not load properly
    #     across different versions of OpenSoundscape. To save and load models across
    #     different versions of OpenSoundscape, use .save_torch_dict(), but note that
    #     preprocessing and other customized settings will not be retained.
    #     """
    #     model_dict = torch.load(path)
    #     state_dict = model_dict.pop("weights")
    #     model = cls(**model_dict)
    #     model.network.load_state_dict(state_dict)
    #     return model

    # def save_weights(self, path):
    #     """save just the weights of the network

    #     This allows the saved weights to be used more flexibly than model.save()
    #     which will pickle the entire object. The weights are saved in a pickled
    #     dictionary using torch.save(self.network.state_dict())

    #     Args:
    #         path: location to save weights file
    #     """
    #     torch.save(self.network.state_dict(), path)

    # def load_weights(self, path, strict=True):
    #     """load network weights state dict from a file

    #     For instance, load weights saved with .save_weights()
    #     in-place operation

    #     Args:
    #         path: file path with saved weights
    #         strict: (bool) see torch.load()
    #     """
    #     self.network.load_state_dict(torch.load(path), strict=strict)


class SpectrogramLightningModule(OpenSoundscapeLightningModule):
    def __init__(
        self,
        architecture,
        classes,
        sample_duration,
        single_target=False,
        channels=1,
        sample_shape=[224, 224, 1],
    ):
        super(SpectrogramLightningModule, self).__init__()
        self.classes = classes
        self.single_target = single_target  # if True: predict only class w max score
        self.name = "SpectrogramLightningModule"
        self.type = type(self)

        ### ARCHITECTURE ###
        # allow user to pass a string, in which case we look up the architecture
        # in cnn_architectures.ARCH_DICT and instantiate it
        if type(architecture) == str:
            assert architecture in cnn_architectures.list_architectures(), (
                f"architecture must be a pytorch model object or string matching "
                f"one of cnn_architectures.list_architectures() options. Got {architecture}"
            )
            self.architecture_name = architecture
            architecture = cnn_architectures.ARCH_DICT[architecture](
                len(classes), num_channels=channels
            )
        else:
            assert issubclass(
                type(architecture), torch.nn.Module
            ), "architecture must be a string or an instance of a subclass of torch.nn.Module"
            if channels != 3:
                # can we try to check if first layer expects input with channels=channels?
                warnings.warn(
                    f"Make sure your architecture expects the number of channels in "
                    f"your input samples ({channels}). "
                    f"Pytorch architectures expect 3 channels by default."
                )
            self.architecture_name = str(type(architecture))
        self.network = architecture

        ### PREPROCESSOR ###
        self.preprocessor = SpectrogramPreprocessor(
            sample_duration=sample_duration,
            height=sample_shape[0],
            width=sample_shape[1],
            channels=sample_shape[2],
        )

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
        if self.single_target:
            self.torch_metrics = {
                "map": MulticlassAveragePrecision(
                    len(self.classes), validate_args=False, thresholds=50
                ),
                # "class_ap": MulticlassAveragePrecision(
                #     len(self.classes),
                #     validate_args=False,
                #     average=None,
                #     thresholds=50,
                # ),
                # TODO: .log() doesn't allow logging lists of values - how should we log per-class metrics?
                "auroc": MulticlassAUROC(
                    len(self.classes),
                    validate_args=False,
                    thresholds=50,  # speeds up computation
                    average="macro",
                ),
                # "class_auroc": MulticlassAUROC(
                #     len(self.classes), validate_args=False, thresholds=50, average=None
                # ),
            }
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

        self.save_hyperparameters()
