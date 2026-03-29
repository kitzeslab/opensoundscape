"""base class for machine learning models in opensoundscape"""

import warnings
from pathlib import Path

import torch
from torchmetrics import Accuracy

import opensoundscape
from opensoundscape.ml.dataloaders import SafeAudioDataloader, collate_audio_samples
from opensoundscape.ml.loss import BCEWithLogitsLoss_hot
from opensoundscape.preprocess.preprocessors import BasePreprocessor
from opensoundscape.utils import identity


class BaseModule:
    """base class for pytorch and lightning models in opensoundscape

    This class is intended to be subclassed by classes with more customized functionality.
    For example, see SpectrogramModule, SpectrogramClassifier, and LightningSpectrogramModule.
    """

    def __init__(self):
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
        batch_tensors, batch_labels = collate_audio_samples(samples)
        batch_tensors = batch_tensors.to(self.device)
        batch_labels = batch_labels.to(self.device)

        batch_size = len(batch_tensors)

        # automatic mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
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
                    "Make sure self.classifier property returns a torch.nn.Module object."
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
        collate_fn=identity,
        raise_errors=False,
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
            final_clip="extend",
            bypass_augmentations=bypass_augmentations,
            shuffle=True,  # SHUFFLE SAMPLES because we are training
            # use pin_memory=True when loading files on CPU and training on CUDA GPU
            pin_memory=self._should_pin_memory(),
            invalid_sample_behavior="raise" if raise_errors else "substitute",
            collate_fn=collate_fn,
            **kwargs,
        )

    def predict_dataloader(
        self, samples, collate_fn=identity, raise_errors=False, **kwargs
    ):
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
            pin_memory=self._should_pin_memory(),
            collate_fn=collate_fn,
            invalid_sample_behavior="raise" if raise_errors else "placeholder",
            **kwargs,
        )

    def _should_pin_memory(self):
        """determine whether to use pin_memory in dataloaders

        returns True if Torch training/inference is on CUDA device
        """
        if (
            hasattr(self, "device")
            and isinstance(self.device, torch.device)
            and self.device.type == "cuda"
        ):
            return True
        return False
