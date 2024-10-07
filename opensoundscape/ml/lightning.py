import warnings
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from opensoundscape.ml.utils import apply_activation_layer, check_labels
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.logging import wandb_table
from opensoundscape.ml.cnn import SpectrogramModule


class LightningSpectrogramModule(SpectrogramModule, L.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lightning_mode = True
        self.save_hyperparameters()

    def train(self, *args, **kwargs):
        """inherit train() method from LightningModule rather than SpectrogramModule

        this is just a method that sets True/False for trianing mode, it doesn't perform training
        """
        return L.LightningModule.train(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """inherit __call__ method from LightningModule rather than SpectrogramModule"""
        return L.LightningModule.__call__(self, *args, **kwargs)

    def forward(self, samples):
        """standard Lightning method defining action to take on each batch for inference

        typically returns logits (raw, untransformed model outputs)
        """
        batch_tensors, _ = samples
        batch_tensors = batch_tensors.to(self.device)
        return self.network(batch_tensors)

    def save(self, path, save_hooks=False, weights_only=False):
        """save model with weights using Trainer.save_checkpoint()

        load from saved file with LightningSpectrogramModule.load_from_checkpoint()

        Note: saving and loading model objects across OpenSoundscape versions
        will not work properly. Instead, use .save_weights() and .load_weights()
        (but note that architecture, customizations to preprocessing, training params,
        etc will not be retained using those functions).

        For maximum flexibilty in further use, save the model with both .save() and
        .save_torch_dict() or .save_weights().

        Args:
            path: file path for saved model object
            save_hooks: retain forward and backward hooks on modules
                [default: False] Note: True can cause issues when using
                wandb.watch()
        """
        import os
        from pathlib import Path
        import copy

        os.makedirs(Path(path).parent, exist_ok=True)

        # save a pickled model object; will not work across opso versions
        model_copy = copy.deepcopy(self)

        # save the preprocessor as a dictionary so we can reload/recreate it
        model_copy.hparams.preprocessor_dict = model_copy.preprocessor.to_dict()

        if not save_hooks:
            # remove all forward and backward hooks on network.modules()
            from collections import OrderedDict

            for m in model_copy.network.modules():
                m._forward_hooks = OrderedDict()
                m._backward_hooks = OrderedDict()

        t = L.Trainer()
        t.strategy.connect(model=model_copy)
        t.save_checkpoint(path, weights_only=weights_only)

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
            strict: (bool) see torch.Module.load_state_dict()
        """
        self.network.load_state_dict(torch.load(path), strict=strict)

    def fit_with_trainer(
        self,
        train_df,
        validation_df=None,
        epochs=1,
        batch_size=1,
        num_workers=0,
        save_path=".",  # TODO: TypeError: unsupported format string passed to NoneType.__format__ if None is passed
        invalid_samples_log="./invalid_training_samples.log",
        raise_errors=False,
        wandb_session=None,
        checkpoint_path=None,
        **kwargs,
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
            invalid_samples_log:
                file path: log all samples that failed in preprocessing
                (file written when training completes)
                - if None,  does not write a file
            raise_errors:
                if True, raise errors when preprocessing fails
                if False, just log the errors to unsafe_samples_log
            wandb_session: a wandb session to log to
                (Note: can also pass `logger` kwarg with any Lightning logger object)
                - pass the value returned by wandb.init() to progress log to a
                Weights and Biases run
                - if None, does not log to wandb
                For example:
                ```
                import wandb
                wandb.login(key=api_key) #find your api_key at https://wandb.ai/settings
                session = wandb.init(enitity='mygroup',project='project1',name='first_run')
                ...
                model.fit_with_trainer(...,wandb_session=session)
                session.finish()
                ```
            **kwargs: any arguments to pytorch_lightning.Trainer(), such as
                accelerator, precision, logger, accumulate_grad_batches, etc.
                Note: the `max_epochs` kwarg is overridden by the `epochs` argument

        Returns:
            a trained pytorch_lightning.Trainer object

        Effects:
            If wandb_session is provided, logs progress and samples to Weights
            and Biases. A random set of training and validation samples
            are preprocessed and logged to a table. Training progress, loss,
            and metrics are also logged.
            Use self.wandb_logging dictionary to change the number of samples
            logged.
        """
        kwargs["max_epochs"] = epochs
        kwargs["default_root_dir"] = save_path

        ### Input Validation ###
        check_labels(train_df, self.classes)
        if validation_df is not None:
            check_labels(validation_df, self.classes)

        # Validation: warn user if no validation set
        if validation_df is None:
            warnings.warn(
                "No validation set was provided. Model will be "
                "evaluated using the performance on the training set."
            )

        # Initialize Weights and Biases (wandb) logging
        # TODO: can we just use built-in lightning logging here instead?
        if wandb_session is not None:
            if not "logger" in kwargs:
                # if a logger was passed, don't override it
                # if not, use the wandb session as the logger
                kwargs["logger"] = L.pytorch.loggers.WandbLogger()

            # update the run config with information about the model
            wandb_session.config.update(self._generate_wandb_config())

            # update the run config with training parameters
            wandb_session.config.update(
                dict(
                    epochs=epochs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    optimizer_params=self.optimizer_params,
                    lr_scheduler_params=self.lr_scheduler_params,
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
                            train_df,
                            self.preprocessor,
                            bypass_augmentations=False,
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

        # Set Up DataLoaders
        train_dataloader = self.train_dataloader(
            samples=train_df,
            batch_size=batch_size,
            num_workers=num_workers,
            raise_errors=raise_errors,
        )

        # TODO: enable multiple validation sets
        val_loader = self.predict_dataloader(
            samples=validation_df,
            batch_size=batch_size,
            num_workers=num_workers,
            raise_errors=raise_errors,
            split_files_into_clips=True,
            clip_overlap=0,
            final_clip=None,
            bypass_augmentations=True,
        )

        # keep best epoch using a callback
        # add this to any user-specified callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            save_top_k=1,
            monitor="val0_loss" if validation_df is not None else "train_loss",
        )
        if "callbacks" in kwargs:
            kwargs["callbacks"].append(checkpoint_callback)
        else:
            kwargs["callbacks"] = [checkpoint_callback]

        ## Train ##

        # initialize lightning.Trainer with user args
        trainer = L.Trainer(**kwargs)

        # train
        # if checkpoint_path is provided, resumes training from training state of checkpoint path
        trainer.fit(self, train_dataloader, val_loader, ckpt_path=checkpoint_path)
        print("Training complete")
        if checkpoint_callback.best_model_score is not None:
            print(
                f"Best model with score {checkpoint_callback.best_model_score:.3f} is saved to {checkpoint_callback.best_model_path}"
            )

        # warn the user if there were invalid samples (samples that failed to preprocess)
        invalid_samples = train_dataloader.dataset.report(log=invalid_samples_log)
        print(
            f"{len(invalid_samples)} of {len(train_df)} total training "
            f"samples failed to preprocess",
        )

        return trainer

    def predict_with_trainer(
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
        # wandb_session=None,
        return_invalid_samples=False,
        lightning_trainer_kwargs=None,
        dataloader_kwargs=None,
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
            lightning_trainer_kwargs: dictionary of keyword args to pass to __call__,
                which are then passed to lightning.Trainer.__init__
                see lightning.Trainer documentation for options. [Default: None] passes no kwargs
            dataloader_kwargs: dictionary of keyword args to self.predict_dataloader()

        Returns:
            df of post-activation_layer scores
            - if return_invalid_samples is True, returns (df,invalid_samples)
            where invalid_samples is a set of file paths that failed to preprocess

        Effects:
            (1) wandb logging
            If wandb_session is provided, logs progress and samples to Weights
            and Biases. A random set of samples is preprocessed and logged to
            a table. Progress over all batches is logged. After prediction,
            top scoring samples are logged.
            Use self.wandb_logging dictionary to change the number of samples
            logged or which classes have top-scoring samples logged.

            (2) unsafe sample logging
            If unsafe_samples_log is not None, saves a list of all file paths that
            failed to preprocess in unsafe_samples_log as a text file

        Note: if loading an audio file raises a PreprocessingError, the scores
            for that sample will be np.nan

        """
        # kwargs: if None, pass empty dictionary
        lightning_trainer_kwargs = lightning_trainer_kwargs or {}
        dataloader_kwargs = dataloader_kwargs or {}

        # for convenience, convert str/pathlib.Path to list of length 1
        if isinstance(samples, (str, Path)):
            samples = [samples]

        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(
            samples=samples,
            split_files_into_clips=split_files_into_clips,
            overlap_fraction=overlap_fraction,
            clip_overlap=clip_overlap,
            clip_overlap_fraction=clip_overlap_fraction,
            clip_step=clip_step,
            final_clip=final_clip,
            bypass_augmentations=bypass_augmentations,
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

        # Could re-add logging samples to wandb table:
        # if wandb_session is not None:
        #     # update the run config with information about the model
        #     wandb_session.config.update(self._generate_wandb_config())

        #     # update the run config with prediction parameters
        #     wandb_session.config.update(
        #         dict(
        #             batch_size=batch_size,
        #             num_workers=num_workers,
        #             activation_layer=activation_layer,
        #         )
        #     )

        #     # Log a table of preprocessed samples to wandb
        #     wandb_session.log(
        #         {
        #             "Peprocessed_samples": wandb_table(
        #                 dataloader.dataset.dataset,
        #                 self.wandb_logging["n_preview_samples"],
        #             )
        #         }
        #     )

        ### Prediction/Inference ###
        # iterate dataloader and run inference (forward pass) to generate scores
        trainer = L.Trainer(**lightning_trainer_kwargs)
        pred_scores = torch.vstack(trainer.predict(self, dataloader))

        ### Apply activation layer ###
        pred_scores = apply_activation_layer(pred_scores, activation_layer)

        # return DataFrame with same index/columns as prediction_dataset's df
        df_index = dataloader.dataset.dataset.label_df.index
        score_df = pd.DataFrame(index=df_index, data=pred_scores, columns=self.classes)

        # warn the user if there were invalid samples (failed to preprocess)
        # and log them to a file
        invalid_samples = dataloader.dataset.report(log=invalid_samples_log)

        # log top-scoring samples per class to wandb table
        # if wandb_session is not None:
        #     classes_to_log = self.wandb_logging["top_samples_classes"]
        #     if classes_to_log is None:  # pick the first few classes if none specified
        #         classes_to_log = self.classes
        #         if len(classes_to_log) > 5:  # don't accidentally log hundreds of tables
        #             classes_to_log = classes_to_log[0:5]

        #     for i, c in enumerate(classes_to_log):
        #         top_samples = score_df.nlargest(
        #             n=self.wandb_logging["n_top_samples"], columns=[c]
        #         )
        #         # note: the "labels" of these samples are actually prediction scores
        #         dataset = AudioFileDataset(
        #             samples=top_samples,
        #             preprocessor=self.preprocessor,
        #             bypass_augmentations=True,
        #         )
        #         table = wandb_table(
        #             dataset=dataset,
        #             classes_to_extract=[c],
        #             drop_labels=True,
        #             gradcam_model=self if self.wandb_logging["gradcam"] else None,
        #         )
        #         wandb_session.log({f"Top_scoring_{c.replace(' ','_')}": table})

        if return_invalid_samples:
            return score_df, invalid_samples
        else:
            return score_df
