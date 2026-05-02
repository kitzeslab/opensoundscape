import warnings

from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import opensoundscape
from opensoundscape.ml.datasets import EmbeddingDataset, HopliteDataset
from opensoundscape.ml.loss import BCELossWeakNegatives
from opensoundscape.ml.utils import _version_mismatch_warn, _infinite_dataloader
from opensoundscape.vector_database import _require_hoplite
from opensoundscape.ml.loss import BCELossWeakNegatives
from opensoundscape.ml.datasets import HopliteDataset
from opensoundscape.vector_database import find_matching_windows, windows_to_dataframe


class MLPClassifier(torch.nn.Module):
    """initialize a fully connected NN (MLP) with ReLU activations

    Args:
        input_size: length of 1-d tensors passed as input samples
        output_size: number of classes at the output layer
        hidden_layer_sizes: default () empty tuple creates a 1-layer regression classifier,
            specify sequence of hidden layers by the number of elements. For example (100,)
            creates 1 hidden layer with 100 element
        classes (optional): list of class names, if provided should have len=output_size
            - default: None
        weights: optionally pass a pytorch weight_dict of model weights to load
            default None initializes the model with random weights
    """

    def __init__(
        self, input_size, output_size, hidden_layer_sizes=(), classes=None, weights=None
    ):
        super().__init__()

        self.in_features = input_size
        self.out_features = output_size
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.classes = classes

        if classes is not None:
            assert (
                len(classes) == output_size
            ), f"if specified, classes must have length {output_size}, but found {len(classes)}"
            classes = list(classes)  # convert to list if tuple

        # compose the MLP network:
        # add fully connected layers and RELU activations
        self.add_module("hidden_layers", torch.nn.Sequential())
        shapes = [input_size] + list(hidden_layer_sizes) + [output_size]
        for i, (in_size, out_size) in enumerate(zip(shapes[:-2], shapes[1:-1])):
            self.hidden_layers.add_module(
                f"layer_{i}", torch.nn.Linear(in_size, out_size)
            )
            self.hidden_layers.add_module(f"relu_{i}", torch.nn.ReLU())

        # add a final fully connected layer (the only layer if no hidden layers)
        self.add_module("classifier", torch.nn.Linear(shapes[-2], shapes[-1]))

        # hint to opensoundscape which layer is the final classifier layer
        self.classifier_layer = "classifier"

        # try loading the weights dictionary if provided
        if weights is not None:
            try:
                self.load_state_dict(weights)
            except Exception as e:
                raise ValueError(
                    f"Error loading weights. Ensure the weights match the model architecture."
                ) from e

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x

    def fit(
        self,
        train_features,
        train_labels,
        validation_features=None,
        validation_labels=None,
        batch_size=128,
        steps=1000,
        optimizer=None,
        criterion=None,
        device=torch.device("cpu"),
        validation_interval=1,
        logging_interval=100,
        early_stopping_patience=None,
    ):
        """train a PyTorch model on features and labels with batching and early stopping

        Assumes all data can fit in memory. Training uses batched DataLoaders for efficient processing.
        If validation data is provided, the model with the lowest validation loss is automatically
        restored at the end of training (early stopping).

        Defaults are for multi-target label problems and assume train_labels is an array of 0/1
        of shape (n_samples, n_classes)

        Note: this is a convenience wrapper around opensoundscape.ml.shallow_classifier.fit()

        Args:
            model: a torch.nn.Module object to train

            train_features: input features for training, often embeddings; should be a valid input to
            model(); generally shape (n_samples,n_features)

            train_labels: labels for training, generally one-hot encoded with shape
            (n_samples,n_classes); should be a valid target for criterion()

            validation_features: input features for validation; if None, does not perform validation

            validation_labels: labels for validation; if None, does not perform validation

            batch_size: batch size for training; if fewer samples than batch_size,
                the entire dataset is used as a single batch
                [Default: 128]

            steps: number of training steps (epochs); each step, all data is passed forward and
            backward, and the optimizer updates the weights
                [Default: 1000]

            optimizer: torch.optim optimizer to use; default None uses AdamW

            criterion: loss function to use; default None uses BCEWithLogitsLoss (appropriate for
            multi-label classification)

            device: torch.device to use; default is torch.device('cpu')

            validation_interval: how often to validate the model during training; if validation_features
            and validation_labels are provided, validation is performed every validation_interval steps

            logging_interval: how often to print training progress; progress is logged every
            logging_interval steps when validation is performed

            early_stopping_patience: if provided and validation data is available, training will stop
            early if validation loss doesn't improve for this many steps (not validation evaluations)
            [Default: None, which means no early stopping]
        """
        return fit(
            model=self,
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            batch_size=batch_size,
            steps=steps,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            validation_interval=validation_interval,
            logging_interval=logging_interval,
            early_stopping_patience=early_stopping_patience,
        )

    def save(self, path):
        torch.save(
            {
                "input_size": self.in_features,
                "output_size": self.out_features,
                "classes": self.classes,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "weights": self.state_dict(),
                "opensoundscape_version": opensoundscape.__version__,
            },
            path,
        )

    @classmethod
    def load(cls, path, **kwargs):
        """load object saved with self.save(); **kwargs like map_location are passed to torch.load"""
        model_dict = torch.load(path, **kwargs)
        opso_version = model_dict.pop("opensoundscape_version")
        _version_mismatch_warn(opso_version)
        # all other keys are used as args to __init__
        return cls(**model_dict)

    @classmethod
    def from_torch_linear(cls, linear_layer, classes=None):
        """initialize an MLPClassifier from a torch.nn.Linear layer

        Initializes 1-layer MLP, copying weights from linear_layer

        Args:
            linear_layer: a torch.nn.Linear layer whose weight and bias will be used to initialize the classifier layer
            of the MLPClassifier; should have shape (output_size, input_size) for weight and (output_size,) for bias
            classes (optional): list of class names, if provided should have len=output_size
                default: None
        """
        if not isinstance(linear_layer, torch.nn.Linear):
            raise ValueError(f"linear_layer must be an instance of torch.nn.Linear")
        input_size = linear_layer.in_features
        output_size = linear_layer.out_features
        model = cls(
            input_size=input_size,
            output_size=output_size,
            classes=classes,
        )
        # copy weights and bias from the linear layer to the classifier layer of the MLPClassifier
        model.classifier.weight.data.copy_(linear_layer.weight.data)
        model.classifier.bias.data.copy_(linear_layer.bias.data)
        return model


def fit_on_hoplite(
    classifier,
    hoplite_db,
    train_df,
    validation_df=None,
    batch_size=128,
    steps=10_000,
    optimizer=None,
    criterion=None,
    device=torch.device("cpu"),
    validation_interval=100,
    logging_interval=100,
    early_stopping_patience=None,
    progress_bar=False,
    **kwargs,
):
    """train a PyTorch classifier on Hoplite Embedding DB and label dataframe

    Defaults are for multi-target label problems and assume train_df is a dataframe of 0/1
    per class with multi-index (file, start_time, end_time)

    Args:
        classifier: a torch.nn.Module object to train

        hoplite_db: a HopliteDB instance containing the embeddings to train on

        train_df: labels for training, generally one-hot encoded with shape
        (n_samples,n_classes); should be a valid target for criterion()

        validation_df: labels for validation; if None, does not perform validation

        validation_labels: labels for validation; if None, does not perform validation

        batch_size: batch size for training; if fewer samples than batch_size,
            the entire dataset is used as a single batch
            [Default: 128]

        steps: number of training steps (epochs; each step, all data is passed forward and
            backward, and the optimizer updates the weights
            [Default: 10_000]

        optimizer: torch.optim optimizer to use; default None uses AdamW

        criterion: loss function to use; default None uses BCELossWeakNegatives() (appropriate for
        multi-label classification); this loss function treats NaN labels as weak negatives,
            using a default weight of 0.01 for NaN labels compared to strong labels

        device: torch.device to use; default is torch.device('cpu')

        validation_interval: how often to validate the model during training; if validation_features
        and validation_labels are provided, validation is performed every validation_interval steps

        logging_interval: how often to print training progress; progress is logged every
        logging_interval steps when validation is performed

        early_stopping_patience: if provided and validation data is available, training will stop
        early if validation loss doesn't improve for this many steps (not validation evaluations)
        [Default: None, which means no early stopping]

        progress_bar: whether to show a progress bar during training; default False

        **kwargs: additional keyword arguments passed to HopliteDataset; see HopliteDataset.__init__()
    """
    _require_hoplite()
    # if no optimizer or criterion provided, use default AdamW and BCEWithLogitsLoss
    if optimizer is None:
        optimizer = torch.optim.AdamW(classifier.parameters())
    if criterion is None:
        criterion = BCELossWeakNegatives()

    # move the model to the device
    classifier.to(device)

    # TODO: could switch to iterating the window_ids directly and using db.get_embeddings_batch
    # (just make label df with index: window_id instead of file/start/end time)
    train_dataset = HopliteDataset(hoplite_db, train_df, **kwargs)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # if validation data provided, convert to tensors and move to the device
    best_val_loss = float("inf")
    best_model_state = None
    best_step = -1
    if validation_df is not None:
        validation_dataset = HopliteDataset(hoplite_db, validation_df, **kwargs)
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    train_loader = _infinite_dataloader(train_loader)

    for step in tqdm(range(steps), disable=not progress_bar):
        classifier.train()
        batch_features, batch_labels = next(train_loader)

        batch_features = torch.as_tensor(batch_features, dtype=torch.float32).to(device)
        batch_labels = torch.as_tensor(batch_labels, dtype=torch.float32).to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = classifier(batch_features)

        # compute loss
        loss = criterion(outputs, batch_labels)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # Validation (optional)
        if validation_df is not None and (step + 1) % validation_interval == 0:
            classifier.eval()
            with torch.no_grad():
                # val_outputs = model(validation_features)
                # val_loss = criterion(val_outputs, validation_labels)
                val_outputs = []
                val_loss = 0.0
                for val_batch_features, val_batch_labels in validation_loader:
                    val_batch_features = torch.as_tensor(
                        val_batch_features, dtype=torch.float32
                    ).to(device)
                    val_batch_labels = torch.as_tensor(
                        val_batch_labels, dtype=torch.float32
                    ).to(device)

                    # forward pass
                    val_output = classifier(val_batch_features)
                    val_outputs.append(val_output)

                    # compute loss
                    val_loss += criterion(val_output, val_batch_labels).item()
                val_outputs = torch.cat(val_outputs, dim=0)
                val_loss /= len(validation_loader)

            # Check if this is the best validation loss and save model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = classifier.state_dict()
                best_step = step

            # Store metrics
            try:
                auroc = roc_auc_score(
                    validation_df.values,
                    val_outputs.detach().cpu().numpy(),
                )
            except:
                auroc = float("nan")
            try:
                map = average_precision_score(
                    validation_df.values,
                    val_outputs.detach().cpu().numpy(),
                )
            except:
                map = float("nan")

            # log the loss and metrics
            if (step + 1) % logging_interval == 0:
                print(
                    f"Epoch {step+1}/{steps}, Loss: {loss:0.3f}, Val Loss: {val_loss:0.3f}"
                )
                print(f"\tval AU ROC: {auroc:0.3f}")
                print(f"\tval MAP: {map:0.3f}")

            # Check early stopping condition based on steps since last improvement
            if early_stopping_patience is not None and best_step >= 0:
                # Calculate steps since last improvement
                steps_since_improvement = step - best_step
                if steps_since_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping triggered after {step + 1} steps (patience: {early_stopping_patience})"
                    )
                    print(
                        f"Best validation loss: {best_val_loss:0.3f} at step {best_step + 1}"
                    )
                    break

    # Load the best model state if validation was used
    if validation_df is not None and best_model_state is not None:
        classifier.load_state_dict(best_model_state)
        print(
            f"Loaded best model with validation loss: {best_val_loss:0.3f} at step {best_step + 1} of {steps}"
        )
        # compute metrics for the best model
        classifier.eval()
        with torch.no_grad():
            val_outputs = []
            for val_batch_features, _ in validation_loader:
                val_batch_features = torch.as_tensor(
                    val_batch_features, dtype=torch.float32
                ).to(device)
                val_output = classifier(val_batch_features)
                val_outputs.append(val_output)
            val_outputs = torch.cat(val_outputs, dim=0)
        try:
            auroc = roc_auc_score(
                validation_df.values,
                val_outputs.detach().cpu().numpy(),
            )
        except:
            auroc = float("nan")
        try:
            map = average_precision_score(
                validation_df.values,
                val_outputs.detach().cpu().numpy(),
            )
        except:
            map = float("nan")
        per_class_auroc = []
        for i in range(validation_df.shape[1]):
            try:
                per_class_auroc.append(
                    roc_auc_score(
                        validation_df.values[:, i],
                        val_outputs[:, i].detach().cpu().numpy(),
                    )
                )
            except:
                per_class_auroc.append(float("nan"))
        best_model_val_metrics = {
            "loss": best_val_loss,
            "auroc": auroc,
            "map": map,
            "per_class_auroc": per_class_auroc,
        }
    else:
        best_model_val_metrics = None
    print("Training complete")
    return best_model_val_metrics


def fit(
    model,
    train_features,
    train_labels,
    validation_features=None,
    validation_labels=None,
    batch_size=128,
    steps=1000,
    optimizer=None,
    criterion=None,
    device=torch.device("cpu"),
    validation_interval=1,
    logging_interval=100,
    early_stopping_patience=None,
):
    """train a PyTorch model on features and labels with batching and early stopping

    Assumes all data can fit in memory. Training uses batched DataLoaders for efficient processing.
    If validation data is provided, the model with the lowest validation loss is automatically
    restored at the end of training (early stopping).

    Defaults are for multi-target label problems and assume train_labels is an array of 0/1
    of shape (n_samples, n_classes)

    Args:
        model: a torch.nn.Module object to train

        train_features: input features for training, often embeddings; should be a valid input to
        model(); generally shape (n_samples,n_features)

        train_labels: labels for training, generally one-hot encoded with shape
        (n_samples,n_classes); should be a valid target for criterion()

        validation_features: input features for validation; if None, does not perform validation

        validation_labels: labels for validation; if None, does not perform validation

        batch_size: batch size for training; if fewer samples than batch_size,
            the entire dataset is used as a single batch
            [Default: 128]

        steps: number of training steps forward/backward passes on one batch
            [Default: 1000]

        optimizer: torch.optim optimizer to use; default None uses AdamW

        criterion: loss function to use; default None uses BCELossWeakNegatives() (appropriate for
        multi-label classification); this loss function treats NaN labels as weak negatives,
        using a default weight of 0.01 for NaN labels compared to strong labels

        device: torch.device to use; default is torch.device('cpu'); can also be e.g.
        torch.device('cuda:0') for first CUDA GPU or torch.device('mps') for Mac with M1/M2

        validation_interval: how often to validate the model during training; if validation_features
        and validation_labels are provided, validation is performed every validation_interval steps

        logging_interval: how often to print training progress; progress is logged every
        logging_interval steps when validation is performed

        early_stopping_patience: if provided and validation data is available, training will stop
        early if validation loss doesn't improve for this many steps (not validation evaluations)
        [Default: None, which means no early stopping]
    """
    # if no optimizer or criterion provided, use default AdamW and BCEWithLogitsLoss
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())
    if criterion is None:
        criterion = BCELossWeakNegatives()

    # move the model to the device
    model.to(device)

    # if dataframes, extract values
    if isinstance(train_features, pd.DataFrame):
        train_features = train_features.values
    if isinstance(train_labels, pd.DataFrame):
        train_labels = train_labels.values
    if validation_features is not None and isinstance(
        validation_features, pd.DataFrame
    ):
        validation_features = validation_features.values
    if validation_labels is not None and isinstance(validation_labels, pd.DataFrame):
        validation_labels = validation_labels.values

    # convert x and y to tensors and move to the device
    train_features = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)

    train_dataset = EmbeddingDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    def make_infinite(loader):
        while True:
            for data in loader:
                yield data

    infinite_data_loader = iter(make_infinite(train_loader))

    # if validation data provided, convert to tensors and move to the device
    best_val_loss = float("inf")
    best_model_state = None
    best_step = -1

    if validation_features is not None:
        validation_features = torch.as_tensor(
            validation_features, dtype=torch.float32, device=device
        )
        validation_labels = torch.as_tensor(
            validation_labels, dtype=torch.float32, device=device
        )
        validation_dataset = EmbeddingDataset(validation_features, validation_labels)
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    for step in range(steps):
        model.train()
        batch_features, batch_labels = next(infinite_data_loader)

        # # iterate over the training data in batches
        # for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(batch_features)

        # compute loss
        loss = criterion(outputs, batch_labels)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # Validation (optional)
        if validation_features is not None and (step + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                # val_outputs = model(validation_features)
                # val_loss = criterion(val_outputs, validation_labels)
                val_outputs = []
                val_loss = 0.0
                for val_batch_features, val_batch_labels in validation_loader:
                    val_batch_features = val_batch_features.to(device)
                    val_batch_labels = val_batch_labels.to(device)

                    # forward pass
                    val_output = model(val_batch_features)
                    val_outputs.append(val_output)

                    # compute loss
                    val_loss += criterion(val_output, val_batch_labels).item()
                val_outputs = torch.cat(val_outputs, dim=0)
                val_loss /= len(validation_loader)

            # Check if this is the best validation loss and save model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_step = step

            # Store metrics
            try:
                auroc = roc_auc_score(
                    validation_labels.detach().cpu().numpy(),
                    val_outputs.detach().cpu().numpy(),
                )
            except:
                auroc = float("nan")
            try:
                map = average_precision_score(
                    validation_labels.detach().cpu().numpy(),
                    val_outputs.detach().cpu().numpy(),
                )
            except:
                map = float("nan")

            # log the loss and metrics
            if (step + 1) % logging_interval == 0:
                print(
                    f"Step {step+1}/{steps}, Loss: {loss:0.3f}, Val Loss: {val_loss:0.3f}, val AU ROC: {auroc:0.3f}, val MAP: {map:0.3f}"
                )

            # Check early stopping condition based on steps since last improvement
            if early_stopping_patience is not None and best_step >= 0:
                # Calculate steps since last improvement
                steps_since_improvement = step - best_step
                if steps_since_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping triggered after {step + 1} steps (patience: {early_stopping_patience})"
                    )
                    print(
                        f"Best validation loss: {best_val_loss:0.3f} at step {best_step + 1}"
                    )
                    break

    # Load the best model state if validation was used
    if validation_features is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"Loaded best model with validation loss: {best_val_loss:0.3f} at step {best_step + 1} of {steps}"
        )
        # compute metrics for the best model
        model.eval()
        with torch.no_grad():
            val_outputs = []
            for val_batch_features, _ in validation_loader:
                val_batch_features = val_batch_features.to(device)
                val_output = model(val_batch_features)
                val_outputs.append(val_output)
            val_outputs = torch.cat(val_outputs, dim=0)
        try:
            auroc = roc_auc_score(
                validation_labels.detach().cpu().numpy(),
                val_outputs.detach().cpu().numpy(),
            )
        except:
            auroc = float("nan")
        try:
            map = average_precision_score(
                validation_labels.detach().cpu().numpy(),
                val_outputs.detach().cpu().numpy(),
            )
        except:
            map = float("nan")
        per_class_auroc = []
        for i in range(validation_labels.shape[1]):
            try:
                per_class_auroc.append(
                    roc_auc_score(
                        validation_labels[:, i].detach().cpu().numpy(),
                        val_outputs[:, i].detach().cpu().numpy(),
                    )
                )
            except:
                per_class_auroc.append(float("nan"))
        best_model_val_metrics = {
            "loss": best_val_loss,
            "auroc": auroc,
            "map": map,
            "per_class_auroc": per_class_auroc,
        }
    else:
        best_model_val_metrics = None
    print("Training complete")
    return best_model_val_metrics


# copy fit docstring to MLPClassifier.fit
MLPClassifier.fit.__doc__ = fit.__doc__


def augmented_embed(
    embedding_model,
    sample_df,
    n_augmentation_variants,
    batch_size=1,
    num_workers=0,
    device=torch.device("cpu"),
    audio_root=None,
):
    """Embed samples using augmentation during preprocessing

    Args:
        embedding_model: a model with an embed() method that takes a dataframe and returns
        embeddings (e.g. a pretrained opensoundscape model or Bioacoustics Model Zoo model like
        Perch, BirdNET, HawkEars)

        sample_df: dataframe with samples to embed

        n_augmentation_variants: number of augmented variants to generate for each sample

        batch_size: batch size for embedding; default 1

        num_workers: number of workers for embedding; default 0

        device: torch.device to use; default is torch.device('cpu')

    Returns:
        x_train, y_train: the embedded training samples and their labels, as torch.tensors
    """
    all_variants = []
    for _ in tqdm(range(n_augmentation_variants)):
        all_variants.append(
            torch.tensor(
                embedding_model.embed(
                    sample_df,
                    return_dfs=False,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    bypass_augmentations=False,
                    progress_bar=False,
                    audio_root=audio_root,
                )
            )
        )

    x_train = torch.vstack(all_variants).to(device)
    # duplicate the labels, which are the same for each variant
    y_train = [
        torch.tensor(sample_df.values).to(device).float()
    ] * n_augmentation_variants
    y_train = torch.vstack(y_train).to(device).float()

    return x_train, y_train


def fit_classifier_on_embeddings(
    embedding_model,
    classifier_model,
    train_df,
    validation_df,
    n_augmentation_variants=0,
    embedding_batch_size=1,
    embedding_num_workers=0,
    steps=1000,
    optimizer=None,
    criterion=None,
    device=torch.device("cpu"),
    early_stopping_patience=None,
    logging_interval=100,
    validation_interval=1,
    audio_root=None,
):
    """Embed samples with an embedding model, then fit a classifier on the embeddings

    wraps embedding_model.embed() with fit(clf,...)

    Also supports generating augmented variations of the training samples

    Note: if embedding takes a while and you might want to fit multiple times, consider embedding
    the samples first then running fit(...) rather than calling this function.

    Args:
        embedding_model: a model with an embed() method that takes a dataframe and returns embeddings
        (e.g. a pretrained opensoundscape model or Bioacoustics Model Zoo model like Perch, BirdNET, HawkEars)
        classifier_model: a torch.nn.Module object to train, e.g. MLPClassifier or final layer of CNN
        train_df: dataframe with training samples and labels; see opensoundscape.ml.cnn.train() train_df argument
        validation_df: dataframe with validation samples and labels; see opensoundscape.ml.cnn.train() validation_df
            if None, skips validation
        n_augmentation_variants: if 0 (default), embeds training samples without augmentation;
            if >0, embeds each training sample with stochastic augmentation num_augmentation_variants times
        embedding_batch_size: batch size for embedding; default 1
        embedding_num_workers: number of workers for embedding; default 0
        steps, optimizer, criterion, device: model fitting parameters, see fit()
        early_stopping_patience: if provided, training will stop early if validation loss doesn't improve
            for this many steps (not validation evaluations)
            [Default: None, which means no early stopping]
        logging_interval: how often to print training progress; progress is logged every logging_interval steps
            when validation is performed
        validation_interval: how often to validate the model during training; if validation_df is provided,
            validation is performed every validation_interval steps
        audio_root: if provided, used as prefix for audio files in train_df and validation_df;
            if None, assumes train_df and validation_df already have absolute audio paths

    Returns:
        x_train, y_train, x_val, y_val, metrics:
        the embedded training and validation samples and their labels, as torch.tensor, plus a
        dictionary of validation metrics for the best model found during training
    """
    if n_augmentation_variants > 0:
        print(
            f"Embedding the training samples {n_augmentation_variants} times with stochastic augmentation"
        )
        x_train, y_train = augmented_embed(
            embedding_model,
            train_df,
            n_augmentation_variants=n_augmentation_variants,
            batch_size=embedding_batch_size,
            num_workers=embedding_num_workers,
            device=device,
            audio_root=audio_root,
        )

    else:
        print(f"Embedding the training samples without augmentation")
        x_train = embedding_model.embed(
            train_df[[]],
            return_dfs=False,
            batch_size=embedding_batch_size,
            num_workers=embedding_num_workers,
            progress_bar=True,
            audio_root=audio_root,
        )
        y_train = train_df.values

    # cast to float tensor and move to device
    # MPS framework doesn't support float64, so force float32 (.float() is an alias for float32)
    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    if validation_df is None:
        x_val = None
        y_val = None
    else:
        print("Embedding the validation samples")
        x_val = embedding_model.embed(
            validation_df[[]],
            return_dfs=False,
            batch_size=embedding_batch_size,
            num_workers=embedding_num_workers,
            audio_root=audio_root,
        )
        y_val = validation_df.values

        # cast to float and move to device
        # note that MPS framework doesn't support float64, use float32 instead
        x_val = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        y_val = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    print("Fitting the classifier")
    metrics = fit(
        model=classifier_model,
        train_features=x_train,
        train_labels=y_train,
        validation_features=x_val,
        validation_labels=y_val,
        steps=steps,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        validation_interval=validation_interval,
        logging_interval=logging_interval,
        early_stopping_patience=early_stopping_patience,
    )

    # returning the embeddings and labels is useful
    # for re-training without re-embedding
    return x_train, y_train, x_val, y_val, metrics


def get_embeddings_from_hoplite(db, samples, **kwargs):
    _require_hoplite()
    dataset = HopliteDataset(db=db, samples=samples, **kwargs)
    return np.array([x for x in dataset])


def predict_on_hoplite(
    db,
    samples,
    classifier,
    clip_duration=None,
    batch_size=1024,
    return_df=True,
    device=torch.device("cpu"),
    **kwargs,
):
    """Apply model to embeddings from database for each clip in samples

    Args:
        db: hoplite database containing embeddings
        samples: a dataframe of clips or list of audio files
            dataframe with columns "file", "start_time", "end_time" specifying clips to apply the model to
        classifier: MLPClassifier object or other classifier object to call on the torch.tensor embeddings
        clip_duration: provide clip length (s) if passing files rather than pre-defined file/start_time/end_time clips
        batch_size: n samples simultaneously processed when applying classifier to embeddings; default 1024
        return_df: if True, returns a dataframe with the same index as samples and columns for each class;
            if False, returns a numpy array of predictions
            uses classifier.classes if available for df column names, otherwise uses integer column names
        **kwargs: additional keyword arguments to pass to HopliteDataset

    Returns:
        pandas.DataFrame or numpy.ndarray: predictions for each clip

    See also:
        select_from_hoplite if samples are already embedded and you wish to select filtered (random/top-scoring/all) clips

    """
    _require_hoplite()

    # move the model to the device
    classifier.to(device)

    dataset = HopliteDataset(db, samples, clip_duration=clip_duration, **kwargs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    classifier.eval()
    all_outputs = []
    with torch.no_grad():
        for batch_features, _ in dataloader:  # discard labels
            batch_features = torch.as_tensor(batch_features, dtype=torch.float32).to(
                device
            )

            # forward pass
            outputs = classifier(batch_features)
            all_outputs.append(outputs.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)

    if return_df:
        classes = (
            classifier.classes
            if hasattr(classifier, "classes")
            else range(all_outputs.shape[1])
        )
        all_outputs = pd.DataFrame(all_outputs, columns=classes)
        all_outputs.index = dataset.label_df.index
    return all_outputs


def select_from_hoplite(
    db,
    classifier,
    classes,
    k=5,
    strategy: Literal["top_k", "random_k", "all"] = "top_k",
    batch_size=1024,
    date_range=None,
    time_range=None,
    min_score=None,
    max_score=None,
    deployments=None,
    projects=None,
    recordings=None,
    deployments_filter=None,
    recordings_filter=None,
    windows_filter=None,
    annotations_filter=None,
    random_state=None,
    return_windows=False,
    progress_bar=False,
    warn_no_matches=False,
):
    """Extract top-scoring or random clips from the database based on classifier predictions and filters

    Args:
        db: hoplite database containing embeddings
        classifier: MLPClassifier object or other classifier object to call on the torch.tensor embeddings
        classes: list of class names to select clips for; if None, selects clips for every class in classifier
        k: number of clips to return per class; default 5 (ignored if strategy="all")
        strategy: which clips to select:
            "top_k" to return the top k clips for each class
            "random_k" to return k random clips
            "all" to return all clips (ignores `k`)
            default "top_k"
        batch_size: n samples simultaneously processed when applying classifier to embeddings; default 1024
        date_range: tuple of (start_date, end_date) to filter clips by date;
            Formats: datetime.datetime, datetime.date, or string in "YYYY-MM-DD" format; if None, does not filter by date
            Can pass (date,None) or (None,date) to filter by only start or end date, respectively
        time_range: tuple of (start_time, end_time) to filter clips by time of day; if None, does not filter by time of day
            Formats: datetime.datetime, datetime.time or string in "HH:MM:SS" format
            Note: filters by time of day of the _recording_ start time (rather than audio clip start time)
            Assumes time zone match between time_range values and recording timestamps in the database
        min_score: minimum score to filter clips by existing score in the database; if None, does not threshold by min score
        max_score: maximum score to filter clips by existing score in the database; if None, does not restrict by max score
        deployments: list of deployment names to filter by; if None, does not filter by deployment
        projects: list of project names to filter by; if None, does not filter by project
        recordings: list of recording names to filter by; if None, does not filter by recording
        deployments_filter: custom filter dict for deployments; if provided, overrides deployments argument
        recordings_filter: custom filter dict for recordings; if provided, overrides recordings argument
        windows_filter: custom filter dict for windows; if provided, overrides date_range, time_range arguments
        annotations_filter: custom filter dict for annotations in hoplite DB
        warn_no_matches: if True, raises a warning if no clips are found for a class after applying filters and score thresholds; default False

    Returns:
        dict of {class_name: list of matching windows} if return_windows=True; otherwise a dataframe with columns for class, score, and window info
    """
    np.random.seed(random_state)

    _require_hoplite()
    # find all matching clips
    from ml_collections import config_dict

    matching_windows = find_matching_windows(
        db=db,
        date_range=date_range,
        time_range=time_range,
        deployments=deployments,
        projects=projects,
        recordings=recordings,
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=windows_filter,
        annotations_filter=annotations_filter,
    )
    if len(matching_windows) == 0:
        if warn_no_matches:
            warnings.warn("No clips found matching the provided filters")
        if return_windows:
            return {}
        else:
            return pd.DataFrame(
                columns=[
                    "file",
                    "start_time",
                    "end_time",
                    "datetime",
                    "deployment",
                    "project",
                    "window_id",
                    "score",
                    "class",
                ]
            )

    # apply classifier in batches to matching windows
    all_scores = []
    device = next(classifier.parameters()).device  # probably just stay on CPU here?
    for i in tqdm(
        range(0, len(matching_windows), batch_size), disable=not progress_bar
    ):
        batch_windows = matching_windows[i : i + batch_size]
        batch_window_ids = [w.id for w in batch_windows]
        batch_embs = db.get_embeddings_batch(batch_window_ids)
        batch_embs_tensor = torch.as_tensor(
            batch_embs, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            batch_scores = classifier(batch_embs_tensor).cpu().numpy()
        all_scores.append(batch_scores)

    all_scores = np.concatenate(all_scores)  # shape (n_matching_windows, n_classes)

    if classes is None:
        classes = (
            classifier.classes
            if hasattr(classifier, "classes")
            else range(all_scores.shape[1])
        )
    else:
        # check that class names are valid
        if hasattr(classifier, "classes"):
            mismatch = set(classes) - set(classifier.classes)
            if len(mismatch) > 0:
                raise ValueError(
                    f"Invalid class names: {mismatch}. Class names must be in classifier.classes: {classifier.classes}"
                )
        else:
            if max(classes) >= all_scores.shape[1]:
                raise ValueError(
                    f"Invalid class indices: {set(classes) - set(range(all_scores.shape[1]))}. Class indices must be between 0 and {all_scores.shape[1]-1}"
                )

    class_dict = {c: i for i, c in enumerate(classes)}  # map class names to indices

    results = {}
    for class_name, clsidx in class_dict.items():
        cls_scores = all_scores[:, clsidx]
        cls_windows = np.array(matching_windows)

        # score filtering by min and max score
        mask = np.ones_like(cls_scores, dtype=bool)
        if min_score is not None:
            mask = mask & (cls_scores >= min_score)
        if max_score is not None:
            mask = mask & (cls_scores <= max_score)
        cls_scores = cls_scores[mask]
        cls_windows = np.array(cls_windows)[mask]

        if len(cls_windows) == 0 and warn_no_matches:
            warnings.warn(
                f"No clips found for class {class_name} after applying filters and score thresholds"
            )

        # select clips based on strategy
        if len(cls_windows) < k:
            # select all since fewer than k
            pass
        elif strategy == "top_k":
            indices_of_k_largest = np.argpartition(cls_scores, -k)[-k:]
            cls_windows = cls_windows[indices_of_k_largest]
            cls_scores = cls_scores[indices_of_k_largest]
        elif strategy == "random_k":
            selected_indices = np.random.choice(len(cls_windows), size=k, replace=False)
            cls_windows = cls_windows[selected_indices]
            cls_scores = cls_scores[selected_indices]
        elif strategy == "all":
            pass  # retain all
        else:
            raise ValueError(
                f"Invalid strategy: {strategy}. Must be one of 'top_k', 'random_k', or 'all'."
            )
        # add score to each window
        for w, s in zip(cls_windows, cls_scores):
            w.score = s

        results[class_name] = cls_windows

    # optimization: if random and not using score filtering, we could select random windows first the apply classifier
    if return_windows:
        return results
    # return dataframe:
    per_class_results = []
    for class_name, windows in results.items():
        df = windows_to_dataframe(windows, extra_keys=["score"])
        df["class"] = class_name
        per_class_results.append(df)
    return pd.concat(per_class_results)


def count_dets_hoplite(
    db,
    classifier,
    classes,
    min_score=None,
    max_score=None,
    score_bins=None,
    batch_size=1024,
    date_range=None,
    time_range=None,
    deployments=None,
    projects=None,
    recordings=None,
    deployments_filter=None,
    recordings_filter=None,
    windows_filter=None,
    annotations_filter=None,
    progress_bar=False,
):
    """Count detections in score bins/ranges based on classifier predictions and filters

    Compared to select_from_hoplite, this function does not return the selected clips but just
    counts the number of clips in each score bin/range for each class. This can be quick and
    memory efficient for counting detections in large datasets if you don't need clip info.

    Args:
        db: hoplite database containing embeddings classifier: MLPClassifier object or other
        classifier object to call on the torch.tensor embeddings classes: list of class names to
        select clips for; if None, selects clips for every class in classifier min_score: minimum
        score to filter clips by existing score in the database; if None, does not threshold by min
        score max_score: maximum score to filter clips by existing score in the database; if None,
        does not restrict by max score score_bins: if provided, a list of tuples (low, high) score
        ranges to count detections in
            - if None, reports all scores above min_score and below max_score in a single bin
            - if provided, min_score and max_score are ignored and bins are determined by score_bins
        batch_size: n samples simultaneously processed when applying classifier to embeddings;
            default 1024
        date_range: tuple of (start_date, end_date) to filter clips by date;
            Formats: datetime.datetime, datetime.date, or string in "YYYY-MM-DD" format; if None,
            does not filter by date Can pass (date,None) or (None,date) to filter by only start or
            end date, respectively
        time_range: tuple of (start_time, end_time) to filter clips by time of day; if None, does
        not filter by time of day
            Formats: datetime.datetime, datetime.time or string in "HH:MM:SS" format Note: filters
            by time of day of the _recording_ start time (rather than audio clip start time)
            Assumes time zone match between time_range values and recording timestamps in the database
        deployments: list of deployment names to filter by; if None, does not filter by deployment
        projects: list of project names to filter by; if None, does not filter by project
        recordings: list of recording names to filter by; if None, does not filter by recording
        deployments_filter: custom filter dict for deployments; if provided, overrides deployments
            argument
        recordings_filter: custom filter dict for recordings; if provided, overrides
            recordings argument
        windows_filter: custom filter dict for windows; if provided, overrides date_range,
        time_range arguments
        annotations_filter: custom filter dict for annotations in hoplite DB

    Returns:
        counts: dict of dicts with counts[class][bin_range] = count of clips for class in score bin;
        if score_bins is None, bin_range is (min_score, max_score)
        (if min_score and/or max_score are also None, uses -inf &/or +inf)
    """

    _require_hoplite()

    if classes is None:
        classes = (
            classifier.classes
            if hasattr(classifier, "classes")
            else range(classifier.out_features)
        )
    else:
        # check that class names are valid
        if hasattr(classifier, "classes"):
            mismatch = set(classes) - set(classifier.classes)
            if len(mismatch) > 0:
                raise ValueError(
                    f"Invalid class names: {mismatch}. Class names must be in classifier.classes: {classifier.classes}"
                )
        else:
            n_classes = classifier.out_features
            if max(classes) >= n_classes:
                raise ValueError(
                    f"Invalid class indices: {set(classes) - set(range(n_classes))}. "
                    f"Class indices must be between 0 and {n_classes-1}."
                )

    class_dict = {c: i for i, c in enumerate(classes)}  # map class names to indices

    # find all matching clips
    matching_windows = find_matching_windows(
        db=db,
        date_range=date_range,
        time_range=time_range,
        deployments=deployments,
        projects=projects,
        recordings=recordings,
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=windows_filter,
        annotations_filter=annotations_filter,
    )

    # apply classifier in batches to matching windows
    if score_bins is None:
        min_score = float("-inf") if min_score is None else min_score
        max_score = float("inf") if max_score is None else max_score
        score_bins = [(min_score, max_score)]
    # else: score bins provides edges; min_score and max_score are ignored

    counts = {c: {bin_range: 0 for bin_range in score_bins} for c in classes}
    device = next(classifier.parameters()).device  # probably just stay on CPU here?
    for i in tqdm(
        range(0, len(matching_windows), batch_size), disable=not progress_bar
    ):
        batch_windows = matching_windows[i : i + batch_size]
        batch_window_ids = [w.id for w in batch_windows]
        batch_embs = db.get_embeddings_batch(batch_window_ids)
        batch_embs_tensor = torch.as_tensor(
            batch_embs, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            batch_scores = classifier(batch_embs_tensor).cpu().numpy()
        # immediately count scores in bins for each class to avoid storing all scores in memory
        for class_name, clsidx in class_dict.items():
            cls_scores = batch_scores[:, clsidx]
            for bin_range in score_bins:
                bin_mask = (cls_scores >= bin_range[0]) & (cls_scores < bin_range[1])
                counts[class_name][bin_range] += np.sum(bin_mask)
    return counts
