from tqdm.autonotebook import tqdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import opensoundscape
from opensoundscape.ml.utils import _version_mismatch_warn
from torch.utils.data import DataLoader, Dataset


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

        # constructor_name tuple hints to .load()
        # how to recreate the network with the appropriate shape
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.classes = classes

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

    def fit(self, *args, **kwargs):
        """fit the MLP classifier on features and labels

        Args: see shallow_classifier.fit()
        """
        fit(self, *args, **kwargs)

    def save(self, path):
        torch.save(
            {
                "input_size": self.input_size,
                "output_size": self.output_size,
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


class EmbeddingDataset(Dataset):
    """simple dataset wrapper for embedding features and labels"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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
    # if no optimizer or criterion provided, use default AdamW and BCEWithLogitsLoss
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    # move the model to the device
    model.to(device)

    # convert x and y to tensors and move to the device
    train_features = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32, device=device)

    train_dataset = EmbeddingDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # if validation data provided, convert to tensors and move to the device
    best_val_loss = float("inf")
    best_model_state = None
    best_step = -1
    if validation_features is not None:
        validation_features = torch.tensor(
            validation_features, dtype=torch.float32, device=device
        )
        validation_labels = torch.tensor(
            validation_labels, dtype=torch.float32, device=device
        )
        validation_dataset = EmbeddingDataset(validation_features, validation_labels)
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    for step in range(steps):
        model.train()

        # iterate over the training data in batches
        for batch_features, batch_labels in train_loader:
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

            # log the loss and metrics
            if (step + 1) % logging_interval == 0:
                print(
                    f"Epoch {step+1}/{steps}, Loss: {loss:0.3f}, Val Loss: {val_loss:0.3f}"
                )
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
                print(f"val AU ROC: {auroc:0.3f}")
                print(f"val MAP: {map:0.3f}")

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

    print("Training complete")


def augmented_embed(
    embedding_model,
    sample_df,
    n_augmentation_variants,
    batch_size=1,
    num_workers=0,
    device=torch.device("cpu"),
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

    Returns:
        x_train, y_train, x_val, y_val: the embedded training and validation samples and their labels, as torch.tensor
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
        )

    else:
        print(f"Embedding the training samples without augmentation")
        x_train = torch.tensor(
            embedding_model.embed(
                train_df,
                return_dfs=False,
                batch_size=embedding_batch_size,
                num_workers=embedding_num_workers,
                progress_bar=True,
            )
        ).to(device)
        y_train = torch.tensor(train_df.values).to(device).float()

    if validation_df is None:
        x_val = None
        y_val = None
    else:
        print("Embedding the validation samples")
        x_val = torch.tensor(
            embedding_model.embed(
                validation_df,
                return_dfs=False,
                batch_size=embedding_batch_size,
                num_workers=embedding_num_workers,
            )
        ).to(device)
        y_val = torch.tensor(validation_df.values).to(device).float()

    print("Fitting the classifier")
    fit(
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
    return x_train, y_train, x_val, y_val
