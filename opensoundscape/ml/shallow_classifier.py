from tqdm.autonotebook import tqdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


class MLPClassifier(torch.nn.Module):
    """initialize a fully connected NN with ReLU activations"""

    def __init__(self, input_size, output_size, hidden_layer_sizes=()):
        super().__init__()

        # constructor_name tuple hints to .load()
        # how to recreate the network with the appropriate shape
        self.constructor_name = (input_size, output_size, hidden_layer_sizes)

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

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x

    def fit(self, *args, **kwargs):
        """fit the weights on features and labels, without batching

        Args: see quick_fit()
        """
        quick_fit(self, *args, **kwargs)

    def save(self, path):
        # self.constructor_name holds tuple of input args: input_size, output_size, hidden_layer_sizes
        torch.save(
            {"weights": self.state_dict(), "architecture": self.constructor_name}, path
        )

    def load(self, path, **kwargs):
        """load object saved with self.save(); **kwargs like map_location are passed to torch.load"""
        model_dict = torch.load(path, **kwargs)
        # model_dict['architecture'] is tuple of init args: input_size, output_size, hidden_layer_sizes
        model = self.__init__(*model_dict["architecture"])
        # state dict is saved in 'weights' key
        model.load_state_dict(model_dict["weights"])
        return model


def quick_fit(
    model,
    train_features,
    train_labels,
    validation_features=None,
    validation_labels=None,
    steps=1000,
    optimizer=None,
    criterion=None,
    device=torch.device("cpu"),
):
    """train a PyTorch model on features and labels without batching

    Assumes all data can fit in memory, so that one step includes all data (i.e. step=epoch)

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

        steps: number of training steps (epochs); each step, all data is passed forward and
        backward, and the optimizer updates the weights
            [Default: 1000]

        optimizer: torch.optim optimizer to use; default None uses Adam

        criterion: loss function to use; default None uses BCEWithLogitsLoss (appropriate for
        multi-label classification)

        device: torch.device to use; default is torch.device('cpu')
    """
    # if no optimizer or criterion provided, use default Adam and CrossEntropyLoss
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    # move the model to the device
    model.to(device)

    # convert x and y to tensors and move to the device
    train_features = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32, device=device)

    # if validation data provided, convert to tensors and move to the device
    if validation_features is not None:
        validation_features = torch.tensor(
            validation_features, dtype=torch.float32, device=device
        )
        validation_labels = torch.tensor(
            validation_labels, dtype=torch.float32, device=device
        )

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_features)
        loss = criterion(outputs, train_labels)

        loss.backward()
        optimizer.step()

        # Validation (optional)
        if validation_features is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(validation_features)
                val_loss = criterion(val_outputs, validation_labels)
            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {step+1}/{steps}, Loss: {loss.item()}, Val Loss: {val_loss.item()}"
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
):
    """Embed samples with an embedding model, then fit a classifier on the embeddings

    wraps embedding_model.embed() with quick_fit(clf,...)

    Also supports generating augmented variations of the training samples

    Note: if embedding takes a while and you might want to fit multiple times, consider embedding
    the samples first then running quick_fit(...) rather than calling this function.

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
        steps, optimizer, criterion, device: model fitting parameters, see quick_fit()

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
    quick_fit(
        model=classifier_model,
        train_features=x_train,
        train_labels=y_train,
        validation_features=x_val,
        validation_labels=y_val,
        steps=steps,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # returning the embeddings and labels is useful
    # for re-training without re-embedding
    return x_train, y_train, x_val, y_val
