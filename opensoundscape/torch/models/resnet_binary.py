# a model class defines a model architecture, and these functions: self.train, predict, evaluate, save, load
# a model class is used in combination with a Dataset, which defines the augmentations/preprocessing and tensor shape provided to the model

#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torchvision

from opensoundscape.metrics import Metrics
import opensoundscape.torch.tensor_augment as tensaug
from opensoundscape.torch.models.utils import BaseModule
import yaml
from os import path
import time
import pandas as pd


class ResnetBinaryModel(BaseModule):
    def __init__(self, pretrained=False):
        super(ResnetBinaryModel, self).__init__()

        # for now, we'll assume the user is providing a train_dataset and test_dataset
        # that are instances of a Dataset class containing all preprocessing (supply tensorX, y)
        # what we want eventually: user provides the preprocessing pipeline and 2 dfs
        self.net = torchvision.models.resnet18(pretrained=pretrained)
        self.net.fc = torch.nn.Linear(
            in_features=self.net.fc.in_features, out_features=2
        )

    def predict(
        self, prediction_dataset, batch_size=1, num_workers=1, apply_softmax=False
    ):
        """Generate predictions on a dataset from a pytorch model object
        Input:
            prediction_dataset:
                            a pytorch dataset object that returns tensors, such as datasets.SingleTargetAudioDataset()
            batch_size:     The size of the batches (# files) [default: 1]
            num_workers:    The number of cores to use for batch preparation [default: 1]
                            - if you want to use all the cores on your machine, set it to 0 (this could freeze your computer)
            apply_softmax:  Apply a softmax activation layer to the raw outputs of the model
            label_dict:     List of names of each class, with indices corresponding to NumericLabels [default: None]
                            - if None, the dataframe returned will have numeric column names
                            - if list of class names, returned dataframe will have class names as column names
        Output:
            A dataframe with the CNN prediction results for each class and each file
        Notes:
            if label_dict is not None, the returned dataframe's columns will be class names instead of numeric labels
        """

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net.eval()
        self.net.to(self.device)

        dataloader = torch.utils.data.DataLoader(
            prediction_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # run prediction
        all_predictions = []
        for i, inputs in enumerate(dataloader):
            predictions = self.net(inputs["X"].to(self.device))
            if apply_softmax:
                softmax_val = softmax(predictions, 1).detach().cpu().numpy()
                for x in softmax_val:
                    all_predictions.append(x[1])  # keep the present, not absent
                labels = prediction_dataset.df.columns.values
            else:
                for x in predictions.detach().cpu().numpy():
                    all_predictions.append(list(x))  # .astype('float64')
                label = prediction_dataset.df.columns.values[0]
                labels = [label + "_absent", label + "_present"]

        img_paths = prediction_dataset.df.index.values
        pred_df = pd.DataFrame(index=img_paths, data=all_predictions, columns=labels)

        return pred_df

    def train(
        self,
        save_dir,
        train_dataset,
        valid_dataset,
        optimizer,
        loss_fn,
        epochs=25,
        batch_size=1,
        num_workers=0,
        log_every=5,
        tensor_augment=False,
        debug=False,
        print_logging=True,
        save_scores=False,
    ):
        """Train the model using examples from train_dataset and evaluate with valid_dataset
        Input:
            save_dir:       A directory to save intermediate results
            train_dataset:  The training Dataset, e.g. created by SingleTargetAudioDataset()
            valid_dataset:  The validation Dataset, e.g. created by SingleTargetAudioDataset()
            optimizer:      A torch optimizer, e.g. torch.optim.SGD(model.parameters(), lr=1e-3)
            loss_fn:        A torch loss function, e.g. torch.nn.CrossEntropyLoss()
            epochs:         The number of epochs [default: 25]
            batch_size:     The size of the batches [default: 1]
            num_workers:    The number of cores to use for batch preparation [default: 1]
            log_every:      Log statistics when epoch % log_every == 0 [default: 5]
            tensor_augment: Whether or not to use the tensor augment procedures [default: False]
            debug:          Whether or not to write intermediate images [default: False]
            print_logging:  Whether to print training progress to stdout [default: True]
            save_scores:    Whether to save the scores on the train/val set each epoch [default: False]
        Side Effects:
            Write a file `epoch-{epoch}.tar` containing (rate of `log_every`):
            - Model state dictionary
            - Optimizer state dictionary
            - Labels in YAML format
            - Train: loss, accuracy, precision, recall, and f1 score
            - Validation: accuracy, precision, recall, and f1 score
            - train_dataset.label_dict
            Write a metadata file with parameter values to save_dir/metadata.txt
        Output:
            None
        Effects:
            model parameters are saved to metadata
        """

        # TODO: eventually, we want the user to simply supply dataframes (I think)
        # and a preprocessing pipeline

        self.tensor_augment = tensor_augment
        # TODO: this raises the question of whether all these parameters belong to class or train() method
        self.print_logging = print_logging
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_scores = save_scores
        self.save_dir = save_dir
        self.current_epoch = 0

        # move this to its own function
        if self.save_dir is not None:
            # save model parameters to metadata file
            metadata = {
                "training_start_time": time.strftime("%X %x %Z"),
                "train_data_len": len(train_dataset),
                "valid_data_len": len(valid_dataset),
                "optimizer": str(self.optimizer),
                "loss_fn": str(self.loss_fn),
                "epochs": epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "log_every": log_every,
                "tensor_augment": self.tensor_augment,
                "debug": debug,
                "cuda_is_available:": torch.cuda.is_available(),
            }

            with open(path.join(self.save_dir, "metadata.txt"), "w") as f:
                f.writelines(yaml.dump(metadata))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # TODO: did we need self.labels_yaml? what am I breaking here?
        # self.labels_yaml = yaml.dump(train_dataset.label_dict)
        self.classes = train_dataset.labels

        # make a dataloader to supply training images from train_dataset
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # make a dataloader to supply training images from valid_dataset
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.net.to(self.device)

        # Model training
        # TODO: modify classes, now they are 1-hot encoded
        # TODO: Metrics is probably broken because it relies on single-label

        # Clases should be integer values
        # try:
        #     classes = [int(x) for x in train_dataset.label_dict.keys()]
        # except:
        #     raise ValueError(
        #         f"The classes should be integers! Got {train_dataset.label_dict.keys()}"
        #     )

        # loop for each training epoch
        # 1 epoch = seeing each training file 1 time
        for epoch in range(epochs):
            self.current_epoch = epoch

            # Set up logging
            if self.print_logging:
                print(f"Epoch {self.current_epoch}")
                print("  Training.")
            self.train_metrics = Metrics(self.classes, len(train_dataset))
            self.valid_metrics = Metrics(self.classes, len(valid_dataset))

            # train one epoch
            epoch_train_scores, epoch_train_targets = self.train_epoch(
                self.current_epoch
            )

            # evaluate on validation set
            epoch_val_scores, epoch_val_targets = self.evaluate_epoch(
                self.current_epoch
            )

            # Save weights at every logging interval and at the last epoch
            if (self.current_epoch % log_every == 0) or (
                self.current_epoch == epochs - 1
            ):
                self.save_epoch_results()

        print("Training complete.")
        return

    def train_epoch(self, epoch):
        # put model in train mode
        self.net.train()

        epoch_train_scores = []
        epoch_train_targets = []

        # iterate through the training files, [batchsize] images at a time
        for t in self.train_loader:
            X, y = t["X"], t["y"]
            X = X.to(self.device)
            y = y.to(self.device)
            # remove the second dimension (1) of size=1
            # (eg, [10,1]->[10])
            targets = y.squeeze(1)

            # NOTE: labels are now one-hot encoded. If loss fn expects
            # numeric labels, need to re-shape
            # in this case, there should only be 1 column in the label df,
            # for the binary classification, so it should remain the same

            # Forward pass
            # (use the input images to generate output values)
            outputs = self.net(X)

            # Backward pass
            # (Learn from batch by updating the network weights)
            loss = self.loss_fn(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics with loss & class predictions for batch
            batch_scores = outputs.clone().detach()
            batch_predictions = batch_scores.argmax(dim=1)

            # TODO: fix metrics
            # self.train_metrics.accumulate_batch_metrics(
            #     loss.clone().detach().item(),
            #     targets.cpu().clone().detach().numpy(),
            #     batch_predictions.cpu().clone().detach().numpy(),
            # )

            # Save copy of scores and targets (labels)
            if self.save_scores:
                epoch_train_scores.extend([sample.numpy() for sample in batch_scores])
                epoch_train_targets.extend(
                    *y.clone().detach().reshape([1, len(y)]).numpy().tolist()
                )

        return epoch_train_scores, epoch_train_targets

    def evaluate_epoch(self, epoch):
        # Run predictions on a held-out validation set and measure accuracy
        if self.print_logging:
            print("  Validating.")

        self.net.eval()
        epoch_val_scores = []
        epoch_val_targets = []
        with torch.no_grad():
            # iterate through validation set, [batch_size] images at a time
            for t in self.valid_loader:
                X, y = t["X"], t["y"]
                X = X.to(self.device)
                y = y.to(self.device)
                targets = y.squeeze(1)

                # Run model
                outputs = self.net(X)

                # Update metrics with class predictions for batch
                batch_scores = outputs.clone().detach()
                batch_predictions = batch_scores.argmax(dim=1)
                # Loss isn't important here
                # TODO: fix
                # self.valid_metrics.accumulate_batch_metrics(
                #     0.0, targets.cpu(), batch_predictions.cpu()
                # )

                # Save copy of scores and true targets
                if self.save_scores:
                    epoch_val_scores.extend([sample.numpy() for sample in batch_scores])
                    epoch_val_targets.extend(
                        *y.clone().detach().reshape([1, len(y)]).numpy().tolist()
                    )

            return epoch_val_scores, epoch_val_targets

    def save_epoch_results(self):
        # save model weights along with accuracy metrics for the current epoch

        # TODO: fix metrics
        train_metrics_d = self.train_metrics.compute_epoch_metrics()
        valid_metrics_d = self.valid_metrics.compute_epoch_metrics()

        epoch_results = {
            "train_loss": train_metrics_d["loss"],
            "train_accuracy": train_metrics_d["accuracy"],
            "train_precision": train_metrics_d["precision"],
            "train_recall": train_metrics_d["recall"],
            "train_f1": train_metrics_d["f1"],
            "train_confusion_matrix": train_metrics_d["confusion_matrix"],
            "valid_accuracy": valid_metrics_d["accuracy"],
            "valid_precision": valid_metrics_d["precision"],
            "valid_recall": valid_metrics_d["recall"],
            "valid_f1": valid_metrics_d["f1"],
            "valid_confusion_matrix": valid_metrics_d["confusion_matrix"],
        }

        if self.print_logging:
            print("  Validation results:")
            for metric, result in epoch_results.items():
                print(f"    {metric}: {result}")

        epoch_results.update(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "labels_yaml": self.classes,  # self.labels_yaml,
            }
        )

        if self.save_scores:
            epoch_results.update(
                {
                    "train_scores": epoch_train_scores,
                    "train_targets": epoch_train_targets,
                    "valid_scores": epoch_val_scores,
                    "valid_targets": epoch_val_targets,
                }
            )

        if self.save_dir is not None:
            epoch_filename = f"{self.save_dir}/epoch-{self.current_epoch}.tar"
            torch.save(epoch_results, epoch_filename)
            if self.print_logging:
                print(f"  Saved results to {epoch_filename}.")

        def save(self, save_dir, name=None):
            """save model weights to .tar file

            Args:
                save_dir: path to save into
                name: model filename
                    (if None, name is epoch-{self.current_epoch})

            Effects:
                saves model with weights and labels to a .tar file
            """
            model_dictionary = {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "labels_yaml": self.classes,  # self.labels_yaml,
            }
            if name is None:
                name = f"epoch-{self.current_epoch}"
            filename = f"{save_dir}/{name}.tar"
            torch.save(model_dictionary, filename)
