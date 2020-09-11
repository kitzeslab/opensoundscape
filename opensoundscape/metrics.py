#!/usr/bin/env python
from sklearn.metrics import confusion_matrix
import numpy as np


class Metrics:
    """ Basic Example

    See opensoundscape.torch.train for an in-depth example

    ```
    dataset = Dataset(...)
    dataloader = DataLoader(dataset, ...)
    classes = [0, 1, 2, 3, 4] # An example list of classes
    for epoch in epochs:
        metrics = Metrics(classes, len(dataset))
        for batch in dataloader:
            X, y = batch["X"], batch["y"]
            targets = y.squeeze(0) # dim: (batch_size)
            ...
            loss = ... # dim: (0)
            predictions = ... # dim: (batch_size)
            metrics.accumulate_batch_metrics(
                loss.item(),
                targets.cpu(),
                predictions.cpu()
            )
        metrics_dictionary = metrics.compute_epoch_metrics()
    ```
    """

    def __init__(self, classes, dataset_len):
        """ Use confusion matrix to compute metrics during learning

        For each batch in an epoch, compute the confusion matrix using sklearn
        and accumulate confusion matrices over an epoch.

        Args:
            classes:        A list of classes, e.g. [0, 1, 2, 3]
                            - classes must match targets and predictions fed to
                              Metrics.accumulate_batch_metrics
            dataset_len:    To compute loss, Metrics needs to know the length of the dataset
        """
        self.classes = classes
        self.num_classes = len(classes)
        self.dataset_len = dataset_len
        self.loss = 0.0
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def accumulate_batch_metrics(self, loss, targets, predictions):
        """ For a batch, accumulate loss and confusion matrix

        For validation pass 0 for loss.

        Args:
            loss:           The loss for this batch
            targets:        The correct y labels
            predictions:    The predicted labels
        """
        # len(targets) returns the first dimension of a pytorch tensor i.e. the batch size
        self.loss += loss * len(targets)
        self.confusion_matrix += confusion_matrix(
            targets, predictions, labels=self.classes
        )

    def compute_epoch_metrics(self):
        """ Compute metrics from learning

        Computes the loss and accuracy, precision, recall, and f1 scores from
        the confusion matrix and returns dictionary with metric name as keys
        and their corresponding values

        Returns:
            dictionary with keys:
                [loss, accuracy, precision, recall, f1, confusion_matrix]
        """
        loss = self.loss / self.dataset_len
        accuracies = [None] * self.num_classes
        precisions = [None] * self.num_classes
        recalls = [None] * self.num_classes
        f1s = [None] * self.num_classes

        total_observations = self.confusion_matrix.sum()
        for idx, cls in enumerate(self.classes):
            class_observations = self.confusion_matrix[idx, :].sum()
            class_predictions = self.confusion_matrix[:, idx].sum()

            true_positives = self.confusion_matrix[idx, idx]
            # row + col observations double subtract true_positives
            true_negatives = (
                total_observations
                - class_observations
                - class_predictions
                + true_positives
            )
            false_positives = class_predictions - true_positives
            false_negatives = class_observations - true_positives

            accuracies[idx] = float(true_positives + true_negatives) / (
                true_positives + true_negatives + false_positives + false_negatives
            )
            precisions[idx] = float(true_positives) / (true_positives + false_positives)
            recalls[idx] = float(true_positives) / (true_positives + false_negatives)
            f1s[idx] = float(2 * true_positives) / (
                2 * true_positives + false_positives + false_negatives
            )

        return {
            "loss": loss,
            "accuracy": accuracies,
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
            "confusion_matrix": self.confusion_matrix,
        }
