#!/usr/bin/env python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss = 0.0
        self.accuracy = 0.0
        self.precision = np.array([0.0 for _ in range(num_classes)])
        self.recall = np.array([0.0 for _ in range(num_classes)])
        self.f1 = np.array([0.0 for _ in range(num_classes)])

    def update_loss(self, loss):
        self.loss += loss

    def update_metrics(self, targets, predictions):
        self.accuracy += accuracy_score(targets, predictions)
        prec_rec_f1 = precision_recall_fscore_support(
            targets, predictions, labels=list(range(self.num_classes))
        )

        self.precision += prec_rec_f1[0]
        self.recall += prec_rec_f1[1]
        self.f1 += prec_rec_f1[2]

    def compute_metrics(self, loader_size):
        return (
            self.loss / loader_size,
            self.accuracy / loader_size,
            self.precision / loader_size,
            self.recall / loader_size,
            self.f1 / loader_size,
        )
