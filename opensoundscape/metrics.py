#!/usr/bin/env python
from sklearn.metrics import (
    jaccard_score,
    hamming_loss,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# from scipy.sparse import csr_matrix
import numpy as np


def binary_predictions(scores, single_target=False, threshold=0.5):  # TODO rename
    """convert numeric scores to binary predictions

    return 0/1 for an array of scores: samples (rows) x classes (columns)

    Args:
        scores:
            a 2-d list or np.array. row=sample, columns=classes
        single_target:
            if True, predict 1 for highest scoring class per sample,
            0 for other classes. If False, predict 1 for all scores > threshold
            [default: False]
        threshold:
            Predict 1 for score > threshold. only used if single_target = False.
            [default: 0.5]
    """
    scores = np.array(scores)
    if single_target:  # predict highest scoring class only
        preds = np.zeros(np.shape(scores)).astype(int)
        for i, score_row in enumerate(scores):
            preds[i, np.argmax(score_row)] = 1
    else:
        preds = (scores >= threshold).astype(int)

    return preds


def multi_target_metrics(targets, scores, class_names, threshold):
    """generate various metrics for a set of scores and labels (targets)

    Args:
        targets: 0/1 lables in 2d array
        scores: continuous values in 2d array
        class_names: list of strings
        threshold: scores >= threshold result in prediction of 1,
            while scores < threshold result in prediction of 0

    Returns:
        metrics_dict: dictionary of various overall and per-class metrics
    """
    metrics_dict = {}

    preds = binary_predictions(scores, single_target=False, threshold=threshold)

    # Store per-class precision, recall, and f1
    class_pre, class_rec, class_f1, _ = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )

    for i, class_i in enumerate(class_names):
        metrics_dict.update(
            {
                class_i: {
                    "precision": class_pre[i],
                    "recall": class_rec[i],
                    "f1": class_f1[i],
                }
            }
        )

    # macro scores are averaged across classes
    metrics_dict["precision"] = class_pre.mean()
    metrics_dict["recall"] = class_rec.mean()
    metrics_dict["f1"] = class_f1.mean()

    try:
        metrics_dict["jaccard"] = jaccard_score(targets, preds, average="macro")
    except ValueError:
        metrics_dict["jaccard"] = np.nan
    try:
        metrics_dict["hamming_loss"] = hamming_loss(targets, preds)
    except ValueError:
        metrics_dict["hamming_loss"] = np.nan
    try:
        metrics_dict["map"] = average_precision_score(targets, preds, average="macro")
    except ValueError:
        metrics_dict["map"] = np.nan
    try:
        metrics_dict["au_roc"] = roc_auc_score(targets, preds, average="macro")
    except ValueError:
        metrics_dict["au_roc"] = np.nan

    return metrics_dict


def single_target_metrics(targets, scores, class_names):
    """generate various metrics for a set of scores and labels (targets)

    Predicts 1 for the highest scoring class per sample and 0 for
    all other classes.

    Args:
        targets: 0/1 lables in 2d array
        scores: continuous values in 2d array
        class_names: list of strings

    Returns:
        metrics_dict: dictionary of various overall and per-class metrics

    """
    if max(np.sum(targets, 1)) > 1:
        raise ValueError(
            "Labels were not single target! "
            "Use multi-target classifier if multiple classes can be present "
            "in a single sample."
        )

    metrics_dict = {}

    preds = binary_predictions(scores, single_target=True)

    # Confusion matrix requires numbered-class not one-hot labels
    t = np.argmax(targets, 1)
    p = np.argmax(preds, 1)
    metrics_dict["confusion_matrix"] = confusion_matrix(t, p)

    # precision, recall, and f1
    pre, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )
    metrics_dict.update({"precision": pre[1], "recall": rec[1], "f1": f1[1]})

    try:
        metrics_dict["jaccard"] = jaccard_score(targets, preds, average="macro")
    except ValueError:
        metrics_dict["jaccard"] = np.nan
    try:
        metrics_dict["hamming_loss"] = hamming_loss(targets, preds)
    except ValueError:
        metrics_dict["hamming_loss"] = np.nan
    try:
        metrics_dict["map"] = average_precision_score(targets, preds, average="macro")
    except ValueError:
        metrics_dict["map"] = np.nan
    try:
        metrics_dict["au_roc"] = roc_auc_score(targets, preds, average="macro")
    except ValueError:
        metrics_dict["au_roc"] = np.nan

    return metrics_dict
