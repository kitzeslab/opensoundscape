"""Utilties for .ml"""

import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_grad_cam
import tqdm

from opensoundscape.ml.sampling import ClassAwareSampler


def cas_dataloader(dataset, batch_size, num_workers):
    """
    Return a dataloader that uses the class aware sampler

    Class aware sampler tries to balance the examples per class in each batch.
    It selects just a few classes to be present in each batch, then samples
    those classes for even representation in the batch.

    Args:
        dataset: a pytorch dataset type object
        batch_size: see DataLoader
        num_workers: see DataLoader
    """

    if len(dataset) == 0:
        return None

    print("** USING CAS SAMPLER!! **")

    # check that the data is single-target
    max_labels_per_file = max(dataset.df.values.sum(1))
    min_labels_per_file = min(dataset.df.values.sum(1))
    assert (
        max_labels_per_file <= 1
    ), "Class Aware Sampler for multi-target labels is not implemented. Use single-target labels."
    assert (
        min_labels_per_file > 0
    ), "Class Aware Sampler requires that every sample have a label. Some samples had 0 labels."

    # we need to convert one-hot labels to digit labels for the CAS
    # first class name -> 0, next class name -> 1, etc
    digit_labels = dataset.df.values.argmax(1)

    # create the class aware sampler object and DataLoader
    sampler = ClassAwareSampler(digit_labels, num_samples_cls=2)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # don't shuffle bc CAS does its own sampling
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return loader


def get_batch(array, batch_size, batch_number):
    """get a single slice of a larger array

    using the batch size and batch index, from zero

    Args:
        array: iterable to split into batches
        batch_size: num elements per batch
        batch_number: index of batch
    Returns:
        one batch (subset of array)

    Note: the final elements are returned as the last batch
    even if there are fewer than batch_size

    Example:
        if array=[1,2,3,4,5,6,7] then:

        - get_batch(array,3,0) returns [1,2,3]

        - get_batch(array,3,3) returns [7]
    """
    start_idx = batch_number * batch_size
    end_idx = min((batch_number + 1) * batch_size, len(array))
    return array[start_idx:end_idx]


def apply_activation_layer(x, activation_layer=None):
    """applies an activation layer to a set of scores

    Args:
        x: input values
        activation_layer:
            - None [default]: return original values
            - 'softmax': apply softmax activation
            - 'sigmoid': apply sigmoid activation
            - 'softmax_and_logit': apply softmax then logit transform
    Returns:
        values with activation layer applied
        Note: if x is None, returns None

    Note: casts x to float before applying softmax, since torch's
    softmax implementation doesn't support int or Long type

    """
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if activation_layer is None:  # scores [-inf,inf]
        pass
    elif activation_layer == "softmax":
        # "softmax" activation: preds across all classes sum to 1
        x = F.softmax(x.float(), dim=1)
    elif activation_layer == "sigmoid":
        # map [-inf,inf] to [0,1]
        x = torch.sigmoid(x)
    elif activation_layer == "softmax_and_logit":
        # softmax, then remap scores from [0,1] to [-inf,inf]
        try:
            x = torch.logit(F.softmax(x, dim=1))
        except NotImplementedError:
            # use cpu because mps aten::logit is not implemented yet
            warnings.warn("falling back to CPU for logit operation")
            original_device = x.device
            x = torch.logit(F.softmax(x, dim=1).cpu()).to(original_device)

    else:
        raise ValueError(f"invalid option for activation_layer: {activation_layer}")

    return x


def collate_audio_samples_to_tensors(batch):
    """
    takes a list of AudioSample objects, returns batched tensors

    use this collate function with DataLoader if you want to use AudioFileDataset (or AudioSplittingDataset)
    but want the traditional output of PyTorch Dataloaders (returns two tensors:
        the first is a tensor of the data with dim 0 as batch dimension,
        the second is a tensor of the labels with dim 0 as batch dimension)

    Args:
        batch: a list of AudioSample objects

    Returns:
        (Tensor of stacked AudioSample.data, Tensor of stacked AudioSample.label.values)

    Example usage:
    ```
        from opensoundscape import AudioFileDataset, SpectrogramPreprocessor

        preprocessor = SpectrogramPreprocessor(sample_duration=2,height=224,width=224)
        audio_dataset = AudioFileDataset(label_df,preprocessor)

        train_dataloader = DataLoader(
            audio_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn = collate_audio_samples_to_tensors
        )
    ```
    """
    tensors = torch.stack([i.data for i in batch])
    labels = torch.tensor([i.labels.tolist() for i in batch])
    return tensors, labels


def check_labels(label_df, classes):
    """check that classes and label_df.columns are the same, otherwise raise a helpful error"""
    class_err = """
            Train and validation datasets must have same classes
            and class order as model object. Consider using
            `train_df=train_df[cnn.classes]` or `cnn.change_classes(train_df.columns)` 
            before training.
            """
    assert list(classes) == list(label_df.columns), class_err

    assert (
        label_df.values.max(axis=None) <= 1 and label_df.values.min(axis=None) >= 0
    ), "Labels must in range [0,1], but found values outside range"
