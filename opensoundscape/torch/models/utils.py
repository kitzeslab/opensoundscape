import torch
import torch.nn as nn
import torch.nn.functional as F
from opensoundscape.torch.sampling import ClassAwareSampler, ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import pandas as pd
import numpy as np


class BaseModule(nn.Module):
    """
    Base class for a pytorch model pipeline class.

    All child classes should define load, save, etc
    """

    name = None

    def __init__(self):
        super(BaseModule, self).__init__()

    def setup_net(self):
        pass

    def setup_critera(self):
        pass

    def load(self, init_path):
        pass

    def save(self, out_path):
        pass

    def update_best(self):
        pass


def get_dataloader(
    safe_dataset, batch_size=64, num_workers=1, shuffle=False, sampler=""
):
    """
    Create a DataLoader from a DataSet
    - chooses between normal pytorch DataLoader and ImbalancedDatasetSampler.
    - Sampler: None -> default DataLoader; 'imbalanced'->ImbalancedDatasetSampler

    """
    if len(safe_dataset) == 0:
        return None

    if sampler == "imbalanced":
        loader = DataLoader(
            safe_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=ImbalancedDatasetSampler(safe_dataset.dataset),
        )
    # could implement other sampler options here
    else:  # just use a regular Pytorch DataLoader
        loader = DataLoader(
            safe_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader


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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # don't shuffle bc CAS does its own sampling
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return loader


def collate_lists_of_audio_clips(batch):
    """
    Collate function for splitting + prediction of long audio files

    Puts each data field into a tensor with outer dimension batch size

    Additionally, concats the dfs from each audio file into one long df for the entire batch
    """

    # first, we remove all None elements: these failed to preprocess
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return {"X": torch.Tensor(), "y": torch.Tensor(), "df": None}

    has_labels = "y" in batch[0].keys()
    if has_labels:
        labels = [d["y"] for d in batch]

    data = [d["X"] for d in batch]

    dfs = [d["df"] for d in batch]

    elem = data[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in data])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

        if has_labels:
            return {
                "X": torch.cat(data, 0, out=out),
                "y": torch.stack(labels),
                "df": pd.concat(dfs),
            }
        else:
            return {"X": torch.cat(data, 0, out=out), "df": pd.concat(dfs)}

    raise TypeError(f"not a tensor. Got {elem}")


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

    """
    if activation_layer == None:  # scores [-inf,inf]
        pass
    elif activation_layer == "softmax":
        # "softmax" activation: preds across all classes sum to 1
        x = softmax(x, 1)
    elif activation_layer == "sigmoid":
        # map [-inf,inf] to [0,1]
        x = torch.sigmoid(x)
    elif activation_layer == "softmax_and_logit":
        # softmax, then remap scores from [0,1] to [-inf,inf]
        x = torch.logit(softmax(x, 1))
    else:
        raise ValueError(f"invalid option for activation_layer: {activation_layer}")

    return x


def tensor_binary_predictions(scores, mode, threshold=None):
    """generate binary 0/1 predictions from continuous scores

    Args:
        scores: torch.Tensor of dim (batch_size, n_classes) with input scores [-inf:inf]
        mode: 'single_target', 'multi_target', or None (return empty tensor)
        threshold: minimum score to predict 1, if mode=='multi_target'. threshold
        can be a single value for all classes or a list of class-specific values.
    Returns:
        torch.Tensor of 0/1 predictions in same shape as scores

    Note: expects real-valued (unbounded) input scores, i.e. scores take
    values in [-inf, inf]. Sigmoid layer is applied before multi-target
    prediction, so the threshold should be in [0,1].
    """
    if mode == "single_target":
        # predict highest scoring class only
        preds = F.one_hot(scores.argmax(1), len(scores[0]))
    elif mode == "multi_target":
        if threshold is None:
            raise ValueError(f"threshold must be specified for multi_target prediction")
        # predict 0 or 1 based on a fixed threshold
        elif type(threshold) in [float, np.float32, np.float64, int]:
            preds = torch.sigmoid(scores) >= threshold
        elif type(threshold) in [np.array, list, torch.Tensor, tuple]:
            if len(threshold) == 1 or len(threshold) == len(scores[0]):
                # will make predictions for either a single threshold value or list of class-specific threshold values
                preds = torch.sigmoid(scores) >= torch.tensor(threshold)
            else:
                raise ValueError(
                    f"threshold must be a single value, or have the same number of values as there are classes"
                )

    elif mode is None:
        preds = torch.Tensor([])
    else:
        raise ValueError(
            f"invalid option for mode: {mode}. "
            "Expected 'single_target', 'multi_target', or None."
        )

    return preds
