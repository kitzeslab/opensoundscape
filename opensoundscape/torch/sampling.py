"""classes for strategically sampling within a DataLoader"""
import random
import numpy as np
from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data
import torchvision
import numpy as np

# Imbalanced Dataset Sampling by davinnovation
# (https://github.com/ufoym/imbalanced-dataset-sampler)
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Args:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self, dataset, indices=None, num_samples=None, callback_get_label=None
    ):
        if max(np.sum(dataset.df.values, 1)) > 1:
            raise ValueError(
                "ImbalancedDatasetSampler does not support multi-target labels"
            )

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return np.argmax(dataset.df.iloc[idx].values)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
## this implimentation by zhmiao
##################################
class RandomCycleIter:
    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    """In each batch of samples, pick a limited number of classes to include and
    give even representation to each class"""

    def __init__(self, labels, num_samples_cls=1):
        num_classes = len(np.unique(labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(
            self.class_iter, self.data_iter_list, self.num_samples, self.num_samples_cls
        )

    def __len__(self):
        return self.num_samples


def get_sampler():
    return ClassAwareSampler


##################################
