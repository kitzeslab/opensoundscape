import torch.nn as nn
from opensoundscape.torch.sampling import ClassAwareSampler, ImbalancedDatasetSampler
from torch.utils.data import DataLoader


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


def get_dataloader(dataset, batch_size=64, num_workers=1, shuffle=False, sampler=""):
    """
    Create a DataLoader from a DataSet
    - chooses between normal pytorch DataLoader and ImbalancedDatasetSampler.
    - Sampler: None -> default DataLoader; 'imbalanced'->ImbalancedDatasetSampler

    """
    if len(dataset) == 0:
        return None

    if sampler == "imbalanced":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=ImbalancedDatasetSampler(dataset),
        )
    # could implement other sampler options here
    else:  # just use a regular Pytorch DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    # will class aware sampler still work?

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
