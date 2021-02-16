import torch.nn as nn
from opensoundscape.torch.class_aware_sampler import ClassAwareSampler
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


def get_dataloader(
    dataset, batch_size=64, num_workers=1, shuffle=False, cas_sampler=False
):
    """
    Dataset loader
    """
    if len(dataset) == 0:
        return None

    if cas_sampler:
        print("** USING CAS SAMPLER!! **")
        # note: I didn't implement dataset.digit_labels, not sure what it is or
        # if we need it
        sampler = ClassAwareSampler(dataset.digit_labels, 2)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader


def cas_dataloader(
    dataset, batch_size=64, num_workers=1
):  # shuffle=False, don't use shuffle for cas?
    """
    Return a dataloader that uses the class aware sampler

    Class aware sampler tries to balance the examples per class in each batch,
    and selects just a few classes to be present in each batch. It then samples
    those classes for even representation in the batch.

    Args:
        dataset: a pytorch dataset type object
        batch_size: see DataLoader
        num_workers: see DataLoader
    """

    if len(dataset) == 0:
        return None

    print("** USING CAS SAMPLER!! **")
    sampler = ClassAwareSampler(dataset.digit_labels, 2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return loader
