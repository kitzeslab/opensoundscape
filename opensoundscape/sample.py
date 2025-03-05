"""Class for holding information on a single sample"""

import copy
from pathlib import Path
import torch
import numpy as np


class Sample:
    """Class for holding information on a single sample

    a Sample in OpenSoundscape contains information about a single
    sample, for instance its data and labels

    Subclass this class to create Samples of specific types
    """

    def __init__(self, data=None):
        self.data = data


class AudioSample(Sample):
    """A class containing information about a single audio sample

    self.preprocessing_exception is intialized as None and will contain
    the exception raised during preprocessing if any exception occurs

    """

    def __init__(
        self,
        source,
        start_time=None,
        duration=None,
        labels=None,
        trace=None,
    ):
        """initialize AudioSample
        Args:
            path: location of audio sample
            start_time=None: start time of sample in seconds from beginning of file
            duration=None: length in seconds of the sample
            labels=None: pd.Series containing class names and 0/1 labels
            trace=None: pd.Series matching a preprocessor.pipeline containing
                the renditions of the sample created by each preprocessing Action
        """
        super().__init__()
        self.source = source  # the initial value, generally an audio path
        self.start_time = start_time
        self.duration = duration
        self.labels = labels
        self.trace = trace
        self.preprocessing_exception = None

        # to begin with, set the data to source
        # the data will be updated, so make a copy
        # to avoid mutating the original object
        self.data = copy.deepcopy(self.source)

    def __repr__(self):
        return (
            f"AudioSample(source={self.source}, start_time={self.start_time},"
            f"end_time={self.end_time}, labels={self.labels})"
        )

    @property
    def categorical_labels(self):
        """list of indices with value==1 in self.labels"""
        if self.labels is None:
            return None
        else:
            return list(self.labels[self.labels == 1].index)

    @classmethod
    def from_series(
        cls,
        labels_series,
        rounding_precision=10,
        audio_root=None,
    ):
        """initialize AudioSample from a pandas Series (optionally containing labels)

        - if series name (dataframe index) is tuple, extracts ['file','start_time','end_time']
        these values to (source, start_time, duration=end_time-start_time)
        - otherwise, series name extracted as source; start_time and duration will be none

        Extracts source (file), start_time, and end_time from multi-index pd.Series (one row
        of a pd.DataFrame with multi index ['file','start_time','end_time']).
        The argument `series` is saved as self.labels. If sparse, converts to dense.
        Creates an AudioSample object.

        Args:
            labels_series: a pd.Series with name = file path or ['file','start_time','end_time']
                and index as classes with 0/1 values as labels. Labels can have no values
                (just a name) if sample does not have labels.
            rounding_precision: rounds duration to this many decimals
                to avoid floating point precision errors. Pass `None` for no rounding.
                Default: 10 decimal places
            audio_root: optionally pass a root directory (pathlib.Path or str) to prepended to each
            file path
                - if None (default), value of `file` must be full path
        """
        # cast (potentially sparse input) to dense boolean #TODO: should it be int or long, or float?
        # note that this implementation doesn't allow soft labels
        # make a copy to avoid modifying original
        labels_series = labels_series.copy().astype(bool)

        if type(labels_series.name) == tuple:
            # if the dataframe has a multi-index, it should be (file,start_time,end_time)
            assert (
                len(labels_series.name) == 3
            ), "series.name must be ('file','start_time','end_time') or a single value 'file'"
            sample_path, start_time, end_time = labels_series.name
            sample_path = Path(sample_path)
        else:
            # Series.name (dataframe index) contains a path to a file
            # No clip times are provided, so the entire file will be loaded
            sample_path = Path(labels_series.name)
            start_time = None
            end_time = None

        if audio_root is not None:
            # check that audio_root argument is valid
            msg = f"audio_root must be str, Path, or None. Got {type(audio_root)}"
            assert isinstance(audio_root, (str, Path)), msg

            # prepend the root directory to the given file path
            sample_path = Path(audio_root) / sample_path

        # calculate duration if start, end given
        if end_time is not None and start_time is not None:
            duration = end_time - start_time
            # avoid annoying floating point precision errors, round
            if rounding_precision is not None:
                duration = round(duration, rounding_precision)
        else:
            duration = None

        # instantiate (create the object)
        return cls(
            source=str(sample_path),
            start_time=start_time,
            duration=duration,
            labels=labels_series.copy(),
            trace=None,
        )

    @property
    def end_time(self):
        "calculate sample end time as start_time + duration"
        if self.duration is None:
            return None
        else:
            if self.start_time is None:
                return self.duration
            else:
                return self.start_time + self.duration


# TODO: move this to dataloaders.py? or preprocessing.utils?
def collate_audio_samples_to_dict(samples):
    """
    generate batched tensors of data and labels (in a dictionary).
    returns collated samples: a dictionary with keys "samples" and "labels"

    assumes that s.data is a Tensor and s.labels is a list/array
    for each sample S, and that every sample has labels for the same classes.

    Args:

        samples: iterable of AudioSample objects (or other objects
        with attributes .data as Tensor and .labels as list/array)

    Returns:
        dictionary of {
            "samples":batched tensor of samples,
            "labels": batched tensor of labels,
        }
    """
    return {
        "samples": torch.stack([s.data for s in samples]),
        "labels": torch.Tensor(np.vstack([s.labels.values for s in samples])),
    }


def collate_audio_samples(samples):
    """
    generate batched tensors of data and labels from list of AudioSample

    assumes that s.data is a Tensor and s.labels is a list/array
    for each item in samples, and that every sample has labels for the same classes.

    Args:
        samples: iterable of AudioSample objects (or other objects
            with attributes .data as Tensor and .labels as list/array)

    Returns:
        (samples, labels) tensors of shape (batch_size, *) & (batch_size, n_classes)
    """
    return (
        torch.stack([s.data for s in samples]),
        torch.Tensor(np.vstack([s.labels.values for s in samples])),
    )
