"""Class for holding information on a single sample"""
import copy
from pathlib import Path
import torch


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
    def from_series(cls, labels_series):
        """initialize AudioSample from a pandas Series (optionally containing labels)

        - if series name (dataframe index) is tuple, extracts ['file','start_time','end_time']
        these values to (source, start_time, duration=end_time-start_time)
        - otherwise, series name extracted as source; start_time and duraiton will be none

        Extracts source (file), start_time, and end_time from multi-index pd.Series (one row
        of a pd.DataFrame with multi index ['file','start_time','end_time']).
        The argument `series` is saved as self.labels
        Creates an AudioSample object.

        Args:
            labels: a pd.Series with name = file path or ['file','start_time','end_time']
                and index as classes with 0/1 values as labels. Labels can have no values
                (just a name) if sample does not have labels.
        """
        if type(labels_series.name) == tuple:
            # if the dataframe has a multi-index, it should be (file,start_time,end_time)
            assert (
                len(labels_series.name) == 3
            ), "series.name must be ('file','start_time','end_time') or a single value 'file'"
            sample_path, start_time, end_time = labels_series.name
        else:
            # Series.name (dataframe index) contains a path to a file
            # No clip times are provided, so the entire file will be loaded
            sample_path = Path(labels_series.name)
            start_time = None
            end_time = None

        # calcualte duration if start, end given
        if end_time is not None and start_time is not None:
            duration = end_time - start_time
        else:
            duration = None

        # instantiate (create the object)
        return cls(
            source=sample_path,
            start_time=start_time,
            duration=duration,
            labels=labels_series,
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
    """generate batched tensors of data and labels (in a dictionary)

    returns collated samples: a dictionary with keys "samples" and "labels"

    assumes that s.data is a Tensor and s.labels is a list/array
    for each sample S

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
        "labels": torch.Tensor([s.labels for s in samples]),
    }
