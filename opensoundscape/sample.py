"""Class for holding information on a single sample"""
import copy


class Sample:
    """Class for holding information on a single sample

    a Sample in OpenSoundscape contains information about a single
    sample, for instance its data and labels

    Subclass this class to create Samples of specific types
    """

    def __init__(self):
        pass


class AudioSample(Sample):
    """A class containing information about a single audio sample

    self.preprocessing_exception is intialized as None and will contain
    the exception raised during preprocessing if any exception occurs

    """

    def __init__(
        self,
        path,
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
        self.path = path  # the initial value
        self.start_time = start_time
        self.duration = duration
        self.labels = labels
        self.trace = trace
        self.preprocessing_exception = None

        # to begin with, set the data to the path
        # the data will be updated, so make a copy
        # to avoid mutating the original object
        self.data = copy.deepcopy(self.path)


@property
def end_time(self):
    "calculate sample end time as start_time + duration"
    return self.start_time + self.duration
