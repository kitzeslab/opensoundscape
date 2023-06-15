"""Preprocessors: pd.Series child with an action sequence & forward method"""
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from opensoundscape.utils import make_clip_df
from opensoundscape.sample import AudioSample


class AudioFileDataset(torch.utils.data.Dataset):
    """Base class for audio datasets with OpenSoundscape (use in place of torch Dataset)

    Custom Dataset classes should subclass this class or its children.

    Datasets in OpenSoundscape contain a Preprocessor object which is
    responsible for the procedure of generating a sample for a given input.
    The DataLoader handles a dataframe of samples (and potentially labels) and
    uses a Preprocessor to generate samples from them.

    Args:
        samples:
            the files to generate predictions for. Can be:
            - a dataframe with index containing audio paths, OR
            - a dataframe with multi-index of (path,start_time,end_time) per clip, OR
            - a list or np.ndarray of audio file paths

            Notes for input dataframe:
             - df must have audio paths in the index.
             - If label_df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
             - If data does not have labels, label_df will have no columns
        preprocessor:
            an object of BasePreprocessor or its children which defines
            the operations to perform on input samples

    Returns:
        sample (AudioSample object)

    Raises:
        PreprocessingError if exception is raised during __getitem__

    Effects:
        self.invalid_samples will contain a set of paths that did not successfully
            produce a list of clips with start/end times, if split_files_into_clips=True
    """

    def __init__(self, samples, preprocessor, bypass_augmentations=False):
        ## Input Validation ##

        # validate type of samples: list, np array, or df
        assert type(samples) in (list, np.ndarray, pd.DataFrame,), (
            f"samples must be type list/np.ndarray of file paths, "
            f"or pd.DataFrame with index containing path (or multi-index of "
            f"path, start_time, end_time). Got {type(samples)}."
        )
        if type(samples) == list or type(samples) == np.ndarray:
            df = pd.DataFrame(index=samples)
        elif type(samples) == pd.DataFrame:
            # can either have index of file path or multi-index (file_path,start_time,end_time)
            df = samples
        # if the dataframe has a multi-index, it should be (file,start_time,end_time)
        self.has_clips = type(df.index) == pd.core.indexes.multi.MultiIndex
        if self.has_clips:
            assert list(df.index.names) == [
                "file",
                "start_time",
                "end_time",
            ], "multi-index must be ('file','start_time','end_time')"

        # give helpful warnings for incorret df, but don't raise Exception
        if len(df) > 0:
            first_path = df.index[0][0] if self.has_clips else df.index[0]
            if not Path(first_path).exists():
                warnings.warn(
                    "Index of dataframe passed to "
                    f"preprocessor must be a file path. First sample {df.index[0]} was not found."
                )
        elif len(df) > 0 and len(df.columns) > 0 and not df.values[0, 0] in (0, 1):
            warnings.warn("if label_df has labels, they must take values of 0 and 1")

        if len(df) == 0:
            warnings.warn("Zero samples!")

        self.classes = df.columns
        self.label_df = df
        self.preprocessor = preprocessor
        self.invalid_samples = set()

        # if True skips Actions with .is_augmentation=True
        self.bypass_augmentations = bypass_augmentations

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx, break_on_key=None, break_on_type=None):
        sample = AudioSample.from_series(self.label_df.iloc[idx])

        # preprocessor.forward will raise PreprocessingError if something fails
        sample = self.preprocessor.forward(
            sample,
            bypass_augmentations=self.bypass_augmentations,
            break_on_key=break_on_key,
            break_on_type=break_on_type,
        )

        return sample

    def __repr__(self):
        return f"{self.__class__} object with preprocessor: {self.preprocessor}"

    def class_counts(self):
        """count number of each label"""
        labels = self.label_df.columns
        counts = np.sum(self.label_df.values, 0)
        return labels, counts

    def sample(self, **kwargs):
        """out-of-place random sample

        creates copy of object with n rows randomly sampled from label_df

        Args: see pandas.DataFrame.sample()

        Returns:
            a new dataset object
        """
        new_ds = copy.deepcopy(self)
        new_ds.label_df = new_ds.label_df.sample(**kwargs)
        return new_ds

    def head(self, n=5):
        """out-of-place copy of first n samples

        performs df.head(n) on self.label_df

        Args:
            n: number of first samples to return, see pandas.DataFrame.head()
            [default: 5]

        Returns:
            a new dataset object
        """
        new_ds = copy.deepcopy(self)
        new_ds.label_df = new_ds.label_df.head(n)
        return new_ds


class AudioSplittingDataset(AudioFileDataset):
    """class to load clips of longer files rather than one sample per file

    Internally creates even-lengthed clips split from long audio files.

    If file labels are provided, applies copied labels to all clips from a file

    NOTE: If you've already created a dataframe with clip start and end times,
    you can use AudioFileDataset. This class is only necessary if you wish to
    automatically split longer files into clips (providing only the file paths).

    Args:
        see AudioFileDataset and make_clip_df
    """

    def __init__(self, samples, preprocessor, overlap_fraction=0, final_clip=None):
        super(AudioSplittingDataset, self).__init__(
            samples=samples, preprocessor=preprocessor
        )

        self.has_clips = True

        # create clip df
        # self.label_df will have multi-index (file,start_time,end_time)
        # can contain rows with start/end time np.nan for failed samples
        self.label_df, self.invalid_samples = make_clip_df(
            files=samples,
            clip_duration=preprocessor.sample_duration,
            clip_overlap=overlap_fraction * preprocessor.sample_duration,
            final_clip=final_clip,
            return_invalid_samples=True,
        )
