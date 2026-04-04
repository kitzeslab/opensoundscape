"""Preprocessors: pd.Series child with an action sequence & forward method"""

import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from opensoundscape.annotations import CategoricalLabels
from opensoundscape.utils import make_clip_df
from opensoundscape.sample import AudioSample
from opensoundscape.vector_database import _find_matching_window_ids
from opensoundscape.annotations import CategoricalLabels


class InvalidIndexError(Exception):
    pass


class NoMatchingWindowIDsError(Exception):
    pass


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
        bypass_augmentations:
            if True, skips Actions with .is_augmentation=True
        audio_root:
            optionally pass a root directory (pathlib.Path or str) to prepend to each file path
            - if None (default), samples must contain full paths to files
        **kwargs:

    Returns:
        sample (AudioSample object)

    Raises:
        PreprocessingError if exception is raised during __getitem__

    Effects:
        self.invalid_samples will contain a set of paths that did not successfully
            produce a list of clips with start/end times, if split_files_into_clips=True
    """

    def __init__(
        self,
        samples,
        preprocessor,
        bypass_augmentations=False,
        audio_root=None,
        **kwargs,
    ):
        super().__init__()

        ## Input Validation ##

        # check that audio_root argument is valid
        msg = f"audio_root must be str, Path, or None. Got {type(audio_root)}"
        assert isinstance(audio_root, (str, Path, type(None))), msg

        # ingest various formats for samples
        samples, invalid_samples = _ingest_samples_argument(
            samples=samples,
            audio_root=audio_root,
            sample_duration=preprocessor.sample_duration,
        )
        _check_first_path(samples)
        _check_label_types(samples)
        if len(samples) == 0:
            warnings.warn("Zero samples!")

        self.classes = samples.columns
        """list of classes to which multi-hot labels correspond"""

        self.label_df = samples
        """dataframe containing file paths, clip times, and multi-hot labels (one column per class)"""

        self.preprocessor = preprocessor
        """Preprocessor object containing a .pipeline of ordered preprocessing operations"""

        self.invalid_samples = set(invalid_samples)
        """set of file paths that raised exceptions during preprocessing"""

        self.audio_root = audio_root
        """path to prepend to all audio file paths when loading"""

        self.bypass_augmentations = bypass_augmentations
        """if True, skips Actions with .is_augmentation=True"""

    @classmethod
    def from_categorical_df(
        cls, categorical_labels, preprocessor, class_list, bypass_augmentations=False
    ):
        """Create AudioFileDataset from a DataFrame with a column listing categorical labels

        e.g. where df['labels'] = [['a','b'], [], ['a','c']]

        Args:
            categorical_labels: DataFrame with index (file) or (file, start_time, end_time) and 'label'
                column containing lists of labels or integers corresponding to class names
            preprocessor: Preprocessor object
            bypass_augmentations: if True, skip augmentations with .is_augmentation=True

        Returns:
            AudioFileDataset object
        """
        from opensoundscape.annotations import (
            categorical_to_multi_hot,
            multi_hot_to_categorical,
        )

        multihot_labels_sp = categorical_to_multi_hot(
            categorical_labels["labels"], class_list, sparse=True
        )
        sparse_df = pd.DataFrame.sparse.from_spmatrix(
            multihot_labels_sp,
            index=categorical_labels.index,
            columns=categorical_labels.columns,
        )

        return cls(
            samples=sparse_df,
            preprocessor=preprocessor,
            bypass_augmentations=bypass_augmentations,
        )

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx, break_on_key=None, break_on_type=None):
        if not isinstance(idx, int):
            raise InvalidIndexError(
                f"idx must be an integer, got {type(idx)}. "
                f"This could happen if you specified a custom sampler that results in returning "
                "lists of indices rather than a single index. AudioFileDataset.__getitem__ "
                "requires that idx is a single integer index."
            )
        sample = AudioSample.from_series(
            self.label_df.iloc[idx], audio_root=self.audio_root
        )

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


class EmbeddingDataset(torch.utils.data.Dataset):
    """simple dataset wrapper for embedding features and labels

    Args:
        features: tensor or np.array of input features
            first dimension should be samples
        labels: tensor or np.array of target labels
            first dimension should be samples
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class HopliteDataset(torch.utils.data.Dataset):
    """Dataset that retrieves embeddings from a HopliteDB for given files and start times"""

    def __init__(
        self,
        db,
        samples,
        selection_mode="first",
        deployment_id=None,
        deployment_name=None,
        project=None,
        audio_root=None,
        clip_duration=None,
        **kwargs,
    ):
        """Initialize the HopliteDataset

        Assumes that the Hoplite DB already contains windows for every audio clip

        #TODO: to go fast, do we need to do batched retrieval?

        Args:
            db: HopliteDB instance to retrieve embeddings from
            samples: defines audio clips, any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with (file, start_time, end_time) as MultiIndex
                - Dataframe with (file, start_time, end_time) as columns
                - Dataframe with (file, start_time) as columns
                - Dataframe with (file) as column
                - CategoricalLabels object
            selection_mode: 'first' or 'random', how to select an embedding if multiple matches are found
            deployment_id, deployment_name, project: optional filters for retrieving windows from the db
            audio_root: if provided, audio paths are given and stored as paths relative to this root directory
            clip_duration: length of clips in seconds; only used to determine clip times when file list is provided
            **kwargs: passed to make_clip_df if samples need to be split into clips
        """
        label_df, invalid_samples = _ingest_samples_argument(
            samples, audio_root=audio_root, clip_duration=clip_duration, **kwargs
        )
        self.db = db
        self.selection_mode = selection_mode
        self.deployment_id = deployment_id
        self.deployment_name = deployment_name
        self.project = project
        # setting label_df also extracts files and start_times and initializes window_ids for caching retrieved window IDs
        self.label_df = label_df  # avoid modifying input df in-place
        self.invalid_samples = invalid_samples

    @property
    def label_df(self):
        return self._label_df

    @label_df.setter
    def label_df(self, new_df):
        self._label_df = new_df
        self.files = new_df.index.get_level_values("file").astype(str).to_numpy()
        self.start_times = (
            new_df.index.get_level_values("start_time").to_numpy().astype(np.float16)
        )

        # list of tuples: matching window IDs for each sample in label_df
        print("Finding matching window IDs for samples in label_df...")
        self.window_ids = _find_matching_window_ids(
            self.db,
            self._label_df,  # clip dataframe
            deployment_id=self.deployment_id,
            deployment_name=self.deployment_name,
            project=self.project,
        )
        print("Finished finding matching window IDs.")

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        window_ids = self.window_ids[idx]
        if len(window_ids) == 0:
            raise NoMatchingWindowIDsError(
                f"No matching window IDs found for sample at index {idx} (file {self.files[idx]}, start_time {self.start_times[idx]})"
            )

        if self.selection_mode == "first":
            id = window_ids[0]
        elif self.selection_mode == "random":
            id = np.random.choice(window_ids)
        else:
            raise ValueError(
                f"Invalid selection_mode {self.selection_mode}. Must be 'first' or 'random'."
            )
        emb = self.db.get_embedding(id)

        # return (embedding, labels)
        return emb, self.label_df.iloc[idx].values


def _ingest_samples_argument(samples, audio_root=None, clip_duration=None, **kwargs):
    """create clip df with MultiIndex (file,start_time,end_time)

    Args:
        samples: input samples, can be any of:
        - list of file paths
        - Dataframe with file as index
        - Dataframe with (file, start_time, end_time) of clips as MultiIndex
        - Dataframe with (file, start_time, end_time) as columns
        - Dataframe with (file, start_time) as column
        - Dataframe with (file) as column
        - CategoricalLabels object
        audio_root: if provided, audio paths are given and stored as paths relative to this root directory
        clip_duration: length of clips in seconds; only used to determine clip times when file list is provided
        **kwargs: passed to make_clip_df if samples need to be split into clips

    """
    invalid_samples = set()
    if isinstance(samples, (str, Path)):
        samples = [samples]
    elif isinstance(samples, CategoricalLabels):
        # extract sparse multihot label df
        samples = samples.mutihot_df_sparse

    # validate type of samples: list or np array of files, or df
    assert type(samples) in (
        list,
        np.ndarray,
        pd.DataFrame,
    ), (
        f"samples must be type list/np.ndarray of file paths, "
        f"or pd.DataFrame with index containing path (or multi-index of "
        f"path, start_time, end_time). Got {type(samples)}."
    )

    if type(samples) == list or type(samples) == np.ndarray:
        assert (
            clip_duration is not None
        ), "clip_duration must be provided when samples is a list of file paths"
        # create clip df
        # self.label_df will have multi-index (file,start_time,end_time)
        # can contain rows with start/end time np.nan for failed samples
        samples, invalid_samples = make_clip_df(
            files=samples,
            clip_duration=clip_duration,
            return_invalid_samples=True,
            audio_root=audio_root,
            **kwargs,
        )
    elif type(samples) == pd.DataFrame:
        samples = samples.copy()  # avoid modifying original df object
        keys = ["file", "start_time", "end_time"]
        # 1. already has file, start_time, end_time in index: do nothing
        if isinstance(samples.index, pd.MultiIndex):
            pass
        # 2: columns for file, start_time, end_time -> just set index
        elif all(col in samples.columns for col in keys):
            samples = samples.set_index(keys)
        else:
            # one row per file, "file" is either a column or the index
            assert (
                clip_duration is not None
            ), "clip_duration must be provided when samples is a df without clip start/end times"

            if not "file" in samples.columns:
                # 3. File as index of df -> move to "file" column for [4]
                assert isinstance(samples.index.values[0], (str, Path))
                samples.index.name = "file"
                samples = samples.reset_index()

            # 4. Samples dataframe has a column for "file"
            # First, make a clip df
            clip_df, invalid_samples = make_clip_df(
                files=samples["file"],
                clip_duration=clip_duration,
                return_invalid_samples=True,
                audio_root=audio_root,
                **kwargs,
            ).reset_index()
            # Second, copy labels from original sample df to each row of the clip_df
            # corresponding to the same file
            clip_df = clip_df.merge(
                samples,
                on="file",
                how="left",
            ).set_index(keys)
    else:
        raise ValueError(f"Unsupported type for samples: {type(samples)}")

    return samples, invalid_samples


def _check_first_path(samples, audio_root=None):
    """check that first item in samples is a valid file path"""
    if len(samples) > 0:
        first_path, start, end = samples.index[0]
        if audio_root is not None:
            first_path = Path(audio_root) / first_path
        assert isinstance(
            first_path, (str, Path)
        ), f"Expected str or Path, got {type(first_path)}"
        assert Path(first_path).exists(), f"First file {first_path} was not found."


def _check_label_types(samples):
    if (
        len(samples) > 0
        and len(samples.columns) > 0
        and not samples.values[0, 0] in (0, 1, True, False, None, np.nan)
    ):
        warnings.warn(
            "If `samples` has labels, they are expected to be one of: (0, 1, True, False, None, np.nan). First value is "
            f"{samples.values[0, 0]}."
        )
