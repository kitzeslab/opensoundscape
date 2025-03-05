import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from opensoundscape.utils import identity, _check_is_path
from opensoundscape.ml.safe_dataset import SafeDataset
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset
from opensoundscape.annotations import CategoricalLabels


class SafeAudioDataloader(torch.utils.data.DataLoader):
    """Create DataLoader for inference, wrapping a SafeDataset

    SafeDataset contains AudioFileDataset or AudioSampleDataset depending on sample type

    During inference, we allow the user to pass any of 3 things to samples:
    - list of file paths
    - Dataframe with file as index
    - Dataframe with file, start_time, end_time of clips as index

    If file as index, default split_files_into_clips=True means that it will
    automatically determine the number of clips that can be created from the file
    (with overlap between subsequent clips based on overlap_fraction)

    Args:
        samples: any of the following:
            - list of file paths
            - Dataframe with file as index
            - Dataframe with file, start_time, end_time of clips as index
            - CategoricalLabels object
        preprocessor: preprocessor object, eg AudioPreprocessor or SpectrogramPreprocessor
        split_files_into_clips=True: use AudioSplittingDataset to automatically split
            audio files into appropriate-lengthed clips
        clip_overlap_fraction, clip_overlap, clip_step, final_clip:
            see `opensoundscape.utils.generate_clip_times_df`
        overlap_fraction: deprecated alias for clip_overlap_fraction
        bypass_augmentations: if True, don't apply any augmentations [default: True]
        raise_errors: if True, raise errors during preprocessing [default: False]
        collate_fn: function to collate list of AudioSample objects into batches
            [default: idenitty] returns list of AudioSample objects,
            use collate_fn=opensoundscape.sample.collate_audio_samples to return
            a tuple of (data, labels) tensors
        audio_root: optionally pass a root directory (pathlib.Path or str)
            - `audio_root` is prepended to each file path
            - if None (default), samples must contain full paths to files
        **kwargs: any arguments to torch.utils.data.DataLoader

    Returns:
        DataLoader that returns lists of AudioSample objects when iterated
        (if collate_fn is identity)
    """

    def __init__(
        self,
        samples,
        preprocessor,
        split_files_into_clips=True,
        clip_overlap=None,
        clip_overlap_fraction=None,
        clip_step=None,
        overlap_fraction=None,
        final_clip=None,
        bypass_augmentations=True,
        raise_errors=False,
        collate_fn=identity,
        audio_root=None,
        **kwargs,
        # TODO: persistent_workers=True?
    ):

        assert type(samples) in (list, np.ndarray, pd.DataFrame, CategoricalLabels), (
            "`samples` must be either: "
            "(a) list or np.array of files, or DataFrame with (b) file as Index, "
            "(c) (file,start_time,end_time) as MultiIndex, or "
            "(d) CategoricalLabels object"
        )

        if isinstance(samples, CategoricalLabels):
            # extract sparse multihot label df
            samples = samples.mutihot_df_sparse

        # setting these attributes seems to be necessary when using Lightning,
        # even though we don't need them as attributes in the DataLoader
        # this could be confusing because user should not modify dl.preprocessor,
        # it is used to initialize self.dataset only
        self.samples = samples
        """do not override or modify this attribute, as it will have no effect"""
        self.preprocessor = preprocessor
        """do not override or modify this attribute, as it will have no effect"""

        # remove `dataset` kwarg possibly passed from Lightning
        kwargs.pop("dataset", None)

        if overlap_fraction is not None:
            warnings.warn(
                "`overlap_fraction` argument is deprecated and will be removed in a future version. Use `clip_overlap_fraction` instead.",
                DeprecationWarning,
            )
            assert (
                clip_overlap_fraction is None
            ), "Cannot specify both overlap_fraction and clip_overlap_fraction"
            clip_overlap_fraction = overlap_fraction

        # validate that file paths are correctly placed in the input index or list
        # and check that the first one exist as a way to quickly catch mistakes
        if len(samples) > 0:
            if isinstance(samples, pd.DataFrame):  # samples is a pd.DataFrame
                if isinstance(samples.index, pd.core.indexes.multi.MultiIndex):
                    # index is (file, start_time, end_time)
                    first_path = samples.index.values[0][0]
                else:  # index of df is just file path
                    first_path = samples.index.values[0]
            else:  # samples is a list of file path
                first_path = samples[0]
            if audio_root is not None:
                first_path = Path(first_path) / Path(audio_root)
            _check_is_path(first_path)

        # set up prediction Dataset, considering three possible cases:
        # (c1) user provided multi-index df with file,start_time,end_time of clips
        # (c2) user provided file list and wants clips to be split out automatically
        # (c3) split_files_into_clips=False -> one sample & one prediction per file provided
        if (
            type(samples) == pd.DataFrame
            and type(samples.index) == pd.core.indexes.multi.MultiIndex
        ):  # c1 user provided multi-index df with file,start_time,end_time of clips
            # raise AssertionError if first item of multi-index is not str or Path
            dataset = AudioFileDataset(
                samples=samples, preprocessor=preprocessor, audio_root=audio_root
            )
        elif (
            split_files_into_clips
        ):  # c2 user provided file list; split each file into appropriate length clips
            # raise AssertionError if first item is not str or Path
            dataset = AudioSplittingDataset(
                samples=samples,
                preprocessor=preprocessor,
                clip_overlap=clip_overlap,
                clip_overlap_fraction=clip_overlap_fraction,
                clip_step=clip_step,
                final_clip=final_clip,
                audio_root=audio_root,
            )
        else:  # c3 samples is list of files and
            # split_files_into_clips=False -> one sample & one prediction per file provided
            # eg, each file is a 5 second clips and the model expects 5 second clips
            dataset = AudioFileDataset(
                samples=samples, preprocessor=preprocessor, audio_root=audio_root
            )

        dataset.bypass_augmentations = bypass_augmentations

        if len(dataset) < 1:
            warnings.warn(
                "prediction_dataset has zero samples. No predictions will be generated."
            )

        # If unsafe_behavior= "substitute", a SafeDataset will not fail on bad files,
        # but will provide a different sample! Later we go back and replace scores
        # with np.nan for the bad samples (using safe_dataset._invalid_indices)
        # this approach to error handling feels hacky
        # however, returning None would break the batching of samples
        # "raise" behavior will raise exceptions
        invalid_sample_behavior = "raise" if raise_errors else "substitute"

        safe_dataset = SafeDataset(
            dataset, invalid_sample_behavior=invalid_sample_behavior
        )

        # initialize the pytorch.utils.data.DataLoader
        super().__init__(
            dataset=safe_dataset,
            collate_fn=collate_fn,
            **kwargs,
        )

        # add any paths that failed to generate a clip df to _invalid_samples
        self.dataset._invalid_samples = self.dataset._invalid_samples.union(
            dataset.invalid_samples
        )
