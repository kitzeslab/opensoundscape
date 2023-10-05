import torch
import numpy as np
import pandas as pd
import warnings

from opensoundscape.utils import identity
from opensoundscape.ml.safe_dataset import SafeDataset
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset


class SafeAudioDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        samples,
        preprocessor,
        split_files_into_clips=True,
        overlap_fraction=0,
        final_clip=None,
        bypass_augmentations=True,
        raise_errors=False,
        collate_fn=identity,
        **kwargs,
    ):
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
            preprocessor: preprocessor object, eg AudioPreprocessor or SpectrogramPreprocessor
            split_files_into_clips=True: use AudioSplittingDataset to automatically split
                audio files into appropriate-lengthed clips
            overlap_fraction: overlap fraction between consecutive clips, ignroed if
                split_files_into_clips is False [default: 0]
            final_clip: how to handle the final incomplete clip in a file
                options:['extend','remainder','full',None] [default: None]
                see opensoundscape.utils.generate_clip_times_df for details
            bypass_augmentations: if True, don't apply any augmentations [default: True]
            raise_errors: if True, raise errors during preprocessing [default: False]
            collate_fn: function to collate samples into batches [default: identity]
            **kwargs: any arguments to torch.utils.data.DataLoader

        Returns:
            DataLoader that returns lists of AudioSample objects when iterated
            (if collate_fn is identity)
        """
        assert type(samples) in (list, np.ndarray, pd.DataFrame), (
            "`samples` must be either: "
            "(a) list or np.array of files, or DataFrame with (b) file as Index or "
            "(c) (file,start_time,end_time) as MultiIndex"
        )

        # set up prediction Dataset, considering three possible cases:
        # (c1) user provided multi-index df with file,start_time,end_time of clips
        # (c2) user provided file list and wants clips to be split out automatically
        # (c3) split_files_into_clips=False -> one sample & one prediction per file provided
        if (
            type(samples) == pd.DataFrame
            and type(samples.index) == pd.core.indexes.multi.MultiIndex
        ):  # c1 user provided multi-index df with file,start_time,end_time of clips
            dataset = AudioFileDataset(samples=samples, preprocessor=preprocessor)
        elif split_files_into_clips:  # c2 user provided file list; split into
            dataset = AudioSplittingDataset(
                samples=samples,
                preprocessor=preprocessor,
                overlap_fraction=overlap_fraction,
                final_clip=final_clip,
            )
        else:  # c3 split_files_into_clips=False -> one sample & one prediction per file provided
            dataset = AudioFileDataset(samples=samples, preprocessor=preprocessor)
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
        super(SafeAudioDataloader, self).__init__(
            dataset=safe_dataset,
            collate_fn=collate_fn,
            **kwargs,
        )

        # add any paths that failed to generate a clip df to _invalid_samples
        self.dataset._invalid_samples = self.dataset._invalid_samples.union(
            dataset.invalid_samples
        )
