import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from opensoundscape.ml.safe_dataset import SafeDataset
from opensoundscape.ml.datasets import AudioFileDataset, AudioFileDataset
from opensoundscape.annotations import CategoricalLabels
from opensoundscape.utils import identity


class SafeAudioDataloader(torch.utils.data.DataLoader):
    """Create DataLoader for inference, wrapping a SafeDataset

    SafeDataset contains AudioFileDataset or AudioSampleDataset depending on sample type

    During inference, we allow the user to pass any of these formatas for `samples`:
    - list of file paths
    - Dataframe with file as index
    - Dataframe with (file, start_time, end_time) of clips as MultiIndex
    - Dataframe with (file, start_time, end_time) as columns
    - Dataframe with (file, start_time) as column
    - Dataframe with (file) as column
    - CategoricalLabels object

    If start_times are not specified, default split_files_into_clips=True means that it will
    automatically determine the number of clips that can be created from the file
    (with overlap between subsequent clips based on overlap_fraction)

    Args:
        samples: any of the following:
            - list of file paths
            - Dataframe with file, start_time, end_time of clips as index
            - Dataframe with (file, start_time, end_time) as columns
            - Dataframe with (file, start_time) as columns
            - Dataframe with file as index
            - Dataframe with (file) as column
            - CategoricalLabels object
        preprocessor: preprocessor object, eg AudioPreprocessor or SpectrogramPreprocessor
        split_files_into_clips=True: use AudioFileDataset to automatically split
            audio files into appropriate-lengthed clips
        clip_overlap_fraction, clip_overlap, clip_step, final_clip:
            see `opensoundscape.utils.generate_clip_times_df`
        overlap_fraction: deprecated alias for clip_overlap_fraction
        bypass_augmentations: if True, don't apply any augmentations [default: True]
        invalid_sample_behavior: how to handle samples that fail to preprocess,
            one of "substitute", "placeholder", "raise", or "none"
            - "substitute": pick another sample
            - "placeholder": return a placeholder value (zeros) for the sample
            - "raise": raise the error
            - "none": return None
        collate_fn: function to collate list of AudioSample objects into batches
            if None (default), uses collate_fn=collate_audio_samples to return
                a tuple of (data, labels) tensors
            use opso.utils.identity to return list of AudioSample objects
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
        clip_overlap=None,
        clip_overlap_fraction=None,
        clip_step=None,
        final_clip="extend",
        bypass_augmentations=True,
        invalid_sample_behavior="placeholder",
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

        # if (file,start_time,end_time) are in the columns, convert to MultiIndex
        if isinstance(samples, pd.DataFrame):
            if all(
                col in samples.columns for col in ["file", "start_time", "end_time"]
            ):
                samples.set_index(["file", "start_time", "end_time"], inplace=True)
        # if (file)

        if isinstance(samples, CategoricalLabels):
            # extract sparse multihot label df
            samples = samples.mutihot_df_sparse

        if collate_fn is None:
            collate_fn = collate_audio_samples

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

        dataset = AudioFileDataset(
            samples=samples,
            preprocessor=preprocessor,
            clip_overlap=clip_overlap,
            clip_overlap_fraction=clip_overlap_fraction,
            clip_step=clip_step,
            final_clip=final_clip,
            audio_root=audio_root,
        )

        dataset.bypass_augmentations = bypass_augmentations

        if len(dataset) < 1:
            warnings.warn(
                "prediction_dataset has zero samples. No predictions will be generated."
            )

        # If unsafe_behavior= "substitute", a SafeDataset will not fail on bad files,
        # but will provide a different sample! This is useful in training
        # "placeholder" returns a zero-valued sample for samples that fail to preprocess,
        # but we still need to replace these with NaN later since the output doesn't correspond to the sample

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
        # self.dataset._invalid_samples = self.dataset._invalid_samples.union(
        #     dataset.invalid_samples
        # )
        # TODO _invalid_samples no longer an attribute - is this happening internally?


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
