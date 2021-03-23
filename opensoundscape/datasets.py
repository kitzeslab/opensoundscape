#!/usr/bin/env python3
import pandas as pd
import numpy as np
from math import ceil
from hashlib import md5
from sys import stderr
from pathlib import Path
from itertools import chain
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from time import time

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
import opensoundscape.torch.tensor_augment as tensaug


def get_md5_digest(input_string):
    """Generate MD5 sum for a string

    Inputs:
        input_string: An input string

    Outputs:
        output: A string containing the md5 hash of input string
    """
    obj = md5()
    obj.update(input_string.encode("utf-8"))
    return obj.hexdigest()


def annotations_with_overlaps_with_clip(df, begin, end):
    """Determine if any rows overlap with current segment

    Inputs:
        df:     A dataframe containing a Raven annotation file
        begin:  The begin time of the current segment (unit: seconds)
        end:    The end time of the current segment (unit: seconds)

    Output:
        sub_df: A dataframe of annotations which overlap with the begin/end times
    """
    return df[
        ((df["begin time (s)"] >= begin) & (df["begin time (s)"] < end))
        | ((df["end time (s)"] > begin) & (df["end time (s)"] <= end))
    ]


class SplitterDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for splitting a WAV files

    Inputs:
        wavs:                   A list of WAV files to split
        annotations:            Should we search for corresponding annotations files? (default: False)
        label_corrections:      Specify a correction labels CSV file w/ column headers "raw" and "corrected" (default: None)
        overlap:                How much overlap should there be between samples (units: seconds, default: 1)
        duration:               How long should each segment be? (units: seconds, default: 5)
        output_directory        Where should segments be written? (default: segments/)
        include_last_segment:   Do you want to include the last segment? (default: False)
        column_separator:       What character should we use to separate columns (default: "\t")
        species_separator:      What character should we use to separate species (default: "|")

    Effects:
        - Segments will be written to the `output_directory`

    Outputs:
        output: A list of CSV rows (separated by `column_separator`) containing
            the source audio, segment begin time (seconds), segment end time
            (seconds), segment audio, and present classes separated by
            `species_separator` if annotations were requested
    """

    def __init__(
        self,
        wavs,
        annotations=False,
        label_corrections=None,
        overlap=1,
        duration=5,
        output_directory="segments",
        include_last_segment=False,
        column_separator="\t",
        species_separator="|",
    ):
        self.wavs = list(wavs)

        self.annotations = annotations
        self.label_corrections = label_corrections
        if self.label_corrections:
            self.labels_df = pd.read_csv(label_corrections)

        self.overlap = overlap
        self.duration = duration
        self.output_directory = output_directory
        self.include_last_segment = include_last_segment
        self.column_separator = column_separator
        self.species_separator = species_separator

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, item_idx):
        wav = self.wavs[item_idx]
        annotation_prefix = self.wavs[item_idx].stem.split(".")[0]

        if self.annotations:
            annotation_file = Path(
                f"{wav.parent}/{annotation_prefix}.Table.1.selections.txt.lower"
            )
            if not annotation_file.is_file():
                stderr.write(f"Warning: Found no Raven annotations for {wav}\n")
                return {"data": []}

        audio_obj = Audio.from_file(wav)
        wav_duration = audio_obj.duration()
        wav_times = np.arange(0.0, wav_duration, wav_duration / len(audio_obj.samples))

        if self.annotations:
            annotation_df = pd.read_csv(annotation_file, sep="\t").sort_values(
                by=["begin time (s)"]
            )

        if self.label_corrections:
            annotation_df["class"] = annotation_df["class"].fillna("unknown")
            annotation_df["class"] = annotation_df["class"].apply(
                lambda cls: self.labels_df[self.labels_df["raw"] == cls][
                    "corrected"
                ].values[0]
            )

        num_segments = ceil(
            (wav_duration - self.overlap) / (self.duration - self.overlap)
        )

        outputs = []
        for idx in range(num_segments):
            if idx == num_segments - 1:
                if self.include_last_segment:
                    end = wav_duration
                    begin = end - self.duration
                else:
                    continue
            else:
                begin = self.duration * idx - self.overlap * idx
                end = begin + self.duration

            if self.annotations:
                overlaps = annotations_with_overlaps_with_clip(
                    annotation_df, begin, end
                )

            unique_string = f"{wav}-{begin}-{end}"
            destination = f"{self.output_directory}/{get_md5_digest(unique_string)}"

            if self.annotations:
                if overlaps.shape[0] > 0:
                    segment_sample_begin = audio_obj.time_to_sample(begin)
                    segment_sample_end = audio_obj.time_to_sample(end)
                    audio_to_write = audio_obj.trim(begin, end)
                    audio_to_write.save(f"{destination}.wav")

                    if idx == num_segments - 1:
                        to_append = [
                            wav,
                            annotation_file,
                            wav_times[segment_sample_begin],
                            wav_times[-1],
                            f"{destination}.wav",
                        ]
                    else:
                        to_append = [
                            wav,
                            annotation_file,
                            wav_times[segment_sample_begin],
                            wav_times[segment_sample_end],
                            f"{destination}.wav",
                        ]
                    to_append.append(
                        self.species_separator.join(overlaps["class"].unique())
                    )

                    outputs.append(
                        self.column_separator.join([str(x) for x in to_append])
                    )
            else:
                segment_sample_begin = audio_obj.time_to_sample(begin)
                segment_sample_end = audio_obj.time_to_sample(end)
                audio_to_write = audio_obj.trim(begin, end)
                audio_to_write.save(f"{destination}.wav")

                if idx == num_segments - 1:
                    to_append = [
                        wav,
                        wav_times[segment_sample_begin],
                        wav_times[-1],
                        f"{destination}.wav",
                    ]
                else:
                    to_append = [
                        wav,
                        wav_times[segment_sample_begin],
                        wav_times[segment_sample_end],
                        f"{destination}.wav",
                    ]

                outputs.append(self.column_separator.join([str(x) for x in to_append]))

        return {"data": outputs}

    @classmethod
    def collate_fn(*batch):
        return chain.from_iterable([x["data"] for x in batch[1]])


from opensoundscape import preprocess
from opensoundscape.preprocess import ParameterRequiredError


class BasePreprocessor(torch.utils.data.Dataset):
    def __init__(self, df, return_labels=True):

        self.df = df
        self.return_labels = return_labels
        self.labels = df.columns

        # actions: a collection of instances of BaseAction child classes
        self.actions = preprocess.ActionContainer()

        # pipeline: an ordered list of operations to conduct on each file,
        # each pulled from self.actions
        self.pipeline = []

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item_idx):

        df_row = self.df.iloc[item_idx]
        x = Path(df_row.name)  # the index contains a path to a file

        for pipeline_element in self.pipeline:
            if pipeline_element.bypass:
                continue
            try:
                x = pipeline_element.go(x)
            except ParameterRequiredError:  # need to pass labels
                x, df_row = pipeline_element.go(x, df_row)

        # Return sample & label pairs (training/validation)
        if self.return_labels:
            labels = torch.from_numpy(df_row.values)
            return {"X": x, "y": labels}

        # Return sample only (prediction)
        return {"X": x}

    def class_counts_cal(self):
        """count number of each label"""
        print("Warning: check if this is the correct behavior")
        labels = self.df.columns
        counts = np.sum(self.df.values, 0)
        return labels, counts


class AudioLoadingPreprocessor(BasePreprocessor):
    """creates Audio objects from file paths"""

    def __init__(self, df, return_labels=True):

        super(AudioLoadingPreprocessor, self).__init__(df, return_labels=return_labels)

        # add an AudioLoader to our (currently empty) action toolkit
        self.actions.load_audio = preprocess.AudioLoader()

        # add the action to our (currently empty) pipeline
        self.pipeline.append(self.actions.load_audio)


class AudioToImagePreprocessor(BasePreprocessor):
    """loads audio paths, performs various augmentations, returns tensor


    df must have audio paths as index and class names as columns
    - if no labels, df will have no columns
    """

    def __init__(self, df, audio_length=None, out_shape=[224, 224], return_labels=True):

        super(AudioToImagePreprocessor, self).__init__(df, return_labels=return_labels)

        self.audio_length = audio_length
        self.return_labels = return_labels

        # add each action to our tool kit, then to pipeline
        self.actions.load_audio = preprocess.AudioLoader()
        self.pipeline.append(self.actions.load_audio)

        self.actions.trim_audio = preprocess.AudioTrimmer()
        self.pipeline.append(self.actions.trim_audio)

        self.actions.to_spec = preprocess.AudioToSpectrogram()
        self.pipeline.append(self.actions.to_spec)

        self.actions.to_img = preprocess.SpecToImg(shape=out_shape)
        self.pipeline.append(self.actions.to_img)

        self.actions.to_tensor = preprocess.ImgToTensor()
        self.pipeline.append(self.actions.to_tensor)

        self.actions.normalize = preprocess.TensorNormalize()
        self.pipeline.append(self.actions.normalize)


class CnnPreprocessor(AudioToImagePreprocessor):
    """Child of AudioToImagePreprocessor with full augmentation pipeline"""

    def __init__(
        self,
        df,
        audio_length=None,
        return_labels=True,
        augmentation=True,
        debug=None,
        overlay_df=None,
        out_shape=[224, 224],
    ):

        super(CnnPreprocessor, self).__init__(
            df,
            audio_length=audio_length,
            out_shape=out_shape,
            return_labels=return_labels,
        )

        self.augmentation = augmentation
        self.debug = debug

        # add extra Actions
        if self.augmentation:
            self.actions.overlay = preprocess.ImgOverlay(
                overlay_df=overlay_df,
                audio_length=self.audio_length,
                overlay_prob=1,
                max_overlay_num=1,
                overlay_class=None,
                # might not update with changes?
                loader_pipeline=self.pipeline[0:4],
                update_labels=True,
            )

            self.actions.color_jitter = preprocess.TorchColorJitter()
            self.actions.random_affine = preprocess.TorchRandomAffine()
            self.actions.tensor_aug = preprocess.TensorAugment()
            self.actions.add_noise = preprocess.TensorAddNoise(std=0.005)

        # re-define the action sequence
        if self.augmentation:
            self.pipeline = [
                self.actions.load_audio,
                self.actions.trim_audio,
                self.actions.to_spec,
                self.actions.to_img,
                self.actions.overlay,
                self.actions.color_jitter,
                self.actions.random_affine,
                self.actions.to_tensor,
                self.actions.tensor_aug,
                self.actions.add_noise,
                self.actions.normalize,
            ]
        else:
            self.pipeline = [
                self.actions.load_audio,
                self.actions.trim_audio,
                self.actions.to_spec,
                self.actions.to_img,
                self.actions.to_tensor,
                self.actions.normalize,
            ]

        if self.debug is not None:
            self.actions.save_img = preprocess.SaveTensorToDisk(self.debug)
            self.pipeline.append(self.actions.save_img)


class ResnetMultilabelPreprocessor(BasePreprocessor):
    """loads audio paths, performs various augmentations, returns tensor"""

    def __init__(
        self,
        df,
        audio_length=None,
        return_labels=True,
        augmentation=True,
        debug=None,
        overlay_df=None,
    ):

        super(ResnetMultilabelPreprocessor, self).__init__(
            df, return_labels=return_labels
        )

        self.audio_length = audio_length
        self.augmentation = augmentation
        self.return_labels = return_labels
        self.debug = debug

        # add each action to our tool kit, then to pipeline
        self.actions.load_audio = preprocess.AudioLoader()
        self.pipeline.append(self.actions.load_audio)

        self.actions.trim_audio = preprocess.AudioTrimmer()
        self.pipeline.append(self.actions.trim_audio)

        self.actions.to_spec = preprocess.AudioToSpectrogram()
        self.pipeline.append(self.actions.to_spec)

        self.actions.to_img = preprocess.SpecToImg()
        self.pipeline.append(self.actions.to_img)

        # should make one without overlay, then subclass and add overlay
        if self.augmentation:
            self.actions.overlay = preprocess.ImgOverlay(
                overlay_df=overlay_df,
                audio_length=self.audio_length,
                overlay_prob=0.5,
                max_overlay_num=2,
                overlay_weight=[0.2, 0.5],
                # this pipeline might not update with changes to preprocessor?
                loader_pipeline=self.pipeline[0:4],
                update_labels=True,
            )
            self.pipeline.append(self.actions.overlay)

            # color jitter and affine can be applied to img or tensor
            # here, we choose to apply them to the PIL.Image
            self.actions.color_jitter = preprocess.TorchColorJitter()
            self.pipeline.append(self.actions.color_jitter)

            self.actions.random_affine = preprocess.TorchRandomAffine()
            self.pipeline.append(self.actions.random_affine)

        self.actions.to_tensor = preprocess.ImgToTensor()
        self.pipeline.append(self.actions.to_tensor)

        if self.augmentation:
            self.actions.tensor_aug = preprocess.TensorAugment()
            self.pipeline.append(self.actions.tensor_aug)

            self.actions.add_noise = preprocess.TensorAddNoise(std=1.0)
            self.pipeline.append(self.actions.add_noise)

        self.actions.normalize = preprocess.TensorNormalize()
        self.pipeline.append(self.actions.normalize)

        if self.debug is not None:
            self.actions.save_img = preprocess.SaveTensorToDisk(self.debug)
            self.pipeline.append(self.actions.save_img)
