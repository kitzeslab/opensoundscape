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


def get_md5_digest(input_string):
    """Generate MD5 sum for a string

    Args:
        input_string: An input string

    Returns:
        output: A string containing the md5 hash of input string
    """
    obj = md5()
    obj.update(input_string.encode("utf-8"))
    return obj.hexdigest()


def annotations_with_overlaps_with_clip(df, begin, end):
    """Determine if any rows overlap with current segment

    Args:
        df:     A dataframe containing a Raven annotation file
        begin:  The begin time of the current segment (unit: seconds)
        end:    The end time of the current segment (unit: seconds)

    Returns:
        sub_df: A dataframe of annotations which overlap with the begin/end times
    """
    return df[
        ((df["begin time (s)"] >= begin) & (df["begin time (s)"] < end))
        | ((df["end time (s)"] > begin) & (df["end time (s)"] <= end))
    ]


class SplitterDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for splitting a WAV files

    Segments will be written to the `output_directory`

    Args:
        wavs:                   A list of WAV files to split
        annotations:            Should we search for corresponding annotations files? (default: False)
        label_corrections:      Specify a correction labels CSV file w/ column headers "raw" and "corrected" (default: None)
        overlap:                How much overlap should there be between samples (units: seconds, default: 1)
        duration:               How long should each segment be? (units: seconds, default: 5)
        output_directory        Where should segments be written? (default: segments/)
        include_last_segment:   Do you want to include the last segment? (default: False)
        column_separator:       What character should we use to separate columns (default: "\t")
        species_separator:      What character should we use to separate species (default: "|")

    Returns:
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
