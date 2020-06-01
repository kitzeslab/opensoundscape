#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from math import ceil, floor
from hashlib import md5
from librosa.output import write_wav
from librosa.core import load, get_duration
from sys import stderr
from Pathlib import Path


def get_segment(clip_begin, clip_end, samples, sr):
    """ Extract a segment from the samples

    Inputs:
        clip_begin:         Beginning of the clip (units: seconds)
        clip_end:           End of the clip (units: seconds)
        samples:            The samples to extract the clip from
        sr:                 The sample rate for the given samples

    Outputs:
        segment_samples:    The segment extracted from the samples
        begin:              The index for clip_begin from samples
        end:                The index for clip_end from samples
    """
    begin = floor(clip_begin * sr)
    end = ceil(clip_end * sr)
    return samples[begin:end], begin, end


def get_md5_digest(input_string):
    """ Generate MD5 sum for a string

    Inputs:
        input_string: An input string

    Outputs:
        output: A string containing the md5 hash of input string
    """
    with md5() as obj:
        obj.update(input_string.encode("utf-8"))
        return obj.hexdigest()


def annotations_with_overlaps_with_clip(df, begin, end):
    """ Determine if any rows overlap with current segment

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



# TODO:
# - Should be able to process WAV or MP3 files?
# - Should use the opensoundscape.audio.Audio object, need to input and forward audio configuration
class Splitter(torch.utils.data.Dataset):
    """ A PyTorch Dataset for splitting a WAV files

    Inputs:
        wavs:               A list of WAV files to split
        annotations:        Should we search for corresponding annotations files? (default: False)
        labels:             Specify a correction labels CSV file w/ column headers "raw" and "corrected" (default: None)
        overlap:            How much overlap should there be between samples (units: seconds, default: 1)
        duration:           How long should each segment be? (units: seconds, default: 5)
        output_directory    Where should segments be written? (default: segments/)

    Effects:
        - Segments will be written to the output_directory

    Outputs:
        output: A list of CSV rows containing the source audio, segment begin
                time (seconds), segment end time (seconds), segment audio, and present
                classes separated by '|' if annotations were requested
    """
    def __init__(
        self,
        wavs,
        annotations=False,
        labels=None,
        overlap=1,
        duration=5,
        output_directory="segments",
    ):
        self.wavs = list(wavs)

        self.annotations = annotations
        self.labels = labels
        if self.labels:
            self.labels_df = pd.read_csv(labels)

        self.overlap = overlap
        self.duration = duration
        self.output_directory = output_directory

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
                sys.stderr.write(f"Warning: Found no Raven annotations for {wav}\n")
                return {"data": []}

        wav_samples, wav_sample_rate = load(wav)
        wav_duration = get_duration(wav_samples, sr=wav_sample_rate)
        wav_times = np.arange(0.0, wav_duration, wav_duration / len(wav_samples))

        if self.annotations:
            annotation_df = pd.read_csv(annotation_file, sep="\t").sort_values(
                by=["begin time (s)"]
            )

        if self.labels:
            annotation_df["class"] = annotation_df["class"].fillna("unknown")
            annotation_df["class"] = annotation_df["class"].apply(
                lambda cls: self.labels_df[self.labels_df["raw"] == cls]["corrected"].values[
                    0
                ]
            )

        num_segments = ceil(
            (wav_duration - self.overlap) / (self.duration - self.overlap)
        )

        outputs = []
        for idx in range(num_segments):
            if idx == num_segments - 1:
                end = wav_duration
                begin = end - self.duration
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
                    segment_samples, segment_sample_begin, segment_sample_end = get_segment(
                        begin, end, wav_samples, wav_sample_rate
                    )
                    write_wav(f"{destination}.WAV", segment_samples, wav_sample_rate)

                    if idx == num_segments - 1:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}.WAV"
                    else:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}.WAV"
                    to_append += f",{'|'.join(overlaps['class'].unique())}"

                    outputs.append(to_append)
            else:
                segment_samples, segment_sample_begin, segment_sample_end = get_segment(
                    begin, end, wav_samples, wav_sample_rate
                )
                write_wav(f"{destination}.WAV", segment_samples, wav_sample_rate)

                if idx == num_segments - 1:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}.WAV"
                else:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}.WAV"

                outputs.append(to_append)

        return {"data": outputs}

    @classmethod
    def collate_fn(batch):
        return chain.from_iterable([x["data"] for x in batch])
