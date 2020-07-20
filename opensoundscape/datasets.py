#!/usr/bin/env python3
import pandas as pd
import numpy as np
from math import ceil, floor
from hashlib import md5
from sys import stderr
from pathlib import Path
from itertools import chain
import torch
from torchvision import transforms
from PIL import Image

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram


def get_md5_digest(input_string):
    """ Generate MD5 sum for a string

    Inputs:
        input_string: An input string

    Outputs:
        output: A string containing the md5 hash of input string
    """
    obj = md5()
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


class Splitter(torch.utils.data.Dataset):
    """ A PyTorch Dataset for splitting a WAV files

    Inputs:
        wavs:                   A list of WAV files to split
        annotations:            Should we search for corresponding annotations files? (default: False)
        label_corrections:      Specify a correction labels CSV file w/ column headers "raw" and "corrected" (default: None)
        overlap:                How much overlap should there be between samples (units: seconds, default: 1)
        duration:               How long should each segment be? (units: seconds, default: 5)
        output_directory        Where should segments be written? (default: segments/)
        include_last_segment:   Do you want to include the last segment? (default: False)

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
        label_corrections=None,
        overlap=1,
        duration=5,
        output_directory="segments",
        include_last_segment=False,
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

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, item_idx):
        wav = self.wavs[item_idx]
        suffix = wav.suffix
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
                    audio_to_write.save(f"{destination}{suffix}")

                    if idx == num_segments - 1:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}{suffix}"
                    else:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}{suffix}"
                    to_append += f",{'|'.join(overlaps['class'].unique())}"

                    outputs.append(to_append)
            else:
                segment_sample_begin = audio_obj.time_to_sample(begin)
                segment_sample_end = audio_obj.time_to_sample(end)
                audio_to_write = audio_obj.trim(begin, end)
                audio_to_write.save(f"{destination}{suffix}")

                if idx == num_segments - 1:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}{suffix}"
                else:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}{suffix}"

                outputs.append(to_append)

        return {"data": outputs}

    @classmethod
    def collate_fn(*batch):
        return chain.from_iterable([x["data"] for x in batch[1]])


class BinaryFromAudio(torch.utils.data.Dataset):
    """ Binary Audio -> Image Dataset

    Given a DataFrame with audio files in one of the columns, generate
    a DataSet for basic machine learning tasks

    Input:
        df: A DataFrame with a column containing audio files
        audio_column: The column in the DataFrame which contains audio files [default: Destination]
        label_column: The column with numeric labels [default: NumericLabels]
        height: Height for resulting Tensor [default: 224]
        width: Width for resulting Tensor [default: 224]
        add_noise: Apply RandomAffine and ColorJitter filters [default: False]
        debug: Save images to a directory [default: None]
        spec_augment: If True, prepare audio for spec_augment procedure [default: False]
        random_trim_length: Extract a clip of this many seconds of audio starting at a random time
            If None, the original clip will be used [default: None]
        overlay_prob: Probability of an image from a different class being overlayed (combined as a weighted sum)
        on the training image. typical values: 0, 0.66 [default: 0] 

    Output:
        Dictionary:
            { "X": (1, H, W) if spec_augment else (3, H, W)
            , "y": (1)
            }
    """

    def __init__(
        self,
        df,
        audio_column="Destination",
        label_column="NumericLabels",
        height=224,
        width=224,
        add_noise=False,
        debug=None,
        spec_augment=False,
        random_trim_length=None,
        overlay_prob = 0
    ):
        self.df = df
        self.audio_column = audio_column
        self.label_column = label_column
        self.height = height
        self.width = width
        self.debug = debug
        self.spec_augment = spec_augment
        self.random_trim_length = random_trim_length
        self.overlay_prob = overlay_prob

        if add_noise:
            self.transform = transforms.Compose(
                [transforms.Resize((self.height, self.width)), transforms.ToTensor()]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.height, self.width)),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.2, 0.03), fillcolor=50
                    ),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0
                    ),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item_idx):
        row = self.df.iloc[item_idx]

        audio_p = Path(row[self.audio_column])
        audio = Audio.from_file(audio_p)
        spectrogram = Spectrogram.from_audio(audio)
        
        #TODO: test this
        if self.random_trim_length is not None:
            audio_length = len(audio.samples)/audio.sample_rate
            if self.random_trim_length < audio_length:
                raise ValueError('the length of the original file ({audio_length}) was less than the length to extract ({self.random_trim_length}) for the file {file_path}')
            extra_time = audio_length - self.random_trim_length
            start_time = np.random.uniform()*extra_time
            spectrogram = spectrogram.trim(start_time,start_time+random_trim_length)
        
                
        image = Image.fromarray(spectrogram.spectrogram)
        image = image.convert("RGB")
        
        #TODO: test this
        if self.overlay_prob > np.random.uniform():
            # select a random training file from a different class
            other_classes_df = self.df[self.df[self.label_column]!=row[self.label_column]]
            file_path = np.random.choice(other_classes_df[self.audio_column].values())
            overlay_audio = Audio.from_file(audio_p)
            overlay_spectrogram = Spectrogram.from_audio(overlay_audio)
            
            # trim to desired length if needed
            if self.random_trim_length is not None:
                audio_length = len(overlay_audio.samples)/overlay_audio.sample_rate
                if self.random_trim_length < audio_length:
                    raise ValueError('the length of the original file ({audio_length}) was less than the length to extract ({self.random_trim_length}) for the file {file_path}')
                extra_time = audio_length - self.random_trim_length
                start_time = np.random.uniform()*extra_time
                overlay_spectrogram = overlay_spectrogram.trim(start_time,start_time+random_trim_length)
                
            # create an image and add blur
            overlay_image = Image.fromarray(overlay_spectrogram.spectrogram)
            blur_r = np.random.randint(0, 8) / 10
            overlay_image = overlay.filter(ImageFilter.GaussianBlur(radius=blur_r))

            # use a weighted sum to overlay the images
            overlap_weight = random.randint(5, 10) / 10
            
            image = image * (1-overlap_weight) + overlay_image * overlap_weight

        if self.debug:
            image.save(f"{self.debug}/{audio_p.stem}.png")

        labels = np.array([row[self.label_column]])

        X = self.transform(image)

        if self.spec_augment:
            X = X[0].unsqueeze(0)
        return {"X": X, "y": torch.from_numpy(labels)}
