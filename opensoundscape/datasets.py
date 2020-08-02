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
from PIL import Image, ImageFilter
from time import time

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


class SplitterDataset(torch.utils.data.Dataset):
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
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}.wav"
                    else:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}.wav"
                    to_append += f",{'|'.join(overlaps['class'].unique())}"

                    outputs.append(to_append)
            else:
                segment_sample_begin = audio_obj.time_to_sample(begin)
                segment_sample_end = audio_obj.time_to_sample(end)
                audio_to_write = audio_obj.trim(begin, end)
                audio_to_write.save(f"{destination}.wav")

                if idx == num_segments - 1:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}.wav"
                else:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}.wav"

                outputs.append(to_append)

        return {"data": outputs}

    @classmethod
    def collate_fn(*batch):
        return chain.from_iterable([x["data"] for x in batch[1]])


class SingleTargetAudioDataset(torch.utils.data.Dataset):
    """ Single Target Audio -> Image Dataset

    Given a DataFrame with audio files in one of the columns, generate
    a Dataset of spectrogram images for basic machine learning tasks.

    This class provides access to several types of augmentations that act on
    audio and images with the following arguments:
    - add_noise: for adding RandomAffine and ColorJitter noise to images
    - random_trim_length: for only using a short random clip extracted from the training data
    - max_overlay_num / overlay_prob / overlay_weight:
        controlling the maximum number of additional spectrograms to overlay,
        the probability of overlaying an individual spectrogram,
        and the weight for the weighted sum of the spectrograms

    Additional augmentations on tensors are available when calling `train()`
    from the module `opensoundscape.torch.train`.

    Input:
        df: A DataFrame with a column containing audio files
        label_dict: a dictionary mapping numeric labels to class names,
            - for example: {0:'American Robin',1:'Northern Cardinal'}
            - pass `None` if you wish to retain numeric labels
        filename_column: The column in the DataFrame which contains paths to data [default: Destination]
        from_audio: Whether the raw dataset is audio [default: True]
        label_column: The column with numeric labels if present [default: None]
        height: Height for resulting Tensor [default: 224]
        width: Width for resulting Tensor [default: 224]
        add_noise: Apply RandomAffine and ColorJitter filters [default: False]
        save_dir: Save images to a directory [default: None]
        random_trim_length: Extract a clip of this many seconds of audio starting at a random time
            If None, the original clip will be used [default: None]
        extend_short_clips: If a file to be overlaid or trimmed from is too short,
            extend it to the desired length by repeating it. [default: False]
        max_overlay_num: The maximum number of additional images to overlay, each with probability overlay_prob [default: 0]
        overlay_prob: Probability of an image from a different class being overlayed (combined as a weighted sum)
            on the training image. typical values: 0, 0.66 [default: 0.2]
        overlay_weight: The weight given to the overlaid image during augmentation.
            When 'random', will randomly select a different weight between 0.2 and 0.5 for each overlay
            When not 'random', should be a float between 0 and 1 [default: 'random']

    Output:
        Dictionary:
            { "X": (3, H, W)
            , "y": (1) if label_column != None
            }
    """

    def __init__(
        self,
        df,
        label_dict,
        filename_column="Destination",
        from_audio=True,
        label_column=None,
        height=224,
        width=224,
        add_noise=False,
        save_dir=None,
        random_trim_length=None,
        extend_short_clips=False,
        max_overlay_num=0,
        overlay_prob=0.2,
        overlay_weight="random",
    ):
        self.df = df
        self.filename_column = filename_column
        self.from_audio = from_audio
        self.label_column = label_column
        self.height = height
        self.width = width
        self.save_dir = save_dir
        self.random_trim_length = random_trim_length
        self.extend_short_clips = extend_short_clips
        self.max_overlay_num = max_overlay_num
        self.overlay_prob = overlay_prob
        if (overlay_weight != "random") and (not 0 < overlay_weight < 1):
            raise ValueError(
                f"overlay_weight not in 0<overlay_weight<1 (given overlay_weight: {overlay_weight})"
            )
        self.overlay_weight = overlay_weight
        self.transform = self.set_transform(add_noise=add_noise)
        self.label_dict = label_dict

    def set_transform(self, add_noise):
        # Warning: some transforms only act on first channel
        transform_list = [transforms.Resize((self.height, self.width))]
        if add_noise:
            transform_list.extend(
                [
                    transforms.RandomAffine(
                        degrees=0, translate=(0.2, 0.03), fillcolor=(50, 50, 50)
                    ),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0
                    ),
                ]
            )

        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def random_audio_trim(self, audio, audio_length, audio_path):
        audio_length = len(audio.samples) / audio.sample_rate
        if self.random_trim_length > audio_length:
            if not self.extend_short_clips:
                raise ValueError(
                    f"the length of the original file ({audio_length} sec) was less than the length to extract ({self.random_trim_length} sec) for the file {audio_path}. . To extend short clips, use extend_short_clips=True"
                )
            else:
                return audio.extend(self.random_trim_length)
        extra_time = audio_length - self.random_trim_length
        start_time = np.random.uniform() * extra_time
        return audio.trim(start_time, start_time + self.random_trim_length)

    def image_from_audio(self, audio, mode="RGB"):
        """ Create a PIL image from audio

        Inputs:
            audio: audio object
            mode: PIL image mode, e.g. "L" or "RGB" [default: RGB]
        """
        spectrogram = Spectrogram.from_audio(audio)
        return spectrogram.to_image(shape=(self.width, self.height), mode=mode)

    def overlay_random_image(
        self, original_image, original_length, original_class, original_path
    ):
        """ Overlay an image from another class

        Select a random file from a different class. Trim if necessary to the
        same length as the given image. Overlay the images on top of each other
        with a weight
        """
        # select a random file from a different class
        other_classes_df = self.df[self.df[self.label_column] != original_class]
        overlay_path = np.random.choice(other_classes_df[self.filename_column].values)
        overlay_audio = Audio.from_file(overlay_path)

        # trim to same length as main clip
        overlay_audio_length = len(overlay_audio.samples) / overlay_audio.sample_rate
        if overlay_audio_length < original_length and not self.extend_short_clips:
            raise ValueError(
                f"the length of the overlay file ({overlay_audio_length} sec) was less than the length of the file {original_path} ({original_length} sec). To extend short clips, use extend_short_clips=True"
            )
        elif overlay_audio_length != original_length:
            overlay_audio = self.random_audio_trim(
                overlay_audio, original_length, overlay_path
            )
        overlay_image = self.image_from_audio(overlay_audio, mode="L")

        # create an image and add blur
        blur_r = np.random.randint(0, 8) / 10
        overlay_image = overlay_image.filter(ImageFilter.GaussianBlur(radius=blur_r))

        # Select weight; <0.5 means more emphasis on original image
        if self.overlay_weight == "random":
            weight = np.random.randint(2, 5) / 10
        else:
            weight = self.overlay_weight

        # use a weighted sum to overlay (blend) the images
        return Image.blend(original_image, overlay_image, weight)

    def upsample(self):
        raise NotImplementedError("Upsampling is not implemented yet")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item_idx):

        row = self.df.iloc[item_idx]
        audio_path = Path(row[self.filename_column])
        audio = Audio.from_file(audio_path)

        # trim to desired length if needed
        # (if self.random_trim_length is specified, select a clip of that length at random from the original file)
        audio_length = len(audio.samples) / audio.sample_rate
        if self.random_trim_length is not None:
            audio = self.random_audio_trim(audio, audio_length, audio_path)
            audio_length = self.random_trim_length
        image = self.image_from_audio(audio, mode="L")

        # add a blended/overlayed image from another class directly on top
        for _ in range(self.max_overlay_num):
            if self.overlay_prob > np.random.uniform():
                image = self.overlay_random_image(
                    original_image=image,
                    original_length=audio_length,
                    original_class=row[self.label_column],
                    original_path=audio_path,
                )
            else:
                break

        if self.save_dir:
            image.save(f"{self.save_dir}/{audio_path.stem}_{time()}.png")

        # apply desired random transformations to image and convert to tensor
        image = image.convert("RGB")
        X = self.transform(image)

        # Return data : label pairs (training/validation)
        if self.label_column:
            labels = np.array([row[self.label_column]])
            return {"X": X, "y": torch.from_numpy(labels)}

        # Return data only (prediction)
        return {"X": X}
