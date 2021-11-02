"""functions and classes for manipulating annotations of audio

includes BoxedAnnotations class and utilities to combine or "diff" annotations,
etc.
"""
from opensoundscape.helpers import overlap, overlap_fraction, generate_clip_times_df
import pandas as pd
import numpy as np


class BoxedAnnotations:
    """ container for frequency-time annotations of audio

    for instance, those created in Raven software
    includes functionality to load annotations from Raven txt files,
    output one-hot labels for specific clip lengths or clip start/end times,
    apply corrections to annotations, and more.

    Contains some analogous functions to Audio and Spectrogram, such as
    trim() [limit time range] and bandpass() [limit frequency range]
    """

    def __init__(self, df, audio_file=None):  # ,corrections_dict=None):
        """
        create object directly from DataFrame of frequency-time annotations

        For loading annotations from Raven txt files, use from_raven_file()

        Args:
            df: DataFrame of frequency-time labels. Columns must include:
                - "annotation": string or numeric labels (can be None/nan)
                - "start_time": left bound, sec since beginning of audio
                - "end_time": right bound, sec since beginning of audio
                - "low_f": upper frequency bound (can be None/nan)
                - "high_f": upper frequency bound (can be None/nan)
            corrections_dict: optionally provide a dictionary mapping
                { wrong (accidental) label : corrected label }

                TODO: should it be applied on creation of the object?

            audio_file: optionally include the name or path of corresponding

        Returns:
            BoxedAnnotations object
        """
        for col in ["annotation", "start_time", "end_time", "low_f", "high_f"]:
            assert (
                col in df.columns
            ), 'df columns must include all of these: ["annotation","start_time","end_time","low_f","high_f"]'
        self.df = df
        # self.corrections_dict = corrections_dict
        self.audio_path = audio_file

    @classmethod
    def from_raven_file(
        cls,
        path,
        annotation_column,
        keep_extra_columns=True,
        audio_file=None,
        # corrections_dict=None,
    ):
        """load annotations from Raven txt file

        Args:
            keep_extra_columns: True (all), False (none), or iterable of specific columns to keep
            - always includes start_time, end_time, low_f, high_f, annotation
            corrections_dict: optionally include a dictionary of
        """
        df = pd.read_csv(path, delimiter="\t")
        df = df.rename(
            columns={
                annotation_column: "annotation",
                "Begin Time (s)": "start_time",
                "End Time (s)": "end_time",
                "Low Freq (Hz)": "low_f",
                "High Freq (Hz)": "high_f",
            }
        )

        # remove undesired columns
        standard_columns = ["start_time", "end_time", "low_f", "high_f", "annotation"]
        if hasattr(keep_extra_columns, "__iter__"):
            # keep the desired columns
            df = df[standard_columns + list(keep_extra_columns)]
        elif not keep_extra_columns:
            # only keep required columns
            df = df[standard_columns]
        else:
            # keep all columns
            pass

        return cls(
            df=df,
            audio_file=audio_file,
            # corrections_dict=corrections_dict,
        )

    def to_raven_file(self):
        """save annotations in Raven-style tab-separated text file"""
        raise NotImplementedError

    def to_csv(self):
        """save annotations in Raven-style tab-separated text file"""
        raise NotImplementedError

    def trim(self, start_t, end_t, edge_mode="trim"):
        """edge_mode: 'trim','keep','remove'"""
        raise NotImplementedError

    def bandpass(self, low_f, high_f, edge_mode):
        """edge_mode: 'trim','keep','remove'"""
        raise NotImplementedError

    def one_hot_labels_dict(classes=None):
        """get a dictionary of one-hot labels for entire duration

        classes=None: keep all classes (or: iterable of classes to keep)
        """
        raise NotImplementedError

    def one_hot_labels_like(
        self, clip_df, min_label_overlap, classes=None, max_ignored_label_fraction=1
    ):
        """create a dataframe of one-hot clip labels based on given starts/ends

        Uses start and end clip times from clip_df to define a set of clips.
        Then extracts annotatations associated overlapping with each clip.
        Required overlap parameters are selected by user: annotation must satisfy
        the minimum time overlap OR minimum % overlap to be included (doesn't
        require both conditions to be met, only one)

        clip_df can be created using opensoundscap.helpers.generate_clip_times_df


        min_label_overlap, #seconds, required
        max_ignored_label_fraction=1, #fraction of label that overlaps with clip
        #if >= this fraction of label is on this clip, will be included even
        #if the min_label_overlap length (seconds) is not met. <=0.5 will
        #ensure that all of the original labels end up on at least one clip.
        classes=None,#None->keep all (or, provide list):
        """
        # drop nan annotations
        df = self.df.dropna(subset=["annotation"])

        if classes is None:  # include all annotations
            classes = np.unique(df["annotation"])
        else:  # the user specified a list of classes
            # subset annotations to user-specified classes
            df = df[df["annotation"].apply(lambda x: x in classes)]

        # the clip_df should have ['start_time','end_time'] in columns
        clip_df = clip_df.set_index(["start_time", "end_time"])
        clip_df = clip_df[[]]  # remove any additional columns
        clip_df[classes] = np.nan  # add columns for each class

        for (start, end), _ in clip_df.iterrows():
            clip_df.loc[(start, end), :] = one_hot_labels_on_time_interval(
                self.df,
                start_time=start,
                end_time=end,
                min_label_overlap=min_label_overlap,
                max_ignored_label_fraction=max_ignored_label_fraction,
                classes=classes,
            )
        return clip_df

    def one_hot_clip_labels(
        self,
        full_duration,
        clip_duration,
        clip_overlap,
        final_clip,
        min_label_overlap,
        classes=None,
        max_ignored_label_fraction=1,
    ):
        """Generate one-hot labels for clips of fixed duration

        wraps generate_clip_times_df() with self.one_hot_labels_like():
        - Clips are created in the same way as audio.split()
        - Labels are applied based on overlap, as in self.one_hot_labels_like()

        Args:

        Returns:
            dataframe with index ['start_time','end_time'] and columns=classes
        """
        # generate list of start and end times for each clip
        clip_df = generate_clip_times_df(
            full_duration, clip_duration, clip_overlap, final_clip
        )
        # then create 0/1 labels for each clip and each class
        return self.one_hot_labels_like(
            clip_df, min_label_overlap, classes, max_ignored_label_fraction
        )

    def apply_corrections(self):  # out of place
        raise NotImplementedError


def combine(list_of_annotation_objects):
    """combine annotations with user specified preferences"""

    raise NotImplementedError


def diff(base_annotations, comparison_annotations):
    """ look at differences between two BoxedAnnotations objects
    """
    raise NotImplementedError


def one_hot_labels_on_time_interval(
    df,  # annotations df with columns: start_time, end_time, annotation
    start_time,  # sec since beginning of audio
    end_time,  # sec since beginning of audio
    min_label_overlap,  # seconds, required
    max_ignored_label_fraction=0.5,  # fraction of label that overlaps with clip
    # if >= this fraction of label is on this clip, will be included even
    # if the min_label_overlap length (seconds) is not met. <=0.5 will
    # ensure that all of the original labels end up on at least one clip.
    classes=None,  # None->keep all (or, provide list):
):
    """generate a dictionary of one-hot labels for given time-interval

    TODO: definitly sub-optimal computationally (keeps filtering the df
    over and over for each clip). Worth optimizing?
    """

    # calculate amount of overlap of each clip with this time window
    df["overlap"] = [
        overlap([start_time, end_time], [t0, t1])
        for t0, t1 in zip(df.start_time, df.end_time)
    ]
    # discard annotations that do not overlap with the time window
    df = df[df["overlap"] > 0]
    # calculate the fraction of each annotation that overlaps with this time window
    df["overlap_fraction"] = [
        overlap_fraction([t0, t1], [start_time, end_time])
        for t0, t1 in zip(df.start_time, df.end_time)
    ]

    one_hot_labels = [0] * len(classes)
    for i, c in enumerate(classes):
        # subset annotations to those of this class
        df_cls = df[df.annotation == c]

        if len(df_cls) == 0:
            continue  # no annotations, leave it as zero

        # if any annotation overlaps by >= max_ignored_label_fraction
        # with the time interval, clip is labeled 1
        if max(df_cls.overlap_fraction) >= max_ignored_label_fraction:
            one_hot_labels[i] = 1
        # or if any annotation overlaps with the clip by >= min_overlap, label=1
        elif max(df_cls.overlap) >= min_label_overlap:
            one_hot_labels[i] = 1
        # otherwise, we leave the label as 0

    # return a dictionary mapping classes to 0/1 labels
    return {c: l for c, l in zip(classes, one_hot_labels)}
