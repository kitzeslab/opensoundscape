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
            audio_file: optionally include the name or path of corresponding

        Returns:
            BoxedAnnotations object
        """
        for col in ["annotation", "start_time", "end_time", "low_f", "high_f"]:
            assert (
                col in df.columns
            ), 'df columns must include all of these: ["annotation","start_time","end_time","low_f","high_f"]'
        self.df = df
        self.audio_file = audio_file

    @classmethod
    def from_raven_file(
        cls, path, annotation_column, keep_extra_columns=True, audio_file=None
    ):
        """load annotations from Raven txt file

        Args:
            keep_extra_columns: True (all), False (none), or iterable of specific columns to keep
            - always includes start_time, end_time, low_f, high_f, annotation
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

        return cls(df=df, audio_file=audio_file)

    def to_raven_file(self, path):
        """save annotations in Raven-style tab-separated text file"""
        raise NotImplementedError

    def to_csv(self, path):
        """save annotations in csv file"""
        raise NotImplementedError

    def trim(self, start_time, end_time, edge_mode="trim"):
        """Trim a set of annotations, analogous to Audio/Spectrogram.trim()

        Out-of-place operation: does not modify itself, returns new object

        Args:
            start_time: time (seconds) since beginning for left bound
            end_time: time (seconds) since beginning for right bound
            edge_mode: what to do when boxes overlap with edges of trim region
                - 'trim': trim boxes to bounds
                - 'keep': allow boxes to extend beyond bounds
                - 'remove': completely remove boxes that extend beyond bounds
        Returns:
            a copy of the BoxedAnnotations object on the trimmed region.
            - note that, like Audio.trim(), there is a new reference point for
            0.0 seconds (located at start_time in the original object)

        """
        assert edge_mode in [
            "trim",
            "keep",
            "remove",
        ], f"invalid edge_mode argument: {edge_mode} (must be 'trim','keep', or 'remove')"
        assert start_time >= 0, "start time must be non-negative"
        assert end_time > start_time, "end time_must be > start_time"

        df = self.df.copy()

        # remove annotations that don't overlap with window
        df = df[
            [
                overlap([start_time, end_time], [t0, t1]) > 0
                for t0, t1 in zip(df["start_time"], df["end_time"])
            ]
        ]

        if edge_mode == "trim":  # trim boxes to start and end times
            df["start_time"] = [max(start_time, x) for x in df["start_time"]]
            df["end_time"] = [min(end_time, x) for x in df["end_time"]]
        elif edge_mode == "remove":  # remove boxes that extend beyond edges
            df = df[df["start_time"] >= start_time]
            df = df[df["end_time"] <= end_time]
        else:  #'keep': leave boxes hanging over the edges
            pass

        # as in Audio.trim, the new object's labels should be relative to the
        # new start time; so we need to offset the old values.
        df["start_time"] = df["start_time"] - start_time
        df["end_time"] = df["end_time"] - start_time

        return BoxedAnnotations(df, self.audio_file)

    def bandpass(self, low_f, high_f, edge_mode="trim"):
        """Bandpass a set of annotations, analogous to Spectrogram.bandpass()

        Out-of-place operation: does not modify itself, returns new object

        Args:
            low_f: low frequency (Hz) bound
            high_f: high frequench (Hz) bound
            edge_mode: what to do when boxes overlap with edges of trim region
                - 'trim': trim boxes to bounds
                - 'keep': allow boxes to extend beyond bounds
                - 'remove': completely remove boxes that extend beyond bounds
        Returns:
            a copy of the BoxedAnnotations object on the bandpassed region
        """
        assert edge_mode in [
            "trim",
            "keep",
            "remove",
        ], f"invalid edge_mode argument: {edge_mode} (must be 'trim','keep', or 'remove')"
        assert low_f >= 0, "low_f must be non-negative"
        assert high_f > low_f, "high_f be > low_f"

        df = self.df.copy()

        # remove annotations that don't overlap with bandpass range
        df = df[
            [
                overlap([low_f, high_f], [f0, f1]) > 0
                for f0, f1 in zip(df["low_f"], df["high_f"])
            ]
        ]

        # handle edges
        if edge_mode == "trim":  # trim boxes to start and end times
            df["low_f"] = [max(low_f, x) for x in df["low_f"]]
            df["high_f"] = [min(high_f, x) for x in df["high_f"]]
        elif edge_mode == "remove":  # remove boxes that extend beyond edges
            df = df[df["low_f"] >= low_f]
            df = df[df["high_f"] <= high_f]
        else:  #'keep': leave boxes hanging over the edges
            pass

        return BoxedAnnotations(df, self.audio_file)

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
                df,
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

    def apply_corrections(self, correction_table):
        """modify annotations according to a correction table

        Changes the values of 'annotation' column of dataframe.

        Returns a new BoxedAnnotations object, does not modify itself
        (out-of-place operation). So use could look like:
        `my_annotations = my_annotations.apply_corrections(table)`

        Args:
            correction_table: current values -> new values. can be either
                - pd.DataFrame with 2 columns [current value, new values] or
                - dictionary {current values: new values}
        Returns:
            new BoxedAnnotations object
        """
        df = self.df.copy()

        ## handle two input types for correction_table ##
        if (
            type(correction_table) == pd.DataFrame
            and len(correction_table.columns) == 2
        ):
            # check that keys are unique
            keys = correction_table.values[:, 0]
            assert len(set(keys)) == len(
                keys
            ), "keys of correction table must be unique"
            # convert df to dictionary
            correction_table = {a: b for a, b in correction_table.values}
        elif type(correction_table) != dict:
            raise TypeError(
                "correction table must be dict or pd.DataFrame with 2 columns."
            )

        ## input validation ##
        # keys and values should not overlap
        keys_in_values = set(correction_table).intersection(
            set(correction_table.values())
        )
        assert (
            len(keys_in_values) == 0
        ), f"These correction table keys and values overlap: {keys_in_values}"

        ## apply corrections ##
        df["annotation"] = [
            correction_table[k] if k in correction_table else k
            for k in df["annotation"]
        ]

        return BoxedAnnotations(df, self.audio_file)


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
