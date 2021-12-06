"""functions and classes for manipulating annotations of audio

includes BoxedAnnotations class and utilities to combine or "diff" annotations,
etc.
"""
from opensoundscape.helpers import overlap, overlap_fraction, generate_clip_times_df
import pandas as pd
import numpy as np
from pathlib import Path


class BoxedAnnotations:
    """ container for "boxed" (frequency-time) annotations of audio

    (for instance, annotations created in Raven software)
    includes functionality to load annotations from Raven txt files,
    output one-hot labels for specific clip lengths or clip start/end times,
    apply corrections/conversions to annotations, and more.

    Contains some analogous functions to Audio and Spectrogram, such as
    trim() [limit time range] and bandpass() [limit frequency range]
    """

    def __init__(self, df, audio_file=None):
        """
        create object directly from DataFrame of frequency-time annotations

        For loading annotations from Raven txt files, use from_raven_file()

        Args:
            df: DataFrame of frequency-time labels. Columns must include:
                - "annotation": string or numeric labels (can be None/nan)
                - "start_time": left bound, sec since beginning of audio
                - "end_time": right bound, sec since beginning of audio
                - "low_f": upper frequency bound (values can be None/nan)
                - "high_f": upper frequency bound (values can be None/nan)
            audio_file: optionally include the name or path of corresponding

        Returns:
            BoxedAnnotations object
        """
        needed_cols = ["annotation", "start_time", "end_time", "low_f", "high_f"]
        for col in needed_cols:
            assert col in df.columns, (
                f"df columns must include all of these: {str(needed_cols)}\n"
                f"columns in df: {list(df.columns)}"
            )
        self.df = df
        self.audio_file = audio_file

    def __repr__(self):
        return f"<opensoundscape.annotations.BoxedAnnotations object with audio_file={self.audio_file}.>"

    @classmethod
    def from_raven_file(
        cls, path, annotation_column, keep_extra_columns=True, audio_file=None
    ):
        """load annotations from Raven txt file

        Args:
            path: location of raven .txt file, str or pathlib.Path
            annotation_column: (str) column containing annotations
            keep_extra_columns: keep or discard extra Raven file columns
                (always keeps start_time, end_time, low_f, high_f, annotation
                audio_file). [default: True]
                - True: keep all
                - False: keep none
                - or iterable of specific columns to keep
            audio_file: optionally specify the name or path of a corresponding
                audio file.

        Returns:
            BoxedAnnotations object containing annotaitons from the Raven file
        """
        df = pd.read_csv(path, delimiter="\t")
        assert annotation_column in df.columns, (
            f"Raven file does not contain annotation_column={annotation_column}\n"
            f"(columns in file: {list(df.columns)})"
        )
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
        """save annotations to a Raven-compatible tab-separated text file

        Args:
            path: path for saved test file (extension must be ".tsv")
                - can be str or pathlib.Path

        Outcomes:
            creates a file containing the annotations in a format compatible
            with Raven Pro/Lite.

        Note: Raven Lite does not support additional columns beyond a single
        annotation column. Additional columns will not be shown in the Raven
        Lite interface.
        """
        assert Path(path).suffix == ".txt", "file extension must be .txt"

        df = self.df.copy().rename(
            columns={
                "start_time": "Begin Time (s)",
                "end_time": "End Time (s)",
                "low_f": "Low Freq (Hz)",
                "high_f": "High Freq (Hz)",
            }
        )
        df.to_csv(path, sep="\t", index=False)

    def subset(self, classes):
        """subset annotations to those from a list of classes

        out-of-place operation (returns new filtered BoxedAnnotations object)

        Args:
            classes: list of classes to retain (all others are discarded)
            - the list can include `np.nan` or `None` if you want to keep them

        Returns:
            new BoxedAnnotations object containing only annotations in `classes`
        """
        df = self.df[self.df["annotation"].apply(lambda x: x in classes)]
        return BoxedAnnotations(df, self.audio_file)

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
        assert edge_mode in ["trim", "keep", "remove"], (
            f"invalid edge_mode"
            f"argument: {edge_mode} (must be 'trim','keep', or 'remove')"
        )
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

    def unique_labels(self):
        """get list of all unique (non-Falsy) labels"""
        return np.unique(self.df.dropna(subset=["annotation"])["annotation"])

    def global_one_hot_labels(self, classes):
        """get a dictionary of one-hot labels for entire duration
        Args:
            classes: iterable of class names to give 0/1 labels

        Returns:
            list of 0/1 labels for each class
        """
        all_labels = self.unique_labels()
        return [int(c in all_labels) for c in classes]

    def one_hot_labels_like(
        self,
        clip_df,
        classes,
        min_label_overlap,
        min_label_fraction=None,
        keep_index=False,
    ):
        """create a dataframe of one-hot clip labels based on given starts/ends

        Uses start and end clip times from clip_df to define a set of clips.
        Then extracts annotatations associated overlapping with each clip.
        Required overlap parameters are selected by user: annotation must satisfy
        the minimum time overlap OR minimum % overlap to be included (doesn't
        require both conditions to be met, only one)

        clip_df can be created using opensoundscap.helpers.generate_clip_times_df

        Args:
            clip_df: dataframe with 'start_time' and 'end_time' columns
                specifying the temporal bounds of each clip
            min_label_overlap: minimum duration (seconds) of annotation within the
                time interval for it to count as a label. Note that any annotation
                of length less than this value will be discarded.
                We recommend a value of 0.25 for typical bird songs, or shorter
                values for very short-duration events such as chip calls or
                nocturnal flight calls.
            min_label_fraction: [default: None] if >= this fraction of an annotation
                overlaps with the time window, it counts as a label regardless of
                its duration. Note that *if either* of the two
                criterea (overlap and fraction) is met, the label is 1.
                if None (default), this criterion is not used (i.e., only min_label_overlap
                is used). A value of 0.5 for ths parameter would ensure that all
                annotations result in at least one clip being labeled 1
                (if there are no gaps between clips).
            classes: list of classes for one-hot labels. If None, classes will
                be all unique values of self.df['annotation']
            keep_index: if True, keeps the index of clip_df as an index
                in the returned DataFrame. [default:False]

        Returns:
            DataFrame of one-hot labels (multi-index of (start_time, end_time),
            columns for each class, values 0=absent or 1=present)
        """
        # drop nan annotations
        df = self.df.dropna(subset=["annotation"])

        if classes is None:  # include all annotations
            classes = np.unique(df["annotation"])
        else:  # the user specified a list of classes
            # subset annotations to user-specified classes
            df = df[df["annotation"].apply(lambda x: x in classes)]

        # if we want to keep the original index, the best way is
        # to store it and add again later
        if keep_index:
            og_idx = clip_df.index

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
                min_label_fraction=min_label_fraction,
                classes=classes,
            )

        # re-add the original index, if desired
        if keep_index:
            old_idx = clip_df.index.to_frame()
            old_idx.insert(0, og_idx.name, og_idx.values)
            clip_df.index = pd.MultiIndex.from_frame(old_idx)

        return clip_df

    def one_hot_clip_labels(
        self,
        full_duration,
        clip_duration,
        clip_overlap,
        classes,
        min_label_overlap,
        min_label_fraction=1,
        final_clip=None,
    ):
        """Generate one-hot labels for clips of fixed duration

        wraps helpers.generate_clip_times_df() with self.one_hot_labels_like()
        - Clips are created in the same way as Audio.split()
        - Labels are applied based on overlap, using self.one_hot_labels_like()

        Args:
            full_duration: The amount of time (seconds) to split into clips
            clip_duration (float):  The duration in seconds of the clips
            clip_overlap (float):   The overlap of the clips in seconds [default: 0]
            classes: list of classes for one-hot labels. If None, classes will
                be all unique values of self.df['annotation']
            min_label_overlap: minimum duration (seconds) of annotation within the
                time interval for it to count as a label. Note that any annotation
                of length less than this value will be discarded.
                We recommend a value of 0.25 for typical bird songs, or shorter
                values for very short-duration events such as chip calls or
                nocturnal flight calls.
            min_label_fraction: [default: None] if >= this fraction of an annotation
                overlaps with the time window, it counts as a label regardless of
                its duration. Note that *if either* of the two
                criterea (overlap and fraction) is met, the label is 1.
                if None (default), this criterion is not used (i.e., only min_label_overlap
                is used). A value of 0.5 for ths parameter would ensure that all
                annotations result in at least one clip being labeled 1
                (if there are no gaps between clips).
            final_clip (str):       Behavior if final_clip is less than clip_duration
                seconds long. By default, discards remaining time if less than
                clip_duration seconds long [default: None].
                Options:
                    - None:         Discard the remainder (do not make a clip)
                    - "extend":     Extend the final clip beyond full_duration to reach clip_duration length
                    - "remainder":  Use only remainder of full_duration (final clip will be shorter than clip_duration)
                    - "full":       Increase overlap with previous clip to yield a clip with clip_duration length
        Returns:
            dataframe with index ['start_time','end_time'] and columns=classes
        """
        # generate list of start and end times for each clip
        clip_df = generate_clip_times_df(
            full_duration=full_duration,
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
            final_clip=final_clip,
        )
        # then create 0/1 labels for each clip and each class
        return self.one_hot_labels_like(
            clip_df=clip_df,
            classes=classes,
            min_label_overlap=min_label_overlap,
            min_label_fraction=min_label_fraction,
        )

    def convert_labels(self, conversion_table):
        """modify annotations according to a conversion table

        Changes the values of 'annotation' column of dataframe.
        Any labels that do not have specified conversions are left unchanged.

        Returns a new BoxedAnnotations object, does not modify itself
        (out-of-place operation). So use could look like:
        `my_annotations = my_annotations.convert_labels(table)`

        Args:
            conversion_table: current values -> new values. can be either
                - pd.DataFrame with 2 columns [current value, new values] or
                - dictionary {current values: new values}
        Returns:
            new BoxedAnnotations object with converted annotation labels

        """
        df = self.df.copy()

        ## handle two input types for conversion_table ##
        if (
            type(conversion_table) == pd.DataFrame
            and len(conversion_table.columns) == 2
        ):
            # check that keys are unique
            keys = conversion_table.values[:, 0]
            assert len(set(keys)) == len(
                keys
            ), "keys of conversion_table must be unique"
            # convert df to dictionary
            conversion_table = {a: b for a, b in conversion_table.values}
        elif type(conversion_table) != dict:
            raise TypeError(
                "conversion_table must be dict or pd.DataFrame with 2 columns."
            )

        ## input validation ##
        # keys and values should not overlap
        keys_in_values = set(conversion_table).intersection(
            set(conversion_table.values())
        )
        assert len(keys_in_values) == 0, (
            f"conversion_table keys and values"
            f"should not overlap. Overlapping values: {keys_in_values}"
        )

        ## apply conversions ##
        df["annotation"] = [
            conversion_table[k] if k in conversion_table else k
            for k in df["annotation"]
        ]

        return BoxedAnnotations(df, self.audio_file)


def combine(list_of_annotation_objects):
    """combine annotations with user-specified preferences
    Not Implemented.
    """

    raise NotImplementedError


def diff(base_annotations, comparison_annotations):
    """ look at differences between two BoxedAnnotations objects
    Not Implemented.

    Compare different labels of the same boxes
    (Assumes that a second annotator used the same boxes as the first,
    but applied new labels to the boxes)
    """
    raise NotImplementedError


def one_hot_labels_on_time_interval(
    df, classes, start_time, end_time, min_label_overlap, min_label_fraction=None
):
    """generate a dictionary of one-hot labels for given time-interval

    Each class is labeled 1 if any annotation overlaps sufficiently with the
    time interval. Otherwise the class is labeled 0.

    Args:
        df: DataFrame with columns 'start_time', 'end_time' and 'annotation'
        classes: list of classes for one-hot labels. If None, classes will
            be all unique values of self.df['annotation']
        start_time: beginning of time interval (seconds)
        end_time: end of time interval (seconds)
        min_label_overlap: minimum duration (seconds) of annotation within the
            time interval for it to count as a label. Note that any annotation
            of length less than this value will be discarded.
            We recommend a value of 0.25 for typical bird songs, or shorter
            values for very short-duration events such as chip calls or
            nocturnal flight calls.
        min_label_fraction: [default: None] if >= this fraction of an annotation
            overlaps with the time window, it counts as a label regardless of
            its duration. Note that *if either* of the two
            criterea (overlap and fraction) is met, the label is 1.
            if None (default), the criterion is not used (only min_label_overlap
            is used). A value of 0.5 would ensure that all annotations result
            in at least one clip being labeled 1 (if no gaps between clips).

    Returns:
        dictionary of {class:label 0/1} for all classes
    """

    # calculate amount of overlap of each clip with this time window
    df.loc[:, "overlap"] = [
        overlap([start_time, end_time], [t0, t1])
        for t0, t1 in zip(df["start_time"], df["end_time"])
    ]

    # discard annotations that do not overlap with the time window
    df = df[df["overlap"] > 0].reset_index()

    # calculate the fraction of each annotation that overlaps with this time window
    df.loc[:, "overlap_fraction"] = [
        overlap_fraction([t0, t1], [start_time, end_time])
        for t0, t1 in zip(df["start_time"], df["end_time"])
    ]

    one_hot_labels = [0] * len(classes)
    for i, c in enumerate(classes):
        # subset annotations to those of this class
        df_cls = df[df["annotation"] == c]

        if len(df_cls) == 0:
            continue  # no annotations, leave it as zero

        # add label=1 if any annotation overlaps with the clip by >= min_overlap
        if max(df_cls.overlap) >= min_label_overlap:
            one_hot_labels[i] = 1

        elif (  # add label=1 if annotation's overlap exceeds minimum fraction
            min_label_fraction is not None
            and max(df_cls.overlap_fraction) >= min_label_fraction
        ):
            one_hot_labels[i] = 1
        else:  # otherwise, we leave the label as 0
            pass

    # return a dictionary mapping classes to 0/1 labels
    return {c: l for c, l in zip(classes, one_hot_labels)}


def categorical_to_one_hot(labels, classes=None):
    """transform multi-target categorical labels (list of lists) to one-hot array

    Args:
        labels: list of lists of categorical labels, eg
            [['white','red'],['green','white']] or [[0,1,2],[3]]
        classes=None: list of classes for one-hot labels. if None,
            taken to be the unique set of values in `labels`
    Returns:
        one_hot: 2d array with 0 for absent and 1 for present
        classes: list of classes corresponding to columns in the array
    """
    if classes is None:
        from itertools import chain

        classes = list(set(chain(*labels)))

    one_hot = np.zeros([len(labels), len(classes)]).astype(int)
    for i, sample_labels in enumerate(labels):
        for label in sample_labels:
            if label in classes:
                one_hot[i, classes.index(label)] = 1

    return one_hot, classes


def one_hot_to_categorical(one_hot, classes):
    """transform one_hot labels to multi-target categorical (list of lists)

    Args:
        one_hot: 2d array with 0 for absent and 1 for present
        classes: list of classes corresponding to columns in the array

    Returns:
        labels: list of lists of categorical labels for each sample, eg
            [['white','red'],['green','white']] or [[0,1,2],[3]]
    """
    classes = np.array(classes)
    return [list(classes[np.array(row).astype(bool)]) for row in one_hot]
