"""functions and classes for manipulating annotations of audio

includes BoxedAnnotations class and utilities to combine or "diff" annotations,
etc.
"""
from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import warnings

from opensoundscape.utils import (
    overlap,
    overlap_fraction,
    generate_clip_times_df,
    make_clip_df,
    GetDurationError,
)


class BoxedAnnotations:
    """container for "boxed" (frequency-time) annotations of audio
    (for instance, annotations created in Raven software)

    includes functionality to load annotations from Pandas DataFrame
    or Raven Selection tables (.txt files), output one-hot labels for
    specific clip lengths or clip start/end times, apply
    corrections/conversions to annotations, and more.

    Contains some analogous functions to Audio and Spectrogram, such as
    trim() [limit time range] and bandpass() [limit frequency range]

    the .df attribute is a Pandas DataFrame containing the annotations
    with time and frequency bounds

    the .annotation_files and .audio_files attributes are lists of
    annotation and audio file paths, respectively. They are retained as
    a record of _what audio was annotated_, rather than what annotations
    were placed on the audio. For instance, an audio file may have no entries
    in the dataframe if it contains no annotations, but is listed in audio_files
    because it was annotated/reviewed.
    """

    __slots__ = (
        "df",
        "annotation_files",
        "audio_files",
    )

    def __init__(self, df, annotation_files=None, audio_files=None):
        """
        create object directly from DataFrame of frequency-time annotations

        For loading annotations from Raven txt files, use `from_raven_files`

        Args:
            df: DataFrame of frequency-time labels. Columns must include:
                - "annotation": string or numeric labels (can be None/nan)
                - "start_time": left bound, sec since beginning of audio
                - "end_time": right bound, sec since beginning of audio
                optional columns:
                - "audio_file": name or path of corresponding audio file
                - "low_f": lower frequency bound (values can be None/nan)
                - "high_f": upper frequency bound (values can be None/nan)
                Note: other columns will be retained in the .df
            annotation_files: list of annotation file paths (as str or pathlib.Path)
                (e.g., raven .txt files) or None [default: None]
            audio_files: list of audio file paths (as str or pathlib.Path)
                or None [default: None]

        Returns:
            BoxedAnnotations object

        """
        # save lists
        self.annotation_files = annotation_files
        self.audio_files = audio_files

        standard_cols = [
            "audio_file",
            "annotation_file",
            "annotation",
            "start_time",
            "end_time",
            "low_f",
            "high_f",
        ]
        required_cols = ["annotation", "start_time", "end_time"]
        for col in required_cols:
            assert col in df.columns, (
                f"df columns must include all of these: {str(required_cols)}\n"
                f"columns in df: {list(df.columns)}"
            )
        # re-order columns
        # keep any extras from input df and add any missing standard columns
        ordered_cols = standard_cols + list(set(df.columns) - set(standard_cols))
        self.df = df.reindex(columns=ordered_cols)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    @classmethod
    def from_raven_files(
        cls,
        raven_files,
        audio_files=None,
        annotation_column_idx=8,
        annotation_column_name=None,
        keep_extra_columns=True,
        column_mapping_dict=None,
    ):
        """load annotations from Raven .txt files

        Args:
            raven_files: list of raven .txt file paths (as str or pathlib.Path)
            audio_files: (list) optionally specify audio files corresponding to each
                raven file (length should match raven_files)
                - if None (default), .one_hot_clip_labels() will not be able to
                check the duration of each audio file, and will raise an error
                unless `full_duration` is passed as an argument
            annotation_column_idx: (int) position of column containing annotations
                - [default: 8] will be correct if the first user-created column
                in Raven contains annotations. First column is 1, second is 2 etc.
                - pass `None` to load the raven file without explicitly
                assigning a column as the annotation column. The resulting
                object's `.df` will have an `annotation` column with nan values!
                NOTE: If `annotatino_column_name` is passed, this argument is ignored.
            annotation_column_name: (str) name of the column containing annotations
                - default: None will use annotation-column_idx to find the annotation column
                - if not None, this value overrides annotation_column_idx, and the column with
                this name will be used as the annotations.
            keep_extra_columns: keep or discard extra Raven file columns
                (always keeps start_time, end_time, low_f, high_f, annotation
                audio_file). [default: True]
                - True: keep all
                - False: keep none
                - or iterable of specific columns to keep
            column_mapping_dict: dictionary mapping Raven column names to
                desired column names in the output dataframe. The columns of the
                laoded Raven file are renamed according to this dictionary. The resulting
                dataframe must contain: ['start_time','end_time','low_f','high_f']
                [default: None]
                If None (or for any unspecified columns), will use the standard column names:
                   {
                        "Begin Time (s)": "start_time",
                        "End Time (s)": "end_time",
                        "Low Freq (Hz)": "low_f",
                        "High Freq (Hz)": "high_f",
                    }
                This dictionary will be updated with any user-specified mappings.

        Returns:
            BoxedAnnotations object containing annotations from the Raven files
            (the .df attribute is a dataframe containing each annotation)
        """
        all_file_dfs = []

        # mapping of Raven file columns to standard opensoundscape names
        # key: Raven file; value: opensoundscape name
        column_mapping_dict = {
            "Begin Time (s)": "start_time",
            "End Time (s)": "end_time",
            "Low Freq (Hz)": "low_f",
            "High Freq (Hz)": "high_f",
        }
        # update defaults with any user-specified mappings
        column_mapping_dict.update(column_mapping_dict or {})

        if audio_files is not None:
            assert len(audio_files) == len(
                raven_files
            ), """
            `audio_files` and `raven_files` lists must have one-to-one correspondence,
            but their lengths did not match.
            """
        for i, raven_file in enumerate(raven_files):
            df = pd.read_csv(raven_file, delimiter="\t")
            if annotation_column_name is not None:
                # annotation_column_name argument takes precedence over
                # annotation_column_idx. If it is passed, we use it and ignore
                # annotation_column_idx!
                if annotation_column_name in list(df.columns):
                    df = df.rename(
                        columns={
                            annotation_column_name: "annotation",
                        }
                    )
                else:
                    # to be flexible, we'll give nan values if the column is missing
                    df["annotation"] = np.nan

            elif annotation_column_idx is not None:
                # use the column number to specify which column contains annotations
                # first column is 1, second is 2, etc (default: 8th column)
                df = df.rename(
                    columns={
                        df.columns[annotation_column_idx - 1]: "annotation",
                    }
                )
            else:  # None was passed to annotatino_column_idx
                # we'll create an empty `annotation` column
                df["annotation"] = np.nan

            # rename Raven columns to standard opensoundscape names
            try:
                df = df.rename(
                    columns=column_mapping_dict,
                    errors="raise",
                )
            except KeyError as e:
                raise KeyError(
                    "Raven file is missing a required column. "
                    "Raven files must have columns matching the following names: "
                    f"{column_mapping_dict.keys()}"
                ) from e

            # add column containing the raven file path
            df["annotation_file"] = raven_file

            # remove undesired columns
            standard_columns = [
                "annotation_file",
                "start_time",
                "end_time",
                "low_f",
                "high_f",
            ]
            if annotation_column_idx is not None:
                standard_columns.append("annotation")
            if hasattr(keep_extra_columns, "__iter__"):
                # keep the desired columns
                # if values in keep_extra_columns are missing, fill with nan
                df = df.reindex(
                    columns=standard_columns + list(keep_extra_columns),
                    fill_value=np.nan,
                )
            elif not keep_extra_columns:
                # only keep required columns
                df = df.reindex(columns=standard_columns)
            else:
                # keep all columns
                pass

            # add audio file column
            if audio_files is not None:
                df["audio_file"] = audio_files[i]
            else:
                df["audio_file"] = np.nan

            all_file_dfs.append(df)

        # we drop the original index from the Raven annotations when we combine tables
        # if the dataframes have different columns, we fill missing columns with nan values
        # and keep all unique columns
        all_annotations = pd.concat(all_file_dfs).reset_index(drop=True)

        return cls(
            df=all_annotations,
            annotation_files=raven_files,
            audio_files=audio_files,
        )

    def to_raven_files(self, save_dir, audio_files=None):  # TODO implement to_csv
        """save annotations to a Raven-compatible tab-separated text files

        Creates one file per unique audio file in 'file' column of self.df

        Args:
            save_dir: directory for saved files
                - can be str or pathlib.Path
            audio_files: list of audio file paths (as str or pathlib.Path)
                or None [default: None]. If None, uses self.audio_files.
                Note that it does not use self.df['audio_file'].unique()

        Outcomes:
            creates files containing the annotations for each audio file
            in a format compatible with Raven Pro/Lite. File is tab-separated
            and contains columns matching the Raven defaults.

        Note: Raven Lite does not support additional columns beyond a single
        annotation column. Additional columns will not be shown in the Raven
        Lite interface.
        """
        assert Path(save_dir).exists(), f"Output directory {save_dir} does not exist"

        if audio_files is None:
            if self.audio_files is None:
                raise ValueError(
                    "`audio_files` must be specified since `self.audio` is `None`."
                    "This function creates one annotation file per item in `audio_files`."
                )
            audio_files = self.audio_files

        df = self.df.copy()  # avoid modifying df of original object

        # rename columns to match Raven defaults
        df = df.rename(
            columns={
                "start_time": "Begin Time (s)",
                "end_time": "End Time (s)",
                "low_f": "Low Freq (Hz)",
                "high_f": "High Freq (Hz)",
            }
        )

        # warn user if self.df has audio files not in `audio_files`
        # because these won't make it into any of the output files
        if audio_files is not None:
            if len(set(df["audio_file"].unique()) - set(audio_files)) > 0:
                warnings.warn(
                    """
                    Some audio files in self.df['audio_file'] are not in `audio_files`.
                    Annotations from these files will not be saved to Raven files.

                    Consider passing `audio_files=ba.df['audio_file'].unique()` if 
                    you want to save an annotation file for each file in `.df`.
                    """
                )

        # we will create one selection table for each file
        # this list may contain NaN, which we handle below
        unique_files = list(set(audio_files))

        # If file names are not unique, raise an Exception
        # otherwise, multiple selection table files with the same name would be
        # written to one directory
        file_stems = set([Path(f).stem for f in unique_files])
        if len(file_stems) < len(unique_files):  # TODO write test for this exception
            raise Exception(
                """
                File names were not unique! Some files in different folders have the same name.
                This is not allowed since we plan to save each file's annotations to a
                selection table with the same name in `save_dir`.  

                Consider subsetting `.df` to avoid this issue. 
                """
            )

        # save each file's annotations to a separate raven-formatted txt file
        for file in unique_files:
            # for NaN values of file, call the output file "unspecified_audio.selections.txt"
            if not file == file:
                file_df = df[df["audio_file"].isnull()]
                fname = "unspecified_audio.selections.txt"
            else:
                # subset to annotations for this file
                file_df = df[df["audio_file"] == file]
                fname = f"{Path(file).stem}.selections.txt"
            file_df.to_csv(f"{save_dir}/{fname}", sep="\t", index=False)

    def _spawn(self, **kwargs):
        """return copy of object, replacing any desired fields from __slots__

        pass any desired updates as kwargs
        """
        assert np.all([k in self.__slots__ for k in kwargs.keys()]), (
            "only pass members of BoxedAnnotations.__slots__ to _spawn as kwargs! "
            f"slots: {self.__slots__}"
        )
        # load the current values from each __slots__ key
        slots = {key: self.__getattribute__(key) for key in self.__slots__}
        # update any user-specified values
        slots.update(kwargs)
        # create new instance of the class
        return self.__class__(**slots)

    def subset(self, classes):
        """subset annotations to those from a list of classes

        out-of-place operation (returns new filtered BoxedAnnotations object)

        Args:
            classes: list of classes to retain (all others are discarded)
            - the list can include `nan` or `None` if you want to keep them

        Returns:
            new BoxedAnnotations object containing only annotations in `classes`
        """
        df = self.df.copy()  # avoid modifying df of original object
        df = df[df["annotation"].apply(lambda x: x in classes)]
        # keep the same lists of annotation_files and audio_files
        return self._spawn(df=df)

    def trim(self, start_time, end_time, edge_mode="trim"):
        """Trim the annotations of each file in time

        Trims annotations from outside of the time bounds. Note that the annotation
        start and end times of different files may not represent the same real-world times.
        This function only uses the numeric values of annotation start and end times in
        the annotations, which should be relative to the beginning of the corresponding
        audio file.

        For zero-length annotations (start_time = end_time), start and end times are
        inclusive on the left and exclusive on the right, ie [lower,upper).
        For instance start_time=0, end_time=1 includes zero-length annotations at 0
        but excludes zero-length annotations a 1.

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
            0.0 seconds (located at start_time in the original object). For example,
            calling .trim(5,10) will result in an annotation previously starting at 6s
            to start at 1s in the new object.

        """
        assert edge_mode in [
            "trim",
            "keep",
            "remove",
        ], f"invalid edge_mode argument: {edge_mode} (must be 'trim','keep', or 'remove')"
        assert start_time >= 0, "start time must be non-negative"
        assert end_time > start_time, "end time_must be greater than start_time"

        df = self.df.copy()  # avoid modifying df of original object

        # select annotations that overlap with window
        def in_bounds(t0, t1):
            """check if annotation from t0 to t1 falls within
            the range [start_range,end_range)

            inclusive on left, exclusive on right
            """
            assert t0 <= t1
            ends_before_bounds = t1 < start_time
            starts_after_bounds = t0 >= end_time
            return not (ends_before_bounds or starts_after_bounds)

        df = df.loc[
            [in_bounds(t0, t1) for t0, t1 in zip(df["start_time"], df["end_time"])], :
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

        return self._spawn(df=df)

    def bandpass(self, low_f, high_f, edge_mode="trim"):
        """Bandpass a set of annotations, analogous to Spectrogram.bandpass()

        Reduces the range of annotation boxes overlapping with the bandpass limits,
        and removes annotation boxes entirely if they lie completely outside of the
        bandpass limits.

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
        df = df.loc[
            [
                overlap([low_f, high_f], [f0, f1]) > 0
                for f0, f1 in zip(df["low_f"], df["high_f"])
            ],
            :,
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

        return self._spawn(df=df)

    def unique_labels(self):
        """get list of all unique labels

        ignores null/Falsy labels by performing .df.dropna()
        """
        return self.df.dropna(subset=["annotation"])["annotation"].unique()

    def global_one_hot_labels(self, classes):
        """get a list of one-hot labels for entire set of annotations
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
        min_label_overlap,
        min_label_fraction=None,
        class_subset=None,
        warn_no_annotations=False,
    ):
        """create a dataframe of one-hot clip labels based on given starts/ends

        Uses start and end clip times from clip_df to define a set of clips
        for each file. Then extracts annotations overlapping with each clip.

        Required overlap to consider an annotation to overlap with a clip
        is defined by user: an annotation must satisfy
        the minimum time overlap OR minimum % overlap to be included (doesn't
        require both conditions to be met, only one)

        clip_df can be created using `opensoundscap.utils.make_clip_df`

        See also: `.one_hot_clip_labels()`, which creates even-lengthed clips
        automatically and can often be used instead of this function.

        Args:
            clip_df: dataframe with (file, start_time, end_time) MultiIndex
                specifying the temporal bounds of each clip
                (clip_df can be created using `opensoundscap.helpers.make_clip_df`)
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
            class_subset: list of classes for one-hot labels. If None, classes will
                be all unique values of self.df['annotation']
            warn_no_annotations: bool [default:False] if True, raises warnings for
                any files in clip_df with no corresponding annotations in self.df

        Returns:
            DataFrame of one-hot labels w/ multi-index of (file, start_time, end_time),
            a column for each class, and values of 0=absent or 1=present
        """
        # drop nan annotations
        df = self.df.dropna(subset=["annotation"])  # creates new copy of df object

        if class_subset is None:  # include all annotations
            classes = df["annotation"].unique()
        else:  # the user specified a list of classes
            classes = class_subset
            # subset annotations to user-specified classes,
            # removing rows with annotations for other classes
            df = df[df["annotation"].isin(classes)]

        # the clip_df should have ['file','start_time','end_time'] as the index
        clip_df[classes] = float("nan")  # add columns for each class

        for file, start, end in clip_df.index:
            if not file == file:  # file is NaN, get corresponding rows
                file_df = df[df["audio_file"].isnull()]
            else:  # subset annotations to this file
                file_df = df[df["audio_file"] == file]

            # warn user if no annotations correspond to this file
            if warn_no_annotations and len(file_df) == 0:
                warnings.warn(
                    f"No annotations matched the file {file}. All "
                    "clip labels will be zero for this file."
                )

            # add clip labels for this row of clip dataframe
            clip_df.loc[(file, start, end), :] = one_hot_labels_on_time_interval(
                file_df,
                start_time=start,
                end_time=end,
                min_label_overlap=min_label_overlap,
                min_label_fraction=min_label_fraction,
                class_subset=classes,
            )

        return clip_df

    def one_hot_clip_labels(
        self,
        clip_duration,
        clip_overlap,
        min_label_overlap,
        min_label_fraction=1,
        full_duration=None,
        class_subset=None,
        final_clip=None,
        audio_files=None,
    ):
        """Generate one-hot labels for clips of fixed duration

        wraps utils.make_clip_df() with self.one_hot_labels_like()
        - Clips are created in the same way as Audio.split()
        - Labels are applied based on overlap, using self.one_hot_labels_like()

        Args:
            clip_duration (float):  The duration in seconds of the clips
            clip_overlap (float):   The overlap of the clips in seconds [default: 0]
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
            full_duration: The amount of time (seconds) to split into clips for each file
                float or `None`; if `None`, attempts to get each file's duration
                using `librosa.get_duration(path=file)` where file is the value
                of `audio` for each row of self.df
            class_subset: list of classes for one-hot labels. If None, classes will
                be all unique values of self.df['annotation']
            final_clip (str): Behavior if final_clip is less than clip_duration
                seconds long. By default, discards remaining time if less than
                clip_duration seconds long [default: None].
                Options:
                - None: Discard the remainder (do not make a clip)
                - "extend": Extend the final clip beyond full_duration to reach
                    clip_duration length
                - "remainder": Use only remainder of full_duration
                    (final clip will be shorter than clip_duration)
                - "full": Increase overlap with previous clip to yield a
                    clip with clip_duration length
            audio_files: list of audio file paths (as str or pathlib.Path)
                to create clips for. If None, uses self.audio_files. [default: None]
        Returns:
            dataframe with index ['file','start_time','end_time'] and columns=classes
        """

        # use self.audio_files as list of files to create clips for, unless user passed audio_file
        if audio_files is None:
            if self.audio_files is None:
                raise ValueError(
                    """
                    self.audio_files cannot be None. 
                    This function uses self.audio_files to determine what files to
                    create clips for. If you want to create clips for all files in
                    self.df, pass `audio_files=self.df['audio_file'].unique()` 
                    """
                )
            else:
                audio_files = self.audio_files

        audio_files = list(set(audio_files))  # remove duplicates

        # generate list of start and end times for each clip
        # if user passes None for full_duration, try to get the duration from each audio file

        if full_duration is None:
            try:
                clip_df = make_clip_df(
                    files=[f for f in audio_files if f == f],  # remove NaN if present
                    clip_duration=clip_duration,
                    clip_overlap=clip_overlap,
                    final_clip=final_clip,
                    raise_exceptions=True,  # raise exceptions from librosa.duration(f)
                )
            except GetDurationError as exc:
                raise GetDurationError(
                    """`full_duration` was None, but failed to retrieve the durations of 
                    some files. This could occur if the values of 'file' in self.df are 
                    not paths to valid audio files. Specifying `full_duration` as an 
                    argument to `one_hot_clip_labels()` will avoid the attempt to get 
                    audio file durations from file paths."""
                ) from exc
        else:  # use fixed full_duration for all files
            # make a clip df, will be re-used for each file
            clip_df_template = generate_clip_times_df(
                full_duration=full_duration,
                clip_duration=clip_duration,
                clip_overlap=clip_overlap,
                final_clip=final_clip,
            )
            # make a clip df for all files
            clip_df = pd.concat([clip_df_template] * len(audio_files))
            # add file column, repeating value of file across each clip
            clip_df["file"] = np.repeat(audio_files, len(clip_df_template))
            clip_df = clip_df.set_index(["file", "start_time", "end_time"])

        # then create 0/1 labels for each clip and each class
        return self.one_hot_labels_like(
            clip_df=clip_df,
            class_subset=class_subset,
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

        return BoxedAnnotations(df)


def diff(base_annotations, comparison_annotations):
    """look at differences between two BoxedAnnotations objects
    Not Implemented.

    Compare different labels of the same boxes
    (Assumes that a second annotator used the same boxes as the first,
    but applied new labels to the boxes)
    """
    raise NotImplementedError


def one_hot_labels_on_time_interval(
    df, class_subset, start_time, end_time, min_label_overlap, min_label_fraction=None
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
    # create a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # calculate amount of overlap of each clip with this time window
    df["overlap"] = [
        overlap([start_time, end_time], [t0, t1])
        for t0, t1 in zip(df["start_time"], df["end_time"])
    ]

    # discard annotations that do not overlap with the time window
    df = df[df["overlap"] > 0].reset_index()

    # calculate the fraction of each annotation that overlaps with this time window
    df["overlap_fraction"] = [
        overlap_fraction([t0, t1], [start_time, end_time])
        for t0, t1 in zip(df["start_time"], df["end_time"])
    ]

    one_hot_labels = [0] * len(class_subset)
    for i, c in enumerate(class_subset):
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
    return {c: l for c, l in zip(class_subset, one_hot_labels)}


def categorical_to_one_hot(labels, class_subset=None):
    """transform multi-target categorical labels (list of lists) to one-hot array

    Args:
        labels: list of lists of categorical labels, eg
            [['white','red'],['green','white']] or [[0,1,2],[3]]
        classes=None: list of classes for one-hot labels. if None,
            taken to be the unique set of values in `labels`
    Returns:
        one_hot: 2d array with 0 for absent and 1 for present
        class_subset: list of classes corresponding to columns in the array
    """
    if class_subset is None:
        class_subset = list(set(itertools.chain(*labels)))

    one_hot = np.zeros([len(labels), len(class_subset)]).astype(int)
    for i, sample_labels in enumerate(labels):
        for label in sample_labels:
            if label in class_subset:
                one_hot[i, class_subset.index(label)] = 1

    return one_hot, class_subset


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
