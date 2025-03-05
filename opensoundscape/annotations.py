"""functions and classes for manipulating annotations of audio

includes BoxedAnnotations class and utilities to combine or "diff" annotations,
etc.
"""

from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import warnings
import crowsetta
from sklearn.model_selection import train_test_split

from opensoundscape.utils import (
    overlap,
    overlap_fraction,
    generate_clip_times_df,
    make_clip_df,
    GetDurationError,
)

import scipy.sparse


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

    __slots__ = ("df", "annotation_files", "audio_files")
    _required_cols = ["annotation", "start_time", "end_time"]
    _standard_cols = [
        "audio_file",
        "annotation_file",
        "annotation",
        "start_time",
        "end_time",
        "low_f",
        "high_f",
    ]

    def __init__(self, df=None, annotation_files=None, audio_files=None):
        """
        create object directly from DataFrame of frequency-time annotations

        For loading annotations from Raven txt files, use `from_raven_files`

        see also: .from_crowsetta() for integration with the crowsetta package

        Args:
            df: DataFrame of frequency-time labels. Columns must include:
                - "annotation": string or numeric labels (can be None/nan)
                - "start_time": left bound, sec since beginning of audio
                - "end_time": right bound, sec since beginning of audio
                optional columns:
                - "audio_file": name or path of corresponding audio file
                - "low_f": lower frequency bound (values can be None/nan)
                - "high_f": upper frequency bound (values can be None/nan)
                    if df is None, creates object with no annotations

                if None (default), creates BoxedAnnotations object with empty .df

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

        if df is None:
            # create empty dataframe with standard columns
            df = pd.DataFrame(columns=self._standard_cols)

        for col in self._required_cols:
            assert col in df.columns, (
                f"df columns must include all of these: {str(self._required_cols)}\n"
                f"columns in df: {list(df.columns)}"
            )
        # re-order columns with standard columns first
        # keep any extras from input df and add any missing standard columns
        ordered_cols = self._standard_cols + list(
            set(df.columns) - set(self._standard_cols)
        )
        self.df = df.reindex(columns=ordered_cols)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    @classmethod
    def from_raven_files(
        cls,
        raven_files,
        annotation_column,
        audio_files=None,
        keep_extra_columns=True,
        column_mapping_dict=None,
        warn_no_annotations=False,
    ):
        """load annotations from Raven .txt files

        Args:
            raven_files: list or iterable of raven .txt file paths (as str or pathlib.Path),
                or a single file path (str or pathlib.Path). Eg ['path1.txt','path2.txt']
            annotation_column: column name(s) or integer position to use as the annotations
                - pass `None` to load the Raven file without explicitly
                assigning a column as the annotation column. The resulting
                object's `.df` will have an `annotation` column with nan values!
                - if a string is passed, the column with this name will be used as the annotations.
                - if an integer is passed, the column at that position will be used as the annotation column.
                    NOTE: column positions are ordered increasingly starting at 0.
                - if a list/tuple is passed, find a column matching any value in the list
                    NOTE: if multiple columns match, an error will be raised
                    Example: ['annotation','label','Species'] will find a column with any of these names
            audio_files: (list) optionally specify audio files corresponding to each
                raven file (length should match raven_files) Eg ['path1.txt','path2.txt']
                - if None (default), .clip_labels() will not be able to
                check the duration of each audio file, and will raise an error
                unless `full_duration` is passed as an argument
            keep_extra_columns: keep or discard extra Raven file columns
                (always keeps start_time, end_time, low_f, high_f, annotation
                audio_file). [default: True]
                - True: keep all
                - False: keep none
                - or iterable of specific columns to keep
            column_mapping_dict: dictionary mapping Raven column names to
                desired column names in the output dataframe. The columns of the
                loaded Raven file are renamed according to this dictionary. The resulting
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
            warn_no_annotations: [default: False] if True, will issue a warning
                if a Raven file has zero rows (meaning no annotations present).

        Returns:
            BoxedAnnotations object containing annotations from the Raven files
            (the .df attribute is a dataframe containing each annotation)
        """
        # check input type of raven_files and audio_files
        # if a single path is passed, convert to list
        if isinstance(raven_files, (str, Path)):
            raven_files = [raven_files]
        else:
            assert (
                len(raven_files) > 0
            ), "raven_files must be a non-empty list or iterable"
            assert isinstance(
                raven_files[0], (str, Path)
            ), f"raven_files must be an iterable of string or pathlib.Path, or a single string or pathlib.Path. Got type: {type(raven_files)}"

        if isinstance(audio_files, (str, Path)):
            audio_files = [audio_files]
        else:
            if audio_files is not None:
                assert isinstance(
                    audio_files[0], (str, Path)
                ), f"audio_files must be an iterable of string or pathlib.Path, or a single string or pathlib.Path. Got type: {type(audio_files)}"

        if audio_files is not None:
            assert len(audio_files) == len(
                raven_files
            ), """
            `audio_files` and `raven_files` lists must have one-to-one correspondence,
            but their lengths did not match.
            """

        assert isinstance(
            annotation_column, (str, int, type(None), list, tuple)
        ), "Annotation column index has to be a string, integer, list, tuple, or None."

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

        for i, raven_file in enumerate(raven_files):
            df = pd.read_csv(raven_file, delimiter="\t")
            if df.empty and warn_no_annotations:
                warnings.warn(f"{raven_file} has zero rows.")
                continue

            # handle varioius options for specifying the annotation column
            if isinstance(annotation_column, str):
                # annotation_column is a string that is present in the annotation file's header
                try:
                    df = df.rename(
                        columns={
                            annotation_column: "annotation",
                        },
                        errors="raise",
                    )
                except KeyError as e:
                    raise KeyError(
                        f"Specified column name, {annotation_column}, does not match any of the column names in the annotation file: "
                        f"{list(df.columns)}. "
                        f"Please provide an annotation column name that exists or None!"
                    ) from e

            elif isinstance(annotation_column, int):
                # using the column number to specify which column contains annotations
                # first column is 1, second is 2, etc
                if not 0 <= annotation_column < len(df.columns):
                    # ensure column number is within bounds
                    raise IndexError(
                        f"""Specified annotation column index ({annotation_column}) is out of bounds
                        of the columns in the annotation file. Please provide a number between 0 and
                        {len(df.columns)-1}! Please keep in mind Python uses zero-based indexing,
                        meaning the column numbers start at 0."""
                    )
                df = df.rename(
                    columns={
                        df.columns[annotation_column]: "annotation",
                    },
                    errors="raise",
                )
            elif isinstance(annotation_column, (list, tuple)):
                annotation_column = list(annotation_column)
                # make sure exactly one value from annotation_column is in the df.columns
                matching_cols = [col for col in annotation_column if col in df.columns]
                if len(matching_cols) == 0:
                    raise KeyError(
                        f"None of the specified annotation columns, {annotation_column}, "
                        f"match any of the column names in the annotation file: {list(df.columns)} "
                        f"when attempting to load {raven_file}. "
                        f"Please ensure all raven files contain one of the specified annotation_column values."
                    )
                elif len(matching_cols) > 1:
                    raise KeyError(
                        f"Multiple columns in the annotation file match the specified annotation columns: "
                        f"{matching_cols}. when attempting to load {raven_file}. "
                        "Please ensure only one column in each raven file matches a value listed in annotation_columns"
                    )
                else:
                    # rename the column to 'annotation'
                    df = df.rename(
                        columns={
                            matching_cols[0]: "annotation",
                        },
                        errors="raise",
                    )

            else:
                # None was passed to annotation_column
                # we'll create an empty `annotation` column
                df["annotation"] = pd.Series(dtype="object")

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

            # add audio file column
            if audio_files is not None:
                df["audio_file"] = audio_files[i]
            else:
                df["audio_file"] = np.nan

            # subset and re-order columns
            if hasattr(keep_extra_columns, "__iter__"):
                # keep the desired columns
                # if values in keep_extra_columns are missing, fill with nan
                df = df.reindex(
                    columns=cls._standard_cols + list(keep_extra_columns),
                    fill_value=np.nan,
                )
            elif not keep_extra_columns:
                # only keep required columns
                df = df.reindex(columns=cls._standard_cols)
            else:
                # keep all columns
                pass

            all_file_dfs.append(df)

        if len(all_file_dfs) > 0:
            # we drop the original index from the Raven annotations when we combine tables
            # if the dataframes have different columns, we fill missing columns with nan values
            # and keep all unique columns
            all_annotations_df = pd.concat(all_file_dfs).reset_index(drop=True)

        else:
            all_annotations_df = pd.DataFrame(columns=cls._required_cols)

        return cls(
            df=all_annotations_df,
            annotation_files=raven_files,
            audio_files=audio_files,
        )

    @classmethod
    def from_crowsetta_bbox(cls, bbox, audio_file, annotation_file):
        """create BoxedAnnotations object from a crowsetta.BBox object

        Args:
            bbox: a crowsetta.BBox object
            audio_file: (str) path of annotated audio file
            annotation_file: (str) path of annotation file

        Returns:
            BoxedAnnotations object

        this classmethod is used by from_crowsetta()

        """
        return cls(
            df=pd.DataFrame(
                {
                    "audio_file": audio_file,
                    "annotation_file": annotation_file,
                    "annotation": bbox.label,
                    "start_time": bbox.onset,
                    "end_time": bbox.offset,
                    "low_f": bbox.low_freq,
                    "high_f": bbox.high_freq,
                },
                index=[0],
            )
        )

    @classmethod
    def from_crowsetta_seq(cls, seq, audio_file, annotation_file):
        """create BoxedAnnotations from crowsetta.Sequence object

        Note: low_f and high_f will be None since Sequence does not
        contain information about frequency

        Note: the `.df` of the returned BoxedAnnotations retains
        the Sequence's `.onset_samples` and `.offset_samples` information,
        but only uses the Sequence's `.onsets_s` and `.offsets_s`
        (which may sometimes be `None`) for the `start_time` and `end_time`
        columns in `BoxedAnnotations.df`.

        Args:
            seq: a crowsetta.Sequence object
            audio_file: (str) path of annotated audio file
            annotation_file: (str) path of annotation file

        Returns:
            BoxedAnnotations object

        this classmethod is used by from_crowsetta()
        """
        return cls(
            df=pd.DataFrame(
                {
                    "audio_file": audio_file,
                    "annotation_file": annotation_file,
                    "annotation": seq.labels,
                    "start_time": seq.onsets_s,
                    "end_time": seq.offsets_s,
                    "low_f": None,
                    "high_f": None,
                    "onset_sample": seq.onset_samples,
                    "offset_sample": seq.offset_samples,
                }
            )
        )

    @classmethod
    def from_crowsetta(cls, annotations, audio_files=None, annotation_files=None):
        """create BoxedAnnotations from crowsetta.Annotation object or list of Annotation objects


        Args:
            annotations: crowsetta.Annotation object or list of objects
                the objects _either_ have
                `.bbox`: list of BBox objects, OR
                `.seq`: Sequence object with list of values for onset/offset
                    (or sample onset/offset), labels
            audio_files: optionally, pass list of the annotated audio files
                (this might include files with zero annotations)
            annotation_files: optionally, pass list of files containing annotations

        Returns:
            BoxedAnnotations object containing the annotations in .df,
            and possibly containing the provided .audio_files and .annotation_files lists

        Note: if an empty list is passed, creates empty BoxedAnnotations object
        """
        # store individual objects in a list, starting with an empty BoxedAnnotations object
        boxed_anns = []

        if type(annotations) == crowsetta.Annotation:
            annotations = [
                annotations
            ]  # now we have a list of Annotation regardless of user input
        for ann_i, ann in enumerate(annotations):
            assert (
                type(ann) == crowsetta.Annotation
            ), f"`annotations` must be a list of crowsetta.Annotations objects, got {type(ann)}"

            # handle three cases: Annotations has .bbox: list of BBox or .seq: Sequence, or .sequence: list of Sequence
            if hasattr(ann, "bboxes"):
                for bbox in ann.bboxes:
                    # ann.bbox is a list of BBox objects with attributes
                    ba = cls.from_crowsetta_bbox(
                        bbox,
                        annotation_file=ann.annot_path,
                        audio_file=ann.notated_path,
                    )
                    ba.df["annotation_id"] = ann_i  # keep record of annotation number
                    boxed_anns.append(ba)
            else:  # create BoxedAnnotations objects from ann.seq (crowsetta.Sequence's)
                # ann.seq might be a list of Sequence or just a Sequence object
                # if single object, convert to list for consistency
                if type(ann.seq) == crowsetta.Sequence:
                    seqs = [ann.seq]
                else:
                    seqs = ann.seq

                for seq_i, seq in enumerate(seqs):
                    ba = cls.from_crowsetta_seq(
                        seq=seq,
                        annotation_file=ann.annot_path,
                        audio_file=ann.notated_path,
                    )
                    ba.df["sequence_id"] = seq_i  # keep record of sequence number
                    ba.df["annotation_id"] = ann_i  # keep record of annotation number
                    boxed_anns.append(ba)

        ba = cls.concat(boxed_anns)
        ba.audio_files = audio_files
        ba.annotation_files = annotation_files

        return ba

    @classmethod
    def from_csv(cls, path):
        """load csv from path and creates BoxedAnnotations object

        Note: the .annotation_files and .audio_files attributes will be none

        Args:
            path: file path of csv.
                see __init__() docstring for required column names

        Returns:
            BoxedAnnotations object
        """
        df = pd.read_csv(path)
        return cls(df)

    def to_csv(self, path):
        """save annotation table as csv-formatted text file

        Note: the .annotation_files and .audio_files attributes are not saved,
        only .df is retained in the generated csv file

        Args:
            path: file path to save to

        Effects:
            creates a text file containing comma-delimited contents of self.df
        """
        self.df.to_csv(path, index=False)

    def to_raven_files(self, save_dir, audio_files=None):
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

        # make list of unique files, while retaining order
        # we will create one selection table for each file
        # this list may contain NaN, which we handle below
        unique_files = [] if audio_files is None else unique(audio_files)

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

    def to_crowsetta(
        self, mode="bbox", ignore_annotation_id=False, ignore_sequence_id=False
    ):
        """create crowsetta.Annotations objects

        Creates (at least) one crowsetta.Annotation object per unique combination of `audio_file`,`annotation_file` in self.df
        - if `annotation_id` column is present, creates one Annotation object per unique value of
        `annotation_id` per unique combination of `audio_file` and `annotation_file`
        - if `sequence_id` column is present and mode=='sequence', creates one Sequence for each unique sequence_id
            within an Annotation object (Annotation.seq will be a list of Sequences). (If `sequence_id` is not
            in the columns, Annotation.seq will just be a Sequence object).

        Args:
            mode: 'bbox' or 'sequence'
            - if mode=='bbox', Annotations have attribute .bboxes
            - if mode=='sequence', Annotations have attribute .seq
                - list of Sequences, one Sequence for each unique value of `sequence_id`
            ignore_annotation_id: [default: False]
                if True, creates on Annotation object per unique `audio_file` and `annotation_file` object
                ignoring `annotation_id`. Otherwise, creates separate objects for each unique `annotation_id`
                for each unique combination of `audio_file` and `annotation_file`.
            ignore_sequence_id: [default: False]
                if True, creates on Sequence object for Annotation.seq ignoring `annotation_id`.
                Otherwise, Annotation.seq will be a list of Sequence objects,
                one for each unique `annotation_id` in the subset of annotations being
                created for a single Annotation object.
                Note: Only relevant for mode='sequence'

        Returns:
            list of crowsetta.Annotation objects (one per unique value of `audio_file` in self.df
            - if mode=='bbox', Annotations have attribute .bboxes
            - if mode=='sequence', Annotations have attribute .seq)
        """
        assert mode in ("bbox", "sequence"), "invalid mode, choose 'bbox' or 'sequence'"

        ann_objects = []  # we plan to return a list of crowsetta.Annotation objects

        # iterate through each unique combination of audio file and annotation file
        for aud_file, ann_file in list(
            self.df.groupby(["audio_file", "annotation_file"]).groups.keys()
        ):
            subset = self.df[
                (self.df["audio_file"] == aud_file)
                & (self.df["annotation_file"] == ann_file)
            ]

            # if `annotation_id` column is present, make one Annotation per unique value
            # (unless user passes `ignore_annotation_id=True`)
            # otherwise, just make one Annotation with all rows
            if not "annotation_id" in subset.columns or ignore_annotation_id:
                subset["annotation_id"] = 0

            for ann_id in subset["annotation_id"].unique():
                ann_labels = subset[subset["annotation_id"] == ann_id]

                # create crowsetta Annotation object
                seq = None
                bboxes = None
                if mode == "bbox":
                    bboxes = _df_to_crowsetta_bboxes(ann_labels)

                else:
                    # create Sequences:
                    # if `sequence_id` column is present,
                    # (and user has not passed ignore_sequence_id=True),
                    # make one Annotation per unique value
                    # otherwise, just make one Annotation using all rows
                    if "sequence_id" in ann_labels.columns and (not ignore_sequence_id):
                        seq = []  # one per `annotation_id` value
                        for seq_id in ann_labels["sequence_id"].unique():
                            seq_labels = ann_labels[ann_labels["sequence_id"] == seq_id]
                            seq.append(_df_to_crowsetta_sequence(seq_labels))
                    else:
                        seq = _df_to_crowsetta_sequence(ann_labels)

                ann_objects.append(
                    crowsetta.Annotation(
                        annot_path=ann_file,
                        notated_path=aud_file,
                        seq=seq,
                        bboxes=bboxes,
                    )
                )
        return ann_objects

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

    def global_multi_hot_labels(self, classes):
        """make list of 0/1 for presence/absence of classes across all annotations

        Args:
            classes: iterable of class names to give 0/1 labels

        Returns:
            list of 0/1 labels for each class
        """
        all_labels = self.unique_labels()
        return [int(c in all_labels) for c in classes]

    def labels_on_index(
        self,
        clip_df,
        min_label_overlap,
        min_label_fraction=None,
        class_subset=None,
        return_type="multihot",
        keep_duplicates=False,
        warn_no_annotations=False,
    ):
        """create a dataframe of clip labels based on given starts/ends.

        Format of label dataframe depends on `return_type` argument:
            'multihot': [default] returns a dataframe with a column for each class
                and 0/1 values for class presence.
            'integers': returns a dataframe with 'labels' column containing lists of
                integer class indices for each clip, corresponding to
                the `classes` list; also returns a second value, the list of class names
            'classes': returns a dataframe with 'labels' column containing lists of
                class names for each clip
            'CategoricalLabels': returns a CategoricalLabels object

        Uses start and end clip times from clip_df to define a set of clips
        for each file. Then extracts annotations overlapping with each clip.

        Required overlap to consider an annotation to overlap with a clip
        is defined by user: an annotation must satisfy
        the minimum time overlap OR minimum % overlap to be included (doesn't
        require both conditions to be met, only one)

        clip_df can be created using `opensoundscap.utils.make_clip_df`

        See also: `.clip_labels()`, which creates even-lengthed clips
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
            return_type: ('multihot','integers','classes', or 'CategoricalLabels'):
                'multihot': [default] returns a dataframe with a column for each class
                    and 0/1 values for class presence.
                'integers': returns a dataframe with 'labels' column containing lists of
                    integer class indices for each clip, corresponding to
                    the `classes` list; also returns a second value, the list of class names
                'classes': returns a dataframe with 'labels' column containing lists of
                    class names for each clip
                'CategoricalLabels': returns a CategoricalLabels object
            keep_duplicates: [default: False] if True, allows multiple annotations
                of a class to be retained for a single clip; e.g. labels ['a','a','b].
                Ignored if return_type is 'multihot'
            warn_no_annotations: bool [default:False] if True, raises warnings for
                any files in clip_df with no corresponding annotations in self.df

        Returns: depends on `return_type` argument
            'multihot': [default] returns a dataframe with a column for each class
                    and 0/1 values for class presence.
            'integers': returns a dataframe with 'labels' column containing lists of
                integer class indices for each clip, corresponding to
                the `classes` list; also returns a second value, the list of class names
            'classes': returns a dataframe with 'labels' column containing lists of
                class names for each clip; also returns a second value, the list of class names
            'CategoricalLabels': returns a CategoricalLabels object
        """
        # TODO: implement sparse df for multihot labels

        # check `return_type` argument is valid
        err = f"invalid return_type: {return_type}, must be one of ('multihot','integers','classes','CategoricalLabels')"
        assert return_type in (
            "multihot",
            "integers",
            "classes",
            "CategoricalLabels",
        ), err

        # drop nan annotations
        df = self.df.dropna(subset=["annotation"])  # creates new copy of df object

        if class_subset is None:  # include all annotations
            classes = df["annotation"].unique()
        else:  # the user specified a list of classes
            classes = class_subset
            # subset annotations to user-specified classes,
            # removing rows with annotations for other classes
            df = df[df["annotation"].isin(classes)]

        # if desired, warn users about files with no annotations
        if warn_no_annotations:
            all_files = clip_df.index.get_level_values(0).unique()
            for file in all_files:
                if not file == file:  # file is NaN, get corresponding rows
                    file_df = df[df["audio_file"].isnull()]
                else:  # subset annotations to this file
                    file_df = df[df["audio_file"] == file]

                # warn user if no annotations correspond to this file
                if len(file_df) == 0:
                    warnings.warn(
                        f"No annotations matched the file {file}. "
                        " There will be no positive labels for this file."
                    )

        # initialize an empty dataframe with the same index as clip_df
        output_df = clip_df.copy()

        # how we store labels depends on `multihot` argument, either
        # multi-hot 2d array of 0/1 or lists of integer class indices
        if return_type == "multihot":
            # add columns for each class with 0s. We will add 1s in the loop below
            output_df[classes] = False

            # add the annotations by adding class index positions to appropriate rows
            for class_name in classes:
                # get just the annotations for this class
                class_annotations = df[df["annotation"] == class_name]
                for _, row in class_annotations.iterrows():
                    # find the rows sufficiently overlapped by this annotation, gets the multi-index back
                    df_idxs = find_overlapping_idxs_in_clip_df(
                        file=row["audio_file"],
                        annotation_start=row["start_time"],
                        annotation_end=row["end_time"],
                        clip_df=clip_df,
                        min_label_overlap=min_label_overlap,
                        min_label_fraction=min_label_fraction,
                    )
                    if len(df_idxs) > 0:
                        output_df.loc[df_idxs, class_name] = True
            return output_df
        else:  # create 'labels' column with lists of integer class indices
            output_df["labels"] = [[] for _ in range(len(output_df))]

            # add the annotations by adding the integer class indices to row label lists
            for class_idx, class_name in enumerate(classes):
                # get just the annotations for this class
                class_annotations = df[df["annotation"] == class_name]
                for _, row in class_annotations.iterrows():
                    # find the rows that overlap with the annotation enough in time
                    df_idxs = find_overlapping_idxs_in_clip_df(
                        file=row["audio_file"],
                        annotation_start=row["start_time"],
                        annotation_end=row["end_time"],
                        clip_df=clip_df,
                        min_label_overlap=min_label_overlap,
                        min_label_fraction=min_label_fraction,
                    )

                    for idx in df_idxs:
                        # add the string class name or integer class index to the appropriate rows' labels
                        # if the class name is already in the list, don't add it again
                        # unless keep_duplicates is True
                        if return_type == "classes":
                            if keep_duplicates or (
                                class_name not in output_df.at[idx, "labels"]
                            ):
                                output_df.at[idx, "labels"].append(class_name)
                        else:
                            if keep_duplicates or (
                                class_idx not in output_df.at[idx, "labels"]
                            ):
                                output_df.at[idx, "labels"].append(class_idx)

            if return_type == "CategoricalLabels":
                return CategoricalLabels.from_categorical_labels_df(
                    df=output_df.reset_index(), classes=classes, integer_labels=True
                )
            else:
                return output_df, classes

    def clip_labels(
        self,
        clip_duration,
        min_label_overlap,
        min_label_fraction=None,
        full_duration=None,
        class_subset=None,
        audio_files=None,
        return_type="multihot",
        keep_duplicates=False,
        **kwargs,
    ):
        """Generate one-hot labels for clips of fixed duration

        wraps utils.make_clip_df() with self.labels_on_index()
        - Clips are created in the same way as Audio.split()
        - Labels are applied based on overlap, using self.labels_on_index()

        Args:
            clip_duration (float):  The duration in seconds of the clips
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
            audio_files: list of audio file paths (as str or pathlib.Path)
                to create clips for. If None, uses self.audio_files. [default: None]
            return_type: ('multihot','integers','classes', or 'CategoricalLabels'):
                'multihot': [default] returns a dataframe with a column for each class
                    and 0/1 values for class presence.
                'integers': returns a dataframe with 'labels' column containing lists of
                    integer class indices for each clip, corresponding to
                    the `classes` list; also returns a second value, the list of class names
                'classes': returns a dataframe with 'labels' column containing lists of
                    class names for each clip
                'CategoricalLabels': returns a CategoricalLabels object
            keep_duplicates: [default: False] if True, allows multiple annotations
                of a class to be retained for a single clip; e.g. labels ['a','a','b].
                Ignored if return_type is 'multihot'.
            **kwargs (such as clip_step, final_clip) are passed to
                opensoundscape.utils.generate_clip_times_df() via make_clip_df()
        Returns: depends on `return_type` argument
            'multihot': [default] returns a dataframe with a column for each class
                    and 0/1 values for class presence.
            'integers': returns a dataframe with 'labels' column containing lists of
                integer class indices for each clip, corresponding to
                the `classes` list; also returns a second value, the list of class names
            'classes': returns a dataframe with 'labels' column containing lists of
                class names for each clip; also returns a second value, the list of class names
            'CategoricalLabels': returns a CategoricalLabels object
        """
        # use self.audio_files as list of files to create clips for, unless user passed audio_files
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

        # make unique list, retain order
        audio_files = unique(audio_files)

        # generate list of start and end times for each clip
        # if user passes None for full_duration, try to get the duration from each audio file

        if full_duration is None:
            try:
                clip_df = make_clip_df(
                    files=[f for f in audio_files if f == f],  # remove NaN if present
                    clip_duration=clip_duration,
                    raise_exceptions=True,  # raise exceptions from librosa.duration(f)
                    **kwargs,
                )
            except GetDurationError as exc:
                raise GetDurationError(
                    """`full_duration` was None, but failed to retrieve the durations of 
                    some files. This could occur if the values of 'file' in self.df are 
                    not paths to valid audio files. Specifying `full_duration` as an 
                    argument to `clip_labels()` will avoid the attempt to get 
                    audio file durations from file paths."""
                ) from exc
        else:  # use fixed full_duration for all files
            # make a clip df, will be re-used for each file
            clip_df_template = generate_clip_times_df(
                full_duration=full_duration, clip_duration=clip_duration, **kwargs
            )
            # make a clip df for all files
            clip_df = pd.concat([clip_df_template] * len(audio_files))
            # add file column, repeating value of file across each clip
            clip_df["file"] = np.repeat(audio_files, len(clip_df_template))
            clip_df = clip_df.set_index(["file", "start_time", "end_time"])

        # call labels_on_index with clip_df
        return self.labels_on_index(
            clip_df=clip_df,
            class_subset=class_subset,
            min_label_overlap=min_label_overlap,
            min_label_fraction=min_label_fraction,
            return_type=return_type,
            keep_duplicates=keep_duplicates,
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

        return self._spawn(df=df)

    @classmethod
    def concat(cls, list_of_boxed_annotations):
        """concatenate a list of BoxedAnnotations objects into one"""
        dfs = [ba.df for ba in list_of_boxed_annotations]
        audio_files = []
        annotation_files = []
        for ba in list_of_boxed_annotations:
            if ba.audio_files is not None:
                audio_files.extend(ba.audio_files)
            if ba.annotation_files is not None:
                annotation_files.extend(ba.annotation_files)
        if len(audio_files) == 0:
            audio_files = None
        if len(annotation_files) == 0:
            annotation_files = None
        return cls(
            pd.concat(dfs).reset_index(drop=True),
            audio_files=audio_files,
            annotation_files=annotation_files,
        )

    def train_test_split(self, **kwargs):
        """split annotations into train and test sets

        Splits annotations into train and test sets by audio file, not by row,
        such that all annotations for a given audio file are in either the train
        or test set. This is useful for ensuring that the same audio files are
        not present in both the train and test sets, which could lead to data leakage.

        Note that because self.audio_files is used, this approach retains audio files that
        do not have any annotations (which would not be true if the bounding box .df table
        were split based on the audio file column)

        self.audio_files must be set to the list of annotated audio files

        Args:
            see sklearn.model_selection.train_test_split for arguments
            (test_size, train_size, random_state, shuffle, stratify)
        """
        assert (
            self.audio_files is not None
        ), ".audio_files was None. Please set the .audio_files attribute to use BoxedAnnotations.train_test_split"
        # pair up audio and annotation files
        train_idx, test_idx = train_test_split(range(len(self.audio_files)), **kwargs)
        train_files = np.array(self.audio_files)[train_idx]
        test_files = np.array(self.audio_files)[test_idx]
        train_ann_files = np.array(self.annotation_files)[train_idx]
        test_ann_files = np.array(self.annotation_files)[test_idx]

        # find class dynamically so that this method works if BoxedAnnotations is subclassed
        cls = type(self)
        train_anns = cls(
            self.df[self.df.audio_file.apply(lambda x: x in train_files)].reset_index(),
            annotation_files=train_ann_files,
            audio_files=train_files,
        )

        test_anns = cls(
            self.df[self.df.audio_file.apply(lambda x: x in test_files)].reset_index(),
            annotation_files=test_ann_files,
            audio_files=test_files,
        )

        return train_anns, test_anns


def diff(base_annotations, comparison_annotations):
    """look at differences between two BoxedAnnotations objects
    Not Implemented.

    Compare different labels of the same boxes
    (Assumes that a second annotator used the same boxes as the first,
    but applied new labels to the boxes)
    """
    raise NotImplementedError


def integer_to_multi_hot(labels, n_classes, sparse=False):
    """transform integer labels to multi-hot array

    Args:
        labels: list of lists of integer labels, eg [[0,1,2],[3]]
        n_classes: number of classes
    Returns:
        if sparse is False: 2d np.array with False for absent and True for present
        if sparse is True: scipy.sparse.csr_matrix with 0 for absent and 1 for present
    """
    # TODO: consider using bool rather than int dtype, much smaller and int is unnecessary
    # but bool leads to FutureWarning, see https://github.com/pandas-dev/pandas/issues/59739
    if sparse:
        vals = []
        rows = []
        cols = []
        for i, row in enumerate(labels):
            for col in row:
                vals.append(1)
                rows.append(i)
                cols.append(col)
        return scipy.sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(labels), n_classes), dtype=int
        )
    else:
        multi_hot = np.zeros((len(labels), n_classes), dtype=bool)
        for i, row in enumerate(labels):
            multi_hot[i, row] = True
        return multi_hot


def categorical_to_multi_hot(labels, classes=None, sparse=False):
    """transform multi-target categorical labels (list of lists) to one-hot array

    Args:
        labels: list of lists of categorical labels, eg
            [['white','red'],['green','white']] or [[0,1,2],[3]]
        classes=None: list of classes for one-hot labels. if None,
            taken to be the unique set of values in `labels`
        sparse: bool [default: False] if True, returns a scipy.sparse.csr_matrix
    Returns: tuple (multi_hot, class_subset)
        multi_hot: 2d array with 0 for absent and 1 for present
        class_subset: list of classes corresponding to columns in the array
    """
    if classes is None:
        classes = unique(itertools.chain(*labels))  # retain order

    label_idx_dict = {l: i for i, l in enumerate(classes)}
    vals = []
    rows = []
    cols = []

    # TODO: consider using bool rather than int dtype, much smaller and int is unnecessary
    # but bool leads to FutureWarning, see https://github.com/pandas-dev/pandas/issues/59739
    def add_labels(i, labels):
        for label in labels:
            if label in classes:
                vals.append(1)
                rows.append(i)
                cols.append(label_idx_dict[label])

    [add_labels(i, l) for i, l in enumerate(labels)]

    multi_hot = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(labels), len(classes)), dtype=int
    )

    if sparse:
        return multi_hot, classes
    else:
        return multi_hot.todense(), classes


def multi_hot_to_integer_labels(labels):
    """transform multi-hot (2d array of 0/1) labels to multi-target
    categorical (list of lists of integer class indices)

    Args:
        labels: 2d array or scipy.sparse.csr_matrix with 0 for absent and 1 for present

    Returns:
        list of lists of categorical labels for each sample, eg
            [[0,1,2],[3]] where 0 corresponds to column 0 of labels
    """
    if scipy.sparse.issparse(labels):
        return [
            [col for col in labels.indices[labels.indptr[i] : labels.indptr[i + 1]]]
            for i in range(len(labels.indptr) - 1)
        ]
    else:
        return [list(itertools.compress(range(len(x)), x)) for x in labels]


def categorical_to_integer_labels(labels, classes):
    """
    Convert a list of categorical labels to a list of numeric labels
    """
    # for large lists, dict lookup is 100x faster than list.index()
    classes_dict = {c: i for i, c in enumerate(classes)}
    return [[classes_dict[li] for li in l] for l in labels]


def integer_to_categorical_labels(labels, classes):
    """
    Convert a list of numeric labels to a list of categorical labels
    """
    return [[classes[li] for li in l] for l in labels]


def multi_hot_to_categorical(labels, classes):
    """transform multi-hot (2d array of 0/1) labels to multi-target categorical (list of lists)

    Args:
        labels: 2d array or scipy.sparse.csr_matrix with 0 for absent and 1 for present
        classes: list of classes corresponding to columns in the array

    Returns:
        list of lists of categorical labels for each sample, eg
            [['white','red'],['green','white']] or [[0,1,2],[3]]
    """
    classes = np.array(classes)
    integer_labels = multi_hot_to_integer_labels(labels)
    return integer_to_categorical_labels(integer_labels, classes)

    # if scipy.sparse.issparse(labels):
    #     return [
    #         [
    #             classes[col]
    #             for col in labels.indices[labels.indptr[i] : labels.indptr[i + 1]]
    #         ]
    #         for i in range(len(labels.indptr) - 1)
    #     ]
    # else:
    #     return [list(classes[np.array(row).astype(bool)]) for row in labels]


def _df_to_crowsetta_sequence(df):
    """create a crowsetta.Sequence from BoxedAnnotations style dataframe"""
    return crowsetta.Sequence.from_dict(
        {
            "onset_samples": (
                df["onset_sample"].values if "onset_sample" in df.columns else None
            ),
            "offset_samples": (
                df["offset_samples"].values if "offset_samples" in df.columns else None
            ),
            "onsets_s": df["start_time"].astype(float).values,
            "offsets_s": df["end_time"].astype(float).values,
            "labels": df["annotation"].values,
        }
    )


def _df_to_crowsetta_bboxes(df):
    """create list of crowsetta.BBoxes from BoxedAnnotations style dataframe"""
    # ensure we have a unique index
    df = df.reset_index(drop=True)

    # create crowsetta Annotation object
    return [
        crowsetta.BBox(
            label=df.at[i, "annotation"],
            onset=df.at[i, "start_time"],
            offset=df.at[i, "end_time"],
            low_freq=df.at[i, "low_f"],
            high_freq=df.at[i, "high_f"],
        )
        for i in df.index
    ]


def find_overlapping_idxs_in_clip_df(
    file,
    annotation_start,
    annotation_end,
    clip_df,
    min_label_overlap,
    min_label_fraction=None,
):
    """
    Finds the (file, start_time, end_time) index values for the rows in the clip_df that overlap with the annotation_start and annotation_end
    Args:
        file: audio file path/name the annotation corresponds to; clip_df is filtered to this file
        annotation_start: start time of the annotation
        annotation_end: end time of the annotation
        clip_df: dataframe with multi-index ['file', 'start_time', 'end_time']
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
                is used). A value of 0.5 for this parameter would ensure that all
                annotations result in at least one clip being labeled 1
                (if there are no gaps between clips).
         Returns:
        [(file, start_time, end_time)]) Multi-index values for the rows in the clip_df that overlap with the annotation_start and annotation_end.
    """
    # filter to rows corresponding to this file
    clip_df = clip_df.loc[clip_df.index.get_level_values(0) == file]
    # ignore all rows that start after the annotation ends. Start is level 1 of multi-index
    clip_df = clip_df.loc[clip_df.index.get_level_values(1) < annotation_end]
    # and all rows that end before the annotation starts. End is level 2 of multi-index
    clip_df = clip_df.loc[clip_df.index.get_level_values(2) > annotation_start]
    # now for each time-window, calculate the overlaps
    clip_df["overlap"] = [
        overlap([annotation_start, annotation_end], [row[1], row[2]])
        for row in clip_df.index
    ]

    # discard annotations that do not overlap with the time window
    clip_df = clip_df[clip_df["overlap"] > 0]

    # calculate the fraction of each annotation that overlaps with this time window
    clip_df["overlap_fraction"] = [
        overlap_fraction([annotation_start, annotation_end], [row[1], row[2]])
        for row in clip_df.index
    ]

    # min_label_overlap is a required argument. If min_label_fraction is passed, it means
    # we should keep annotations that have at least least min_label_fraction overlap
    # EVEN if they have less than min_label_overlap seconds of overlap
    if min_label_fraction is None:  # just use min_label_overlap
        clip_df = clip_df[clip_df["overlap"] >= min_label_overlap]
    else:
        # get all rows that are either above min_label_overlap or min_label_fraction
        clip_df = clip_df[
            (clip_df["overlap"] >= min_label_overlap)
            | (clip_df["overlap_fraction"] >= min_label_fraction)
        ]
    return clip_df.index


from itertools import chain


class CategoricalLabels:
    def __init__(
        self, files, start_times, end_times, labels, classes=None, integer_labels=False
    ):
        """
        Store annotations as list of files, start_times, end_times, and labels

        labels are stored as lists of integer class indices, referring to the classes
        in self.classes (list).

        Provides various methods for initializing from and converting to different formats.

        Args:
            files (list): list of file paths
            start_times (list): list of start times (seconds) for each annotation
            end_times (list): list of end times (seconds) for each annotation
            labels (list): list of lists of integer class indices for a file and time range
            classes (list): list of str class names or list of integer class indices
            integer_labels (bool): if True, labels are integer class indices, otherwise labels are class names

        ClassMethods:
            from_categorical_labels_df: create CategoricalLabels object from dataframe with columns
                'file', 'start_time', 'end_time', 'labels' (either integer or class name labels)
            from_multihot_df: create CategoricalLabels object from multi-hot dataframe

        Methods
            multihot_array: generate multi-hot array of labels
            multihot_df: generate multi-hot dataframe of 0/1 labels
            labels_at_index: get list of class names for labels at a specific numeric index
            multihot_labels_at_index: get multi-hot labels at a specific numeric index

        Properties:
            multihot_sparse: sparse 2d scipy.sparse.csr_matrix of multi-hot labels
            multihot_dense: dense 2d array of multi-hot labels
            multihot_df_sparse: sparse dataframe of multi-hot labels
            multihot_df_dense: dense dataframe of multi-hot labels
        """
        # labels can be list of lists of class names or list of lists of integer class indices
        # if classes are not provided, infer them from unique set of labels
        if classes is None:
            classes = unique(chain(*labels))
        self.classes = list(classes)
        # convert from lists of string class names to lists of integer class indices
        if (
            not integer_labels
        ):  # convert from lists of class names to lists of integer class indices
            labels = categorical_to_integer_labels(labels, self.classes)
        self.df = pd.DataFrame(
            {
                "file": files,
                "start_time": start_times,
                "end_time": end_times,
                "labels": labels,
            }
        )

    @classmethod
    def from_categorical_labels_df(cls, df, classes, integer_labels=False):
        """
        df has columns of 'file', 'start_time', 'end_time', 'labels' with labels as list of class names (integer_labels=False)
        or list of integer class indices (integer_labels=True)

        Args:
            df (pd.DataFrame): dataframe with columns of 'file', 'start_time', 'end_time', 'labels'
            classes (list): list of str class names or list of integer class indices
            integer_labels (bool): if True, labels are integer class indices, otherwise labels are class names
        """
        return cls(
            df["file"],
            df["start_time"],
            df["end_time"],
            df["labels"].values,
            classes=classes,
            integer_labels=integer_labels,
        )

    @classmethod
    def from_multihot_df(cls, df):
        """instantiate from dataframe of 0/1 labels across samples & classes

        Args:
            df (pd.DataFrame): dataframe with multi-index of 'file','start_time','end_time';
                columns are class names, values are 0/1 labels
        """
        # multihot df has multi-index of file, start_time, end_time; columns = classes
        labels_int = multi_hot_to_integer_labels(df.values)
        file = df.index.get_level_values("file")
        start_time = df.index.get_level_values("start_time")
        end_time = df.index.get_level_values("end_time")
        return cls(
            file,
            start_time,
            end_time,
            labels_int,
            classes=df.columns.to_list(),
            integer_labels=True,
        )

    def multihot_array(self, sparse=True):
        """generate multi-hot array of labels"""
        cat_classes = integer_to_categorical_labels(
            self.df["labels"].values, self.classes
        )
        sp_labels, _ = categorical_to_multi_hot(
            labels=list(cat_classes),
            classes=self.classes,
            sparse=sparse,
        )
        return sp_labels

    @property
    def labels(self):
        """list of lists of integer class indices (corresponding to self.classes) for each row in self.df"""
        return self.df["labels"].to_list()

    @property
    def class_labels(self):
        """list of lists of class names for each row in self.df"""
        return integer_to_categorical_labels(self.labels, self.classes)

    @property
    def multihot_sparse(self):
        """sparse 2d scipy.sparse.csr_matrix of multi-hot (0/1) labels across self.df.index and self.classes"""
        return self.multihot_array(sparse=True)

    @property
    def multihot_dense(self):
        """2d array of multi-hot (0/1) labels across self.df.index and self.classes"""
        return self.multihot_array(sparse=False)

    def multihot_df(self, sparse=True):
        """generate multi-hot dataframe of 0/1 labels"""
        # multi-index from columns ('file','start_time','end_time')
        index = self._get_multiindex()
        if sparse:
            return pd.DataFrame.sparse.from_spmatrix(
                self.multihot_sparse, index=index, columns=self.classes
            )
        else:
            return pd.DataFrame(self.multihot_dense, index=index, columns=self.classes)

    @property
    def multihot_df_sparse(self):
        """parse dataframe of multi-hot (0/1) labels across self.df.index and self.classes"""
        return self.multihot_df(sparse=True)

    @property
    def multihot_df_dense(self):
        """dataframe of multi-hot (0/1) labels across self.df.index and self.classes"""
        return self.multihot_df(sparse=False)

    def labels_at_index(self, index):
        """list of class names for labels at a specific numeric index"""
        return [self.classes[i] for i in self.df["labels"].iloc[index]]

    def multihot_labels_at_index(self, index):
        """multi-hot (list of 0/1 for self.classes) labels at a specific numeric index"""
        integer_labels = self.df["labels"].iloc[index]
        return integer_to_multi_hot([integer_labels], len(self.classes))[0]

    def _get_multiindex(self):
        """turns self.df columns ['file','start_time','end_time'] into a pandas multi-index"""
        return pd.MultiIndex.from_tuples(
            list(zip(self.df["file"], self.df["start_time"], self.df["end_time"])),
            names=["file", "start_time", "end_time"],
        )


def unique(x):
    """return unique elements of a list, retaining order"""
    return list(dict.fromkeys(x))
