#!/usr/bin/env python3
""" raven.py: Utilities for dealing with Raven files
"""

from warnings import warn
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from math import ceil
import opensoundscape.audio as audio


def _get_lower_selections(input_p):

    selections = list(Path(input_p).glob("*.selections.txt.lower"))
    if len(selections) == 0:
        raise ValueError(
            f"Found no `selections.txt.lower` files in folder {str(input_p)}. Did you make sure to use `lowercase_annotations()` first?"
        )
    return selections


def _col_in_df(df, col, filename):
    if col not in df.columns:
        warn(f"File `{filename}` is missing the specified column '{col}'", UserWarning)
        return False
    return True


def annotation_check(directory, col):
    """ Check that rows of Raven annotations files contain class labels

    Args:
        directory:  The path which contains Raven annotations file(s)
        col:        Name of column containing annotations

    Returns:
        None
    """
    input_p = Path(directory)
    selections = input_p.rglob("**/*.selections.txt")
    failed = False
    col = col.lower()

    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")
        selection_df.columns = selection_df.columns.str.lower()

        # Compare lowercase
        if not _col_in_df(selection_df, col, filename=selection):
            continue

        if selection_df[col].isnull().values.any():
            failed = True
            warn(
                f"File `{selection}` is missing a label in at least one row. Subsequent scripts will use label 'unknown' if nothing is fixed",
                UserWarning,
            )

    if not failed:
        print(f"All rows in {directory} contain labels in column `{col}`")


def lowercase_annotations(directory, out_dir=None):
    """ Convert Raven annotation files to lowercase and save

    Args:
        directory:  The path which contains Raven annotations file(s)
        out_dir:    The path at which to save (default: save in `directory`, same location as annotations) [default: None]

    Returns:
        None
    """
    input_p = Path(directory)
    if not out_dir:
        output_p = input_p
    else:
        output_p = Path(out_dir)
    selections = list(input_p.rglob("**/*.selections.txt"))
    if len(selections) == 0:
        warn(f"No selections files found in {str(input_p)}", UserWarning)

    for selection in selections:
        lower = output_p.joinpath(f"{selection.name}.lower")
        with open(selection, "r") as inp, open(lower, "w") as out:
            for line in inp:
                out.write(line.lower())


def generate_class_corrections(directory, col):
    """ Generate a CSV to specify any class overrides

    Args:
        directory:  The path which contains lowercase Raven annotations file(s)
        col:        Name of column containing annotations

    Returns:
        csv (string): A multiline string containing a CSV file with two columns
                      `raw` and `corrected`
    """
    header = "raw,corrected\n"
    input_p = Path(directory)

    selections = _get_lower_selections(input_p)
    col = col.lower()

    class_s = set()
    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")

        if not _col_in_df(selection_df, col, filename=selection):
            continue

        selection_df[col] = selection_df[col].fillna("unknown")
        for cls in selection_df[col]:
            class_s.add(cls)

    with StringIO() as f:
        f.write(header)
        for cls in sorted(list(class_s)):
            f.write(f"{cls},{cls}\n")
        return f.getvalue()


def query_annotations(directory, cls, col, print_out=False):
    """ Given a directory of Raven annotations, query for a specific class

    Args:
        directory:  The path which contains lowercase Raven annotations file(s)
        cls:        The class which you would like to query for
        col:        Name of column containing annotations
        print_out:  Format of output.
                        If True, output contains delimiters.
                        If False, returns output
                    [default: False]

    Returns:
        output (string): A multiline string containing annotation file and rows matching the query cls
    """

    input_p = Path(directory)
    col = col.lower()
    selections = _get_lower_selections(input_p)
    output = {}
    pd.set_option("display.max_rows", None)
    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")

        if not _col_in_df(selection_df, col, filename=selection):
            continue

        subset = selection_df[selection_df[col] == cls]
        output[selection] = subset

        num_delimiters = len(str(selection))
        if print_out and subset.shape[0] > 0:
            print(f"{'=' * num_delimiters}")
            print(f"{selection}")
            print(f"{'=' * num_delimiters}\n")
            print(f"{subset}\n")

    return output


def split_starts_ends(raven_file, col, starts, ends, species=None, min_label_len=0):
    """Split Raven annotations using a list of start and end times

    This function takes an array of start times and an array of end times,
    creating a one-hot encoded labels file by finding all Raven labels
    that fall within each start and end time pair.

    This function is called by `split_single_annotation()`, which generates lists
    of start and end times. It is also called by `raven_audio_split_and_save()`,
    which gets the lists from metadata about audio files split by
    opensoundscape.audio.split_and_save.

    Args:
        raven_file (pathlib.Path or str):   path to selections.txt file
        col (str):                          name of column containing annotations
        starts (list):                      start times of clips
        ends (list):                        end times of clips
        species (str or list):              species names for columns of one-hot encoded file [default: None]
        min_label_len (float):              the minimum amount a label must overlap with the split to be considered a label.
                                            Useful for excluding short annotations or annotations that barely overlap the split.
                                            For example, if 1, the label will only be included if the annotation is at least 1s long
                                            and either starts at least 1s before the end of the split, or ends at least 1s
                                            after the start of the split. By default, any label is kept [default: 0]

    Returns:
        splits_df (pd.DataFrame):
            columns: 'seg_start', 'seg_end', and all unique labels ('species')
            rows: one per segment, containing 1/0 annotations for each potential label
    """
    if not len(starts) == len(ends):
        raise ValueError("Arrays of start times and end times must be the same length.")

    selections_df = pd.read_csv(raven_file, sep="\t")
    if col not in selections_df.columns:
        raise ValueError(f"Selections dataframe did not include column {col}")

    # If not specified, get list of species (only gets species in current file)
    if species is None:
        species = selections_df[col].unique()
    elif type(species) == str:
        species = [species]
    species.sort()

    cols = ["seg_start", "seg_end", *species]
    splits_df = pd.DataFrame(columns=cols)

    # Create a dataframe of the annotations in each segment
    dfs = []
    for start, end in zip(starts, ends):

        # Annotations in this section
        annots = selections_df[
            (selections_df["end time (s)"] > start + min_label_len)
            & (selections_df["begin time (s)"] < end - min_label_len)
            & (
                selections_df["end time (s)"] - selections_df["begin time (s)"]
                >= min_label_len
            )
        ]

        segment_df = pd.DataFrame(columns=cols)
        segment_df.loc[0] = [
            start,
            end,
            *list(pd.Series(species).isin(annots[col]).astype(int)),
        ]
        splits_df = splits_df.append(segment_df, ignore_index=True)

    return splits_df


def split_single_annotation(
    raven_file,
    col,
    split_len_s,
    overlap_len_s=0,
    total_len_s=None,
    keep_final=False,
    species=None,
    min_label_len=0,
):
    """Split a Raven selection table into short annotations

    Aggregate one-hot annotations for even-lengthed time segments, drawing
    annotations from a specified column of a Raven selection table

    Args:
        raven_file (str):               path to Raven selections file
        col (str):                      name of column in Raven file to look for annotations in
        split_len_s (float):            length of segments to break annotations into (e.g. for 5s: 5)
        overlap_len_s (float):          length of overlap between segments (e.g. for 2.5s: 2.5)
        total_len_s (float):            length of original file (e.g. for 5-minute file: 300)
                                        If not provided, estimates length based on end time of last annotation [default: None]
        keep_final (string):            whether to keep annotations from the final clip if the final
                                        clip is less than split_len_s long. If using "remainder", "full", "extend", or "loop"
                                        with split_and_save, make this True. Else, make it False. [default: False]
        species (str, list, or None):   species or list of species annotations to look for [default: None]
        min_label_len (float):          the minimum amount a label must overlap with the split to be considered a label.
                                        Useful for excluding short annotations or annotations that barely overlap the split.
                                        For example, if 1, the label will only be included if the annotation is at least 1s long
                                        and either starts at least 1s before the end of the split, or ends at least 1s
                                        after the start of the split. By default, any label is kept [default: 0]
    Returns:
        splits_df (pd.DataFrame): columns 'seg_start', 'seg_end', and all species,
            each row containing 1/0 annotations for each species in a segment
    """

    selections_df = pd.read_csv(raven_file, sep="\t")
    if col not in selections_df.columns:
        return
    if selections_df.shape[0] == 0:
        return pd.DataFrame({"seg_start": [], "seg_end": []})

    # If not specified, get total length of annots file (only gets length of last annotation)
    if not total_len_s:
        total_len_s = ceil(
            selections_df["end time (s)"]
            .sort_values(ascending=False)
            .reset_index(drop=True)[0]
        )

    # If not specified, get list of species (only gets species in current file)
    if species is None:
        species = selections_df[col].unique()

    # Create a dataframe of split_len_s segments and the annotations in each segment
    starts = []
    ends = []
    increment = split_len_s - overlap_len_s
    starts = np.arange(0, total_len_s, increment)
    ends = starts + split_len_s

    if not keep_final:
        # Ignore clip entirely
        keeps = ends <= total_len_s
        ends = ends[keeps]
        starts = starts[keeps]

    return split_starts_ends(
        raven_file=raven_file,
        col=col,
        starts=starts,
        ends=ends,
        species=species,
        min_label_len=min_label_len,
    )


def get_labels_in_dataset(selections_files, col):
    """Get list of all labels in selections_files

    Args:
        selections_files (list):    list of Raven selections.txt files
        col (str):                  the name of the column containing the labels

    Returns:
        a list of the unique values found in the label column of this dataset
    """
    labels = []
    for selection in selections_files:
        selections_df = pd.read_csv(selection, sep="\t")
        if _col_in_df(selections_df, col, selection):
            labels.extend(selections_df[col].values)
    return list(set(labels))


def generate_split_labels_file(
    directory, col, split_len_s, total_len_s=None, species=None, out_csv=None
):
    """Generate binary labels for a directory of Raven annotations

    Given a directory of lowercase Raven annotations, splits the annotations into
    segments that can be used as labels for machine learning programs that only
    take short segments.

    Args:
        directory:                      The path which contains lowercase Raven annotations file(s)
        col (str):                      name of column in Raven file to look for annotations in
        split_len_s (int):              length of segments to break annotations into (e.g. for 5s: 5)
        total_len_s (float):            length of original files (e.g. for 5-minute file: 300).
                                        If not provided, estimates length individually for each file
                                        based on end time of last annotation [default: None]
        species (str, list, or None):   species or list of species annotations to look for [default: None]
        out_csv (str)                   (optional) csv filename to save output at [default: None]

    Returns:
        all_selections (pd.DataFrame): split file of the format
            filename, start_seg, end_seg, species1, species2, ..., speciesN
            orig/fname1, 0, 5, 0, 1, ..., 1
            orig/fname1, 5, 10, 0, 0, ..., 1
            orig/fname2, 0, 5, 1, 1, ..., 1
            ...

        saves all_selections to out_csv if this is specified
    """

    input_p = Path(directory)
    selections = _get_lower_selections(input_p)

    # If list of species not provided, get all species present in dataset
    if not species:
        species = get_labels_in_dataset(selections_files=selections, col=col)

    all_selections = pd.DataFrame()
    for selection in selections:
        selections_df = pd.read_csv(selection, sep="\t")
        if not _col_in_df(selections_df, col, filename=selection):
            continue

        # Split a single annotation file
        ret = split_single_annotation(
            selection,
            split_len_s=split_len_s,
            total_len_s=total_len_s,
            col=col,
            species=species,
        )

        ret.insert(0, "file", selection.stem.split(".")[0])
        all_selections = all_selections.append(ret)

    all_selections = all_selections.reset_index(drop=True)
    if out_csv:
        all_selections.to_csv(out_csv, index=False)

    return all_selections


def raven_audio_split_and_save(
    raven_directory,
    audio_directory,
    destination,
    col,
    sample_rate,
    clip_duration,
    clip_overlap=0,
    final_clip=None,
    extensions=["wav", "WAV", "mp3"],
    csv_name="labels.csv",
    labeled_clips_only=False,
    min_label_len=0,
    species=None,
    dry_run=False,
    verbose=False,
):
    """Split audio and annotations files simultaneously

    Splits audio into short clips with the desired overlap. Saves these clips
    and a one-hot encoded labels CSV into the directory of choice. Labels for
    csv are selected based on all labels in clips.

    Requires that audio and annotation filenames are unique, and that the "stem"
    of annotation filenames is the same as the corresponding stem of the audio
    filename (Raven saves files using this convention by default).

    E.g. The following format is correct:
    audio_directory/audio_file_1.wav
    raven_directory/audio_file_1.Table.1.selections.txt

    Args:
        raven_directory (str or pathlib.Path):  The path which contains lowercase Raven annotations file(s)
        audio_directory (str or pathlib.Path):  The path which contains audio file(s) with names the same as annotation files
        destination (str or pathlib.Path):      The path at which to save the splits and the one-hot encoded labels file
        col (str):                              The column containing species labels in the Raven files
        sample_rate (int):                      Desired sample rate of split audio clips
        clip_duration (float):                  Length of each clip
        clip_overlap (float):                   Amount of overlap between subsequent clips [default: 0]
        final_clip (str or None):               Behavior if final_clip is less than clip_duration seconds long. [default: None]
            By default, ignores final clip entirely.
            Possible options (any other input will ignore the final clip entirely),
                - "full":       Increase the overlap with previous audio to yield a clip with clip_duration length
                - "remainder":  Include the remainder of the Audio (clip will NOT have clip_duration length)
                - "extend":     Similar to remainder but extend the clip with silence to reach clip_duration length
                - "loop":       Similar to remainder but loop (repeat) the clip to reach clip_duration length
        extensions (list):                      List of audio filename extensions to look for. [default: `['wav', 'WAV', 'mp3']`]
        csv_name (str):                         Filename of the output csv, to be saved in the specified destination [default: 'labels.csv']
        min_label_len (float):                  the minimum amount a label must overlap with the split to be considered a label.
                                                Useful for excluding short annotations or annotations that barely overlap the split.
                                                For example, if 1, the label will only be included if the annotation is at least 1s long
                                                and either starts at least 1s before the end of the split, or ends at least 1s
                                                after the start of the split. By default, any label is kept [default: 0]
        labeled_clips_only (bool):              Whether to only save clips that contain labels of the species of interest. [default: False]
        species (str, list, or None):           Species labels to get. If None, gets a list of labels from all selections files. [default: None]
        dry_run (bool):                         If True, skip writing audio and just return clip DataFrame [default: False]
        verbose (bool):                         If True, prints progress information [default:False]

    Returns:
    """

    # List all label files
    all_selections = _get_lower_selections(Path(raven_directory))

    # List all audio files
    audio_directory = Path(audio_directory)
    all_audio = [
        f for f in audio_directory.glob("**/*") if f.suffix.strip(".") in extensions
    ]

    # Get audio files and selection files with same stem
    def _truestem(path_obj):
        return path_obj.name.split(".")[0]

    sel_dict = dict(zip([_truestem(Path(s)) for s in all_selections], all_selections))
    aud_dict = dict(zip([_truestem(Path(a)) for a in all_audio], all_audio))
    keep_keys = set(sel_dict.keys()).intersection(aud_dict.keys())
    keep_keys = list(keep_keys)
    matched_audio = [aud_dict[k] for k in keep_keys]
    matched_selections = [sel_dict[k] for k in keep_keys]
    assert len(matched_audio) == len(matched_selections)

    # Print results for user
    print(
        f"Found {len(matched_audio)} sets of matching audio files and selection tables out of {len(all_audio)} audio files and {len(all_selections)} selection tables"
    )
    if (len(all_audio) - len(matched_audio)) > 0 or (
        len(all_selections) - len(matched_audio) > 0
    ):
        if not verbose:
            print("To see unmatched files, use `verbose = True`")
        else:
            print("Unmatched audio files:")
            print("  " + str(set(all_audio) - set(matched_audio)))
            print("Unmatched selection tables:")
            print("  " + str(set(all_selections) - set(matched_selections)))

    # Get all species in labels file
    if species is None:
        species = get_labels_in_dataset(selections_files=matched_selections, col=col)

    # Create output directory if needed
    destination = Path(destination)
    if not destination.exists():
        if verbose:
            print("Making directory", destination)
        if not dry_run:
            destination.mkdir()

    # If saving labeled clips only, don't split audio on first run
    audio_initial_dry_run = labeled_clips_only | dry_run

    # for each label file:
    # run split_and_save on associated audio
    dfs = []
    for idx, (key, aud_file, sel_file) in enumerate(
        zip(keep_keys, matched_audio, matched_selections)
    ):

        # Split audio and get corresponding start and end times
        a = audio.Audio.from_file(aud_file, sample_rate=sample_rate)
        total_duration = a.duration()
        begin_end_times_df = audio.split_and_save(
            audio=a,
            destination=destination,
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
            final_clip=final_clip,
            dry_run=audio_initial_dry_run,
            prefix=key,
        )

        # Use start and end times to split label file
        df = split_starts_ends(
            raven_file=sel_file,
            col=col,
            starts=begin_end_times_df["begin_time"].values,
            ends=begin_end_times_df["end_time"].values,
            species=species,
            min_label_len=min_label_len,
        )

        # Keep track of clip filenames
        df.index = begin_end_times_df.index

        # For saving only labeled clips:
        if labeled_clips_only:
            df = df[pd.DataFrame(df[species]).sum(axis=1) > 0]
            for clip_name, clip_info in df.iterrows():
                seg_start = clip_info["seg_start"]
                seg_end = clip_info["seg_end"]
                trimmed = a.trim(seg_start, seg_end)
                if seg_end > total_duration:
                    if final_clip == "extend":
                        trimmed.extend(clip_duration)
                    elif final_clip == "loop":
                        trimmed.loop(clip_duration)
                if not dry_run:
                    trimmed.save(clip_name)

        dfs.append(df)

        if verbose:
            print(f"{idx+1}. Finished {aud_file}")

    # Format dataframes as single df with columns filename, label1, label2, ...
    one_hot_encoded_df = pd.concat(dfs)
    one_hot_encoded_df.drop(["seg_start", "seg_end"], axis=1, inplace=True)
    one_hot_encoded_df.index.name = "filename"

    # Save labels file if needed
    if not dry_run:
        one_hot_encoded_df.to_csv(destination.joinpath(csv_name))
    return one_hot_encoded_df
