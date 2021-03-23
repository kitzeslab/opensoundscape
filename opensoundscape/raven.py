#!/usr/bin/env python3
""" raven.py: Utilities for dealing with Raven files
"""

from warnings import warn
import pandas as pd
from pathlib import Path
from io import StringIO


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


def annotation_check(directory, col="class"):
    """ Check that rows of Raven annotations files contain class labels

    Args:
        directory:  The path which contains Raven annotations file(s)
        col:        Name of column containing annotations [default: "class"]

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


def generate_class_corrections(directory, col="class"):
    """ Generate a CSV to specify any class overrides

    Args:
        directory:  The path which contains lowercase Raven annotations file(s)
        col:        Name of column containing annotations [default: "class"]

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


def query_annotations(directory, cls, col="class", print_out=False):
    """ Given a directory of Raven annotations, query for a specific class

    Args:
        directory:  The path which contains lowercase Raven annotations file(s)
        cls:        The class which you would like to query for
        col:        Name of column containing annotations [default: "class"]
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


def split_single_annotation(
    raven_path, split_len_s, total_len_s=None, species=None, col="class"
):
    """Split a Raven selection table into short annotations

    Args:
        raven_path (str):       path to Raven selections file
        split_len_s (int):      length of segments to break annotations into (e.g. for 5s: 5)
        total_len_s (float):    length of original file (e.g. for 5-minute file: 300)
                                If not provided, estimates length based on end time of last annotation [default: None]
        species (list):         list of species annotations to look for [default: None]
        col (str):              name of column in Raven file to look for annotations in [default: "class"]

    Returns:
        splits_df (pd.DataFrame): columns 'seg_start', 'end_start', and all species,
            each row containing 1/0 annotations for each species in a segment
    """

    selections_df = pd.read_csv(raven_path, sep="\t")
    if col not in selections_df.columns:
        return

    # If not specified, get total length of annots file (only gets length of last annotation)
    if not total_len_s:
        total_len_s = ceil(
            selections_df["end time (s)"]
            .sort_values(ascending=False)
            .reset_index(drop=True)[0]
        )

    # If not specified, get list of species (only gets species in current file)
    if species == None:
        species = selections_df[col].unique()

    cols = ["seg_start", "seg_end", *species]
    splits_df = pd.DataFrame(columns=cols)

    # Create a dataframe of split_len_s segments and the annotations in each segment
    for start in range(0, total_len_s, split_len_s):
        end = start + split_len_s

        # Annotations in this section
        annots = selections_df[
            (selections_df["end time (s)"] > start)
            & (selections_df["begin time (s)"] < end)
        ]

        segment_df = pd.DataFrame(columns=cols)
        segment_df.loc[0] = [
            start,
            end,
            *list(pd.Series(species).isin(annots[col]).astype(int)),
        ]
        splits_df = splits_df.append(segment_df)

    return splits_df.reset_index(drop=True)


def generate_split_labels_file(
    directory, split_len_s, total_len_s=None, species=None, col="class", out_csv=None
):
    """Generate binary labels for a directory of Raven annotations

    Given a directory of lowercase Raven annotations, splits the annotations into
    segments that can be used as labels for machine learning programs that only
    take short segments.

    Args:
        directory:              The path which contains lowercase Raven annotations file(s)
        split_len_s (int):      length of segments to break annotations into (e.g. for 5s: 5)
        total_len_s (float):    length of original files (e.g. for 5-minute file: 300).
                                If not provided, estimates length individually for each file
                                based on end time of last annotation [default: None]
        species (list):         list of species annotations to look for [default: None]
        col (str):              name of column in Raven file to look for annotations in [default: "class"]
        out_csv (str)           (optional) csv filename to save output at [default: None]

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
        species = []
        for selection in selections:
            selections_df = pd.read_csv(selection, sep="\t")
            if _col_in_df(selections_df, col, selection):
                species.extend(selections_df[col].values)
        species = list(set(species))

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
