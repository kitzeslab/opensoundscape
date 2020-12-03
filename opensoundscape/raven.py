#!/usr/bin/env python3
""" raven.py: Utilities for dealing with Raven files
"""

from warnings import warn
import pandas as pd
from pathlib import Path
from io import StringIO


def annotation_check(directory, col="class"):
    """ Check that rows of Raven annotations files contain class labels

    Input:
        directory:  The path which contains Raven annotations files
        col:        Name of column containing annotations (default: "class")

    Output:
        None
    """
    input_p = Path(directory)
    selections = input_p.rglob("**/*.selections.txt")
    failed = False
    col = col.lower()

    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")

        # Compare lowercase
        selection_df.columns = selection_df.columns.str.lower()

        if col not in selection_df.columns:
            warn(
                f"File `{selection}` is missing the specified column '{col}'",
                UserWarning,
            )
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

    Input:
        directory:  The path which contains Raven annotations file
        out_dir:    The path at which to save (default: save in `directory`, same location as annotations)

    Output:
        None
    """
    input_p = Path(directory)
    if not out_dir:
        output_p = input_p
    else:
        output_p = Path(out_dir)
    selections = input_p.rglob("**/*.selections.txt")

    for selection in selections:
        lower = output_p.joinpath(f"{selection.name}.lower")
        with open(selection, "r") as inp, open(lower, "w") as out:
            for line in inp:
                out.write(line.lower())


def generate_class_corrections(directory, col="class"):
    """ Generate a CSV to specify any class overrides

    Input:
        directory:  The path which contains Raven annotations files ending in *.selections.txt.lower
        col:        Name of column containing annotations (default: "class")

    Output:
        csv (string): A multiline string containing a CSV file with two columns
                      `raw` and `corrected`
    """
    header = "raw,corrected\n"
    input_p = Path(directory)
    selections = list(input_p.rglob("**/*.selections.txt.lower"))
    if len(selections) == 0:
        warn(
            f"Found no `selections.txt.lower` files in folder {str(input_p)}. Make sure to use `lowercase_annotations` first"
        )
        return header
    col = col.lower()

    class_s = set()
    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")
        if col not in selection_df.columns.str.lower():
            warn(
                f"File `{selection}` is missing the specified column '{col}'. Skipping this file",
                UserWarning,
            )
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

    Input:
        directory:  The path which contains Raven annotations file
        cls:        The class which you would like to query for
        col:        Name of column containing annotations (default: "class")
        print_out:  Format of output.
                        If True, output contains delimiters.
                        If False, returns output

    Output:
        output (string): A multiline string containing annotation file and rows matching the query cls
    """

    input_p = Path(directory)
    col = col.lower()
    selections = input_p.rglob("**/*.selections.txt.lower")
    output = {}
    pd.set_option("display.max_rows", None)
    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")
        if col not in selection_df:
            warn(
                f"File `{selection}` is missing the specified column '{col}'. Continuing without searching.",
                UserWarning,
            )
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
