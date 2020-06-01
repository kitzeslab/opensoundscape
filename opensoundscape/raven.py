#!/usr/bin/env python3
""" raven.py: Utilities for dealing with Raven files
"""

from warnings import warn
import pandas as pd
from pathlib import Path
from io import StringIO


def annotation_check(directory):
    """ Check Raven annotations files for a non-null class

    Input:
        directory: The path which contains Raven annotations file

    Output:
        None
    """
    input_p = Path(directory)
    selections = input_p.rglob("**/*.selections.txt")

    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")
        selection_df.columns = selection_df.columns.str.lower()
        if selection_df["class"].isnull().values.any():
            warn(
                f"File `{selection}` is missing a label! Subsequent scripts will use `unknown` if nothing is fixed",
                UserWarning,
            )


def lowercase_annotations(directory):
    """ Convert Raven annotation files to lowercase

    Input:
        directory: The path which contains Raven annotations file

    Output:
        None
    """
    input_p = Path(directory)
    selections = input_p.rglob("**/*.selections.txt")

    for selection in selections:
        lower = f"{selection}.lower"
        with open(selection, "r") as inp, open(lower, "w") as out:
            for line in inp:
                out.write(line.lower())


def generate_class_corrections(directory):
    """ Generate a CSV to specify any class overrides

    Input:
        directory: The path which contains Raven annotations files ending in *.selections.txt.lower

    Output:
        csv (string): A multiline string containing a CSV file with two columns
                      `raw` and `corrected`
    """
    input_p = Path(directory)
    selections = input_p.rglob("**/*.selections.txt.lower")

    class_s = set()
    for selection in selections:
        selection_df = pd.read_csv(selection, sep="\t")
        selection_df["class"] = selection_df["class"].fillna("unknown")
        for cls in selection_df["class"]:
            class_s.add(cls)

    with StringIO() as f:
        f.write("raw,corrected\n")
        for cls in sorted(list(class_s)):
            f.write(f"{cls},{cls}\n")
        return f.getvalue()


def query_annotations(directory, cls):
    """ Given a directory of Raven annotations, query for a specific class

    Input:
        directory:  The path which contains Raven annotations file
        cls:        The class which you would like to query for

    Output:
        output (string): A multiline string containing annotation file and rows matching the query cls
    """

    input_p = Path(directory)
    selections = input_p.rglob("**/*.selections.txt.lower")
    pd.set_option("display.max_rows", None)
    with StringIO() as f:
        for selection in selections:
            selection_df = pd.read_csv(selection, sep="\t")
            num_delimeters = len(selection.name)
            subset = selection_df[selection_df["class"] == cls]
            if subset.shape[0] > 0:
                f.write(f"{'=' * num_delimeters}\n")
                f.write(f"{selection}\n")
                f.write(f"{'=' * num_delimeters}\n")
                f.write(f"{subset}\n")
                f.write(f"{'=' * num_delimeters}\n")
        return f.getvalue()
