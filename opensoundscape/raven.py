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


def generate_class_corrections(directory, lower=False):
    """ Generate a CSV to specify any class overrides

    Input:
        directory: The path which contains Raven annotations file

    Options:
        lower: Generate class corrections for `**/*.selections.txt.lower`
               files, otherwise `**/*.selections.txt`

    Output:
        csv (string): A multiline string containing a CSV file with two columns
                      `raw` and `corrected`
    """
    input_p = Path(directory)
    if lower:
        selections = input_p.rglob("**/*.selections.txt.lower")
    else:
        selections = input_p.rglob("**/*.selections.txt")

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
