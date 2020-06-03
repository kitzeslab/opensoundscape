#!/usr/bin/env python3
""" Implement data selection utilities
"""

from io import StringIO
from csv import DictReader, DictWriter
from copy import copy


def expand_segments_binary(segments_csv):
    """ Expand multiclass labels from split_audio for binary classification

    Input:
        segments_csv (str): Filename for a segments file with multiclass labels from split_audio

    Output:
        output (str): A string containing the expanded segments
    """

    with open(segments_csv, "r") as inp, StringIO() as sio:
        reader = DictReader(inp)
        writer = DictWriter(sio, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if "|" in row["Labels"]:
                for label in row["Labels"].split("|"):
                    fix_row = copy(row)
                    fix_row["Labels"] = label
                    writer.writerow(fix_row)
            else:
                writer.writerow(row)

        return sio.getvalue()
