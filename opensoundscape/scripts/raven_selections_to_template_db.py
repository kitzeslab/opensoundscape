#!/usr/bin/env python3
""" raven_selections_to_template_db.py

Given an openbird.ini file try to find Raven selections file
and build a template pool database


Usage:
    raven_selections_to_template_db.py [-i <ini>] [-hv]

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of raven_selections_to_template_db.py
    -i --ini <ini>                  Specify an override file [default: opensoundscape.ini]
"""


class OpenSoundscapeNoRavenSelectionsFile(Exception):
    pass


def find_closest_index_of(value, array):
    return (np.abs(array - value)).argmin()


from docopt import docopt
import pandas as pd
import numpy as np
import pathlib

from opensoundscape.config.config import generate_config
from opensoundscape.spect_gen.spect_gen_algo.template_matching.spect_gen_algo import (
    return_spectrogram,
)
from opensoundscape.utils.db_utils import init_client
from opensoundscape.utils.db_utils import close_client
from opensoundscape.utils.db_utils import write_spectrogram
from opensoundscape import __version__ as opso_version


def run():
    arguments = docopt(
        __doc__, version=f"raven_selections_to_template_db.py version {opso_version}"
    )

    config = generate_config(arguments, store_options=False)

    # Use a stub labels_df
    labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col="Filename",
    )
    labels_df = labels_df.fillna(0).astype(int)

    data_dir = pathlib.Path(config["general"]["data_dir"])

    rename_dict = {
        "Begin Time (s)": "x_min",
        "End Time (s)": "x_max",
        "Low Freq (Hz)": "y_min",
        "High Freq (Hz)": "y_max",
    }

    with open("template_pool.csv", "w") as f:
        f.write("Filename,templates\n")

    init_client(config)

    for label in labels_df.index.values:
        # Remove file extension from path
        path = pathlib.Path(label)
        label_no_ext = f"{path.parent}/{path.stem}"

        # Make sure there is a selections file
        f_name = list(data_dir.glob(f"{label_no_ext}.*.selections.txt"))
        if len(f_name) == 0:
            raise OpenSoundscapeNoRavenSelectionsFile(
                f"I can't find a selections file for {label}"
            )

        # Read the definitions from the selections file
        # -> Only need the 4 columns and rename them
        # -> Raven prints duplicate rows
        # -> Raven also prints empty boxes
        # -> Rename the columns
        # -> Finally "resample" the frequencies
        df = pd.read_csv(f_name[0], sep="\t")
        df = df[["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]]
        df = df.rename(index=str, columns=rename_dict)
        df.drop_duplicates(inplace=True)
        df = df[df["x_min"] != df["x_max"]].reset_index(drop=True)

        # Write out the template_pool.csv
        with open("template_pool.csv", "a") as f:
            f.write(f'{label},"{list(df.index.values)}"\n')

        # Now we need to create the spectrogram
        spect, spect_mean, spect_std, times, frequencies = return_spectrogram(
            label, config
        )

        # Need to convert the dataframe from units of seconds and Hz to indices
        df["x_min"] = df["x_min"].apply(lambda x: find_closest_index_of(x, times))
        df["x_max"] = df["x_max"].apply(lambda x: find_closest_index_of(x, times))
        df["y_min"] = df["y_min"].apply(lambda x: find_closest_index_of(x, frequencies))
        df["y_max"] = df["y_max"].apply(lambda x: find_closest_index_of(x, frequencies))

        # Store the spectrogram
        write_spectrogram(label, df, spect, spect_mean, spect_std, config)

    # Make sure to close the MongoDB client
    close_client()
