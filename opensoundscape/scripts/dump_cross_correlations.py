#!/usr/bin/env python3
""" dump_cross_correlations.py

Simply write the cross correlations to a CSV file

Prerequisites:
    - You should have run `opensoundscape.py spect_gen -i <ini>`
    - You should have run `opensoundscape.py model_fit -i <ini>`
    - The <ini>'s must match!

Usage:
    dump_cross_correlations.py <label> [-i <ini>] [-hv] [-c <csv_name>]

Positional Arguments:
    <label>                         The label you would like to interrogate,
                                        must be in the CSV defined as `train_file`

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of dump_cross_correlations.py
    -i --ini <ini>                  Specify an override file [default: opensoundscape.ini]
    -c --csv <csv_name>             Specify a CSV filename [default: cross_correlations.csv]
"""
from docopt import docopt
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from opensoundscape.utils.db_utils import generate_cross_correlation_matrix
from opensoundscape.config.config import generate_config
from opensoundscape.utils.utils import return_cpu_count
from opensoundscape import __version__ as opso_version


def cross_correlations(chunk, species_found, config):
    if len(chunk) != 0:
        return chunk, generate_cross_correlation_matrix(chunk, species_found, config)
    else:
        return chunk, []


def run():
    arguments = docopt(
        __doc__, version=f"dump_cross_correlations.py version {opso_version}"
    )
    config = generate_config(arguments, store_options=False)

    species = arguments["<label>"]
    labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col="Filename",
    )
    labels_df = labels_df.fillna(0).astype(int)
    species_found = labels_df[species][labels_df[species] == 1]

    nprocs = return_cpu_count(config)
    executor = ProcessPoolExecutor(nprocs)

    chunks = np.array_split(labels_df, nprocs)

    fs = [
        executor.submit(cross_correlations, chunk, species_found, config)
        for chunk in chunks
    ]
    _tmp = [None] * labels_df.shape[0]
    for future in as_completed(fs):
        indices, rows = future.result()
        for idx, row in zip(indices.index.values, rows):
            mono_idx = labels_df.index.get_loc(idx)
            _tmp[mono_idx] = row

    results_df = pd.DataFrame(_tmp, index=labels_df.index.values)
    results_df.to_csv(arguments["--csv"], header=None)
