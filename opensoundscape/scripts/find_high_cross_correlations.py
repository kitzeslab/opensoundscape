#!/usr/bin/env python3
""" find_high_cross_correlations.py

Help find more matches for training

Prerequisites:
    - You should have run `opensoundscape.py spect_gen -i <ini>`
    - You should have run `opensoundscape.py model_fit -i <ini>`
    - The <ini>'s must match!

Usage:
    find_high_cross_correlations.py <label> [-i <ini>] [-hv]

Positional Arguments:
    <label>                         The label you would like to interrogate,
                                        must be in the CSV defined as `train_file`

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of find_high_cross_correlations.py
    -i --ini <ini>                  Specify an override file [default: opensoundscape.ini]
"""


def high_cc(chunk, species_found, config):
    if len(chunk) != 0:
        return chunk, generate_cross_correlation_matrix(chunk, species_found, config)
    else:
        return chunk, []


from docopt import docopt
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from opensoundscape.config.config import generate_config
from opensoundscape.utils.utils import return_cpu_count
from opensoundscape.utils.db_utils import generate_cross_correlation_matrix
from opensoundscape import __version__ as opso_version


def run():
    # From the docstring, generate the arguments dictionary
    arguments = docopt(
        __doc__, version=f"find_high_cross_correlations.py version {opso_version}"
    )

    # Generate the config instance
    config = generate_config(arguments, store_options=False)

    # Generate list of files which identify <label>
    labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col="Filename",
    )
    labels_df = labels_df.fillna(0).astype(int)

    # Downsample to particular species
    species = arguments["<label>"]

    # Identify files with and without bird
    species_found = labels_df[species][labels_df[species] == 1]
    species_not_found = labels_df[species][labels_df[species] == 0]

    nprocs = return_cpu_count(config)
    executor = ProcessPoolExecutor(nprocs)

    chunk_species_not_found = np.array_split(species_not_found, nprocs)
    chunk_species_found = np.array_split(species_found, nprocs)

    futs = [
        executor.submit(high_cc, chunk, species_found, config)
        for chunk in chunk_species_not_found
    ]
    # fmt: off
    with open("gt9.txt", "w") as gt, \
        open("7-9.txt", "w") as sn, \
        open("4-6.txt", "w") as fs, \
        open("1-3.txt", "w") as ot:
    # fmt: on
        for future in as_completed(futs):
            indices, rows = future.result()
            for idx, row in zip(indices.index.values, rows):
                highest_cc = row.max()
                build_str = np.array_str(row).replace("\n", "")
                if highest_cc > 0.9:
                    gt.write(f"{idx}: {build_str}\n")
                elif highest_cc >= 0.7 and highest_cc <= 0.9:
                    sn.write(f"{idx}: {build_str}\n")
                elif highest_cc >= 0.4 and highest_cc <= 0.6:
                    fs.write(f"{idx}: {build_str}\n")
                elif highest_cc >= 0.1 and highest_cc <= 0.3:
                    ot.write(f"{idx}: {build_str}\n")

    futs = [
        executor.submit(high_cc, chunk, species_found, config)
        for chunk in chunk_species_found
    ]
    with open("template_ccs.txt", "w") as tcc:
        for future in as_completed(futs):
            indices, rows = future.result()
            for idx, row in zip(indices.index.values, rows):
                build_str = np.array_str(row).replace("\n", "")
                tcc.write(f"{idx}: {build_str}\n")
