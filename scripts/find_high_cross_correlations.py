#!/usr/bin/env python
""" find_high_cross_correlations.py

Help find more matches for training

Prerequisites:
    - You should have run `opensoundscape.py spect_gen -i <ini>`
    - You should have run `opensoundscape.py model_fit -i <ini>`
    - The <ini>'s must match!

Usage:
    find_high_cross_correlations.py <label> [-i <ini>] [-s <template_pool.csv>] [-c <chunk_size>] [-hv]

Positional Arguments:
    <label>             The label you would like to interrogate,
                            must be in the CSV defined as `train_file`

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of crc-squeue.py
    -i --ini <ini>                  Specify an override file [default: opensoundscape.ini]
    -s --save <template_pool.csv>   Generate a <template_pool.csv> for opensoundscape.py
"""


def generate_ff_stats(stats_df, species_found_df):
    items = return_cursor(list(stats_df.index.values), "statistics", config)

    # Get all of the file_file_statistics
    # Use the mono_idx to insert the correct value in the correct place!
    all_file_file_statistics = [None] * stats_df.shape[0]
    for item in items:
        mono_idx = stats_df.index.get_loc(item["label"])
        _, file_file_stats = cursor_item_to_stats(item)
        all_file_file_statistics[mono_idx] = [
            file_file_stats[found] for found in species_found_df.index.values
        ]

    # Convert to numpy array and reshape
    # Dims: n_files, n_templates, 1, n_features
    # Only need the zeroth feature
    # Output Dims: n_files, n_templates
    npify = np.array(all_file_file_statistics)
    return npify[:, :, :, 0].reshape(stats_df.shape[0], -1)


def high_cc(chunk, species_found, config):
    if len(chunk) != 0:
        init_client(config)
        all_file_file_statistics = generate_ff_stats(chunk, species_found)
        close_client()
        return chunk, all_file_file_statistics
    else:
        return chunk, []


from docopt import docopt
import pandas as pd
import numpy as np
from copy import copy
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from itertools import repeat

# Need some functions from our module
import sys

script_dir = sys.path[0]
sys.path.insert(0, f"{script_dir}/..")

from modules.utils import generate_config, return_cpu_count
from modules.db_utils import init_client
from modules.db_utils import close_client
from modules.db_utils import cursor_item_to_stats
from modules.db_utils import return_cursor

# From the docstring, generate the arguments dictionary
arguments = docopt(__doc__, version="find_high_cross_correlations.py version 0.0.1")

# Generate the config instance
config = generate_config("config/opensoundscape.ini", arguments["--ini"])

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
wait(futs)

with open("gt9.txt", "w") as gt, open("7-9.txt", "w") as sn, open(
    "4-6.txt", "w"
) as fs, open("1-3.txt", "w") as ot:
    for res in futs:
        indices, rows = res.result()
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
