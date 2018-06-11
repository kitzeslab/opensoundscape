#!/usr/bin/env python
''' find_important_templates.py

Help find important templates for a given label in a dataset. This script will
generate many decision trees which do well on a train/test split labeled dataset
(the `train_file` in INI file) and identify which templates are most important
to the decision tree.

Prerequisites:
    - You should have run `openbird.py spect_gen -i <ini>`
    - The <ini>'s must match!

Usage:
    find_important_templates.py [-hv]
    find_important_templates.py <label> [-i <ini>] [-s <template_pool.csv>]

Positional Arguments:
    <label>             The label you would like to interrogate,
                            must be in the CSV defined as `train_file`

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of crc-squeue.py
    -i --ini <ini>                  Specify an override file [default: openbird.ini]
    -s --save <template_pool.csv>   Generate a <template_pool.csv> for openbird.py
'''


def build_identification_list(found_df):
    # 1. Generate a dictionary
    num_of_segments_d = {}
    items = return_spectrogram_cursor(list(found_df.index), configs)
    for item in items:
        num_of_segments_d[item['label']] = pd.DataFrame(pickle.loads(item['df'])).shape[0]

    # 2. Order the output
    ordered_num_of_segments = [num_of_segments_d[idx] for idx in found_df.index]

    # 3. Generate list-of-list to expand number of segments to file names
    to_filenames = [[idx] * n for idx, n in zip(found_df.index, ordered_num_of_segments)]

    # 4. Flatten and tuple up
    return [(idx, item) for sl in to_filenames for idx, item in enumerate(sl)]


def sampled_X_y(species_found, species_not_found):
    # Downsample not_found DF to smaller size
    found_length = 2 * species_found.shape[0]
    dummies = species_not_found.sample(n=found_length, random_state=42)

    # Merge the DataFrames
    sampled_df = pd.concat((species_found, dummies))

    # Get the cursor of items
    items = return_spectrogram_cursor(list(sampled_df.index.values), configs)

    # Generate the file_file_stats
    all_file_file_statistics = [None] * sampled_df.shape[0]
    for idx, item in enumerate(items):
        _, file_file_stats = cursor_item_to_stats(item)
        all_file_file_statistics[idx] = [file_file_stats[found] for found in species_found.index.values]

    # Stack internal stats
    # -> convert to NP array
    # -> extract only template matching stat specifically [:, :, 0]
    npify = [None] * sampled_df.shape[0]
    for o_idx, outer in enumerate(all_file_file_statistics):
        stack = np.vstack([all_file_file_statistics[o_idx][x] for x in range(len(all_file_file_statistics[o_idx]))])
        npify[o_idx] = copy(stack)
    all_file_file_statistics = np.array(npify)[:, :, 0]

    # Generate X and y
    return pd.DataFrame(all_file_file_statistics), pd.Series(sampled_df.values)


def identify_templates(X, y, identifiers):
    # Train/test split w/ stratification
    # -> No random state!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    good_templates = []
    for idx in range(1000):
        clf = DecisionTreeClassifier()

        params = {
            "criterion": ["gini"],
            "min_samples_split": [3],
            "max_features": [4]
        }

        # Maximum cross-validation is number of identified species
        max_cv = len([x for x in y_train if x == 1])
        gs = GridSearchCV(clf, params, cv=max_cv, n_jobs=1)
        gs.fit(X_train, y_train)

        y_pred = gs.best_estimator_.predict(X_test)
        if roc_auc_score(y_test, y_pred) >= 0.85:
            feature_importances = gs.best_estimator_.feature_importances_
            important_templates = list(np.where(feature_importances != 0.)[0])
            good_templates.append(sorted([(feature_importances[x], identifiers[x]) for x in important_templates], reverse=True))

    return good_templates


def gen_results_df(idx, species_found, species_not_found, identifiers_list):
    X, y = sampled_X_y(species_found, species_not_found)
    results_df = pd.DataFrame(columns=["weight", "template"])
    output = identify_templates(X, y, identifiers_list)
    for outer in output:
        for inner in outer:
            weight, template = inner
            results_df = results_df.append({"weight": weight, "template": template}, ignore_index=True)
    return results_df


from docopt import docopt
import pandas as pd
import numpy as np
from copy import copy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import progressbar
from itertools import repeat

# Need some functions from our module
import sys
sys.path.append("../modules")

from utils import generate_config
from db_utils import cursor_item_to_stats
from db_utils import return_spectrogram_cursor

# From the docstring, generate the arguments dictionary
arguments = docopt(__doc__, version='find_important_templates.py version 0.0.1')

# Generate the config instance
configs = generate_config('../config/openbird.ini', arguments['--ini'], 'default')

# Generate list of files which identify <label>
labels_df = pd.read_csv("{}/{}".format(configs['data_dir'], configs['train_file']), index_col="Filename")
labels_df = labels_df.fillna(0).astype(int)

# Downsample to particular species
species = arguments["<label>"]

# Identify files with and without bird
species_found = labels_df[species][labels_df[species] == 1]
species_not_found = labels_df[species][labels_df[species] == 0]

# Generate list of tuples with identifying features for templates
identifiers_list = build_identification_list(species_found)

# Now, run a loop to identify useful templates
its = 100
nprocs = cpu_count() - 1
nprocs = min(nprocs, its)
with ProcessPoolExecutor(nprocs) as executor:
    results = executor.map(gen_results_df, range(its), repeat(species_found),
        repeat(species_not_found), repeat(identifiers_list))

# Concatenate all results
results_df = pd.concat(results)

# Keep any weights above 0.35
results_df = results_df[results_df["weight"] >= 0.35]

# Sort by the weights
results_df.sort_values(by=["weight", "template"], ascending=False, inplace=True)
templates = list(set(results_df.template.unique()))

df = pd.DataFrame(templates, columns=["templates", "filenames"])
gb_filenames = df.groupby("filenames")
by_filename = [None] * len(gb_filenames.groups.keys())
for idx, (key, item) in enumerate(gb_filenames):
    by_filename[idx] = (key, sorted(gb_filenames.get_group(key)["templates"].values))

df = pd.DataFrame(by_filename, columns=["Filename", "templates"])
df.to_csv("template_pool.csv", index=False)
