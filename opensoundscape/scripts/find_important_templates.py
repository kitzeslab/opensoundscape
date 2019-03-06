#!/usr/bin/env python3
""" find_important_templates.py

Help find important templates for a given label in a dataset. This script will
generate many decision trees which do well on a train/test split labeled dataset
(the `train_file` in INI file) and identify which templates are most important
to the decision tree.

Prerequisites:
    - You should have run `opensoundscape.py spect_gen -i <ini>`
    - The <ini>'s must match!

Usage:
    find_important_templates.py <label> [-i <ini>] [-s <template_pool.csv>] [-hv]

Positional Arguments:
    <label>             The label you would like to interrogate,
                            must be in the CSV defined as `train_file`

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of find_important_templates.py
    -i --ini <ini>                  Specify an override file [default: opensoundscape.ini]
    -s --save <template_pool.csv>   Generate a <template_pool.csv> for opensoundscape.py [default: template_pool.csv]
"""


def build_identification_list(found_df, config):
    # 1. Generate a dictionary
    num_of_segments_d = {}
    items = return_cursor(list(found_df.index), "spectrograms", config)
    for item in items:
        num_of_segments_d[item["label"]] = pd.DataFrame(pickle.loads(item["df"])).shape[
            0
        ]

    # 2. Order the output
    ordered_num_of_segments = [num_of_segments_d[idx] for idx in found_df.index]

    # 3. Generate list-of-list to expand number of segments to file names
    to_filenames = [
        [idx] * n for idx, n in zip(found_df.index, ordered_num_of_segments)
    ]

    # 4. Flatten and tuple up
    return [(idx, item) for sl in to_filenames for idx, item in enumerate(sl)]


def sampled_X_y(species_found, species_not_found, config):
    # Downsample not_found DF to smaller size
    found_length = 2 * species_found.shape[0]
    dummies = species_not_found.sample(n=found_length)

    # Merge the DataFrames
    sampled_df = pd.concat((species_found, dummies))

    # Get the cursor of items
    items = return_cursor(list(sampled_df.index.values), "statistics", config)

    # Generate the file_file_stats
    all_file_file_statistics = [None] * sampled_df.shape[0]
    for idx, item in enumerate(items):
        _, file_file_stats = cursor_item_to_stats(item)
        all_file_file_statistics[idx] = np.vstack(
            [file_file_stats[found] for found in species_found.index.values]
        )

    # Convert all_file_file_statistics to numpy array
    # -> extract the cross correlations only
    all_file_file_statistics = np.array(all_file_file_statistics)[:, :, 0]

    # Generate X and y
    return pd.DataFrame(all_file_file_statistics), pd.Series(sampled_df.values)


def identify_templates(X, y, identifiers):
    # Train/test split w/ stratification
    # -> No random state!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=y
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    good_templates = []
    for _ in range(1000):
        clf = DecisionTreeClassifier()

        params = {"criterion": ["gini"], "min_samples_split": [3], "max_features": [4]}

        # Maximum cross-validation is number of identified species
        max_cv = len([x for x in y_train if x == 1])
        gs = GridSearchCV(clf, params, cv=max_cv, n_jobs=1)
        gs.fit(X_train, y_train)

        y_pred = gs.best_estimator_.predict(X_test)
        if roc_auc_score(y_test, y_pred) >= 0.85:
            feature_importances = gs.best_estimator_.feature_importances_
            important_templates = list(np.where(feature_importances != 0.0)[0])
            good_templates.append(
                sorted(
                    [
                        (feature_importances[x], identifiers[x])
                        for x in important_templates
                    ],
                    reverse=True,
                )
            )

    return good_templates


def gen_results_df(species_found, species_not_found, identifiers_list, config):
    init_client(config)
    X, y = sampled_X_y(species_found, species_not_found, config)
    close_client()
    results_df = pd.DataFrame(columns=["weight", "template"])
    output = identify_templates(X, y, identifiers_list)
    for outer in output:
        for inner in outer:
            weight, template = inner
            results_df = results_df.append(
                {"weight": weight, "template": template}, ignore_index=True
            )
    return results_df


from docopt import docopt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from opensoundscape.config.config import generate_config
from opensoundscape.utils.utils import return_cpu_count
from opensoundscape.utils.db_utils import init_client
from opensoundscape.utils.db_utils import close_client
from opensoundscape.utils.db_utils import return_cursor
from opensoundscape.utils.db_utils import cursor_item_to_stats
from opensoundscape import __version__ as opso_version


def run():
    # From the docstring, generate the arguments dictionary
    arguments = docopt(
        __doc__, version=f"find_important_templates.py version {opso_version}"
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

    # Generate list of tuples with identifying features for templates
    init_client(config)
    identifiers_list = build_identification_list(species_found, config)
    close_client()

    # Now, run a loop to identify useful templates
    # -> Don't use
    nprocs = return_cpu_count(config)
    executor = ProcessPoolExecutor(nprocs)
    fs = [
        executor.submit(
            gen_results_df, species_found, species_not_found, identifiers_list, config
        )
        for x in range(100)
    ]
    # Concatenate all results
    results_df = pd.concat([future.result() for future in as_completed(fs)])

    # Keep any weights above 0.35
    results_df = results_df[results_df["weight"] >= 0.35]

    # Sort by the weights
    results_df.sort_values(by=["weight", "template"], ascending=False, inplace=True)
    templates = list(set(results_df.template.unique()))

    df = pd.DataFrame(templates, columns=["templates", "filenames"])
    gb_filenames = df.groupby("filenames")
    by_filename = [None] * len(gb_filenames.groups.keys())
    for idx, (key, item) in enumerate(gb_filenames):
        by_filename[idx] = (
            key,
            sorted(gb_filenames.get_group(key)["templates"].values),
        )

    df = pd.DataFrame(by_filename, columns=["Filename", "templates"])
    df.to_csv(arguments["--save"], index=False)
