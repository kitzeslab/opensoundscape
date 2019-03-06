import pandas as pd
import numpy as np
from opensoundscape.utils.db_utils import init_client
from opensoundscape.utils.db_utils import close_client
from opensoundscape.utils.db_utils import cursor_item_to_stats
from opensoundscape.utils.db_utils import recall_model
from opensoundscape.utils.db_utils import return_cursor
from opensoundscape.utils.db_utils import write_file_stats
from opensoundscape.utils.utils import return_cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json


class OpensoundscapeModelFitAlgorithmDoesntExist(Exception):
    """
    model_fit algorithm doesn't exist exception
    """

    pass


def chunk_run_stats(chunk, train_labels_df, config):
    """For each chunk call run_stats

    Run within a parallel executor to generate file and file-file Statistics
    for a given label.

    Args:
        chunk: A chunk of file labels
        labels_df: Passed through to `file_file_stats` function
        config: The parsed ini file for this run

    Returns:
        Nothing, writes to MongoDB

    Raises:
        Nothing.
    """

    init_client(config)

    [run_stats(label, train_labels_df, config) for label in chunk]

    close_client()

    return


def run_stats(predict_idx, train_labels_df, config):
    """Wrapper to run parallel statistics generation

    Run within a parallel executor to generate file and file-file Statistics
    for a given label.

    Args:
        predict_idx: The label to make predictions on
        train_labels_df: Labels of the templates to slide over current prediction label
        config: The opensoundscape ini configuration

    Returns:
        Nothing. Writes prediction data to MongoDB collection.
    """

    if config["model_fit"]["algo"] == "template_matching":
        from opensoundscape.model_fit.model_fit_algo.template_matching.model_fit_algo import (
            get_file_stats,
            get_file_file_stats,
        )
    else:
        raise OpensoundscapeModelFitAlgorithmDoesntExist(
            f"The algorithm '{config['model_fit']['algo']}' doesn't exist?"
        )

    df_predict, spec_predict, spec_mean_predict, spec_std_predict, row_predict = get_file_stats(
        predict_idx, config
    )
    match_stats_dict = get_file_file_stats(
        df_predict,
        spec_predict,
        spec_mean_predict,
        spec_std_predict,
        train_labels_df,
        config,
    )
    write_file_stats(predict_idx, row_predict, match_stats_dict, config)


def build_X(predict_df, train_df, config):
    """Build X from predict and train data

    Build X to fit a RandomForestClassifier

    Args:
        predict_df: labels dataframe for prediction
        train_df: labels dataframe for training
        config: The opensoundscape ini configuration

    Returns:
        X: dataframe containing data to fit on

    Raises:
        Nothing.
    """

    if config["model_fit"]["template_pool"]:
        pools_df = pd.read_csv(config["model_fit"]["template_pool"], index_col=0)
        pools_df.templates = pools_df.templates.apply(lambda x: json.loads(x))
        get_file_file_stats_for = [x for x in pools_df.index]
    else:
        get_file_file_stats_for = [x for x in train_df.index if train_df[x] == 1]

    items = return_cursor(predict_df.index.values.tolist(), "statistics", config)

    file_stats = [None] * predict_df.shape[0]
    file_file_stats = [None] * predict_df.shape[0]
    for item in items:
        mono_idx = predict_df.index.get_loc(item["label"])
        file_stats[mono_idx], file_file_stats[mono_idx] = cursor_item_to_stats(item)
        file_file_stats[mono_idx] = [
            file_file_stats[mono_idx][x] for x in get_file_file_stats_for
        ]

    # Shape: (num_files, 80), 80 is number of file statistics
    # -> sometimes garbage data in file_stats
    file_stats = np.nan_to_num(np.array(file_stats))

    # Reshape file_file_stats into 3D numpy array
    # -> (num_files, num_templates, 3), 3 is the number of file-file statistics
    _tmp = [None] * predict_df.shape[0]
    for o_idx, _ in enumerate(file_file_stats):
        _tmp[o_idx] = np.vstack(
            [file_file_stats[o_idx][x] for x in range(len(file_file_stats[o_idx]))]
        )
    file_file_stats = np.array(_tmp)

    # Short circuit return for only cross correlations
    if config["model_fit"].getboolean("cross_correlations_only"):
        return pd.DataFrame(
            file_file_stats[:, :, 0].reshape(file_file_stats.shape[0], -1)
        )

    return pd.DataFrame(
        np.hstack((file_stats, file_file_stats.reshape(file_file_stats.shape[0], -1)))
    )


def chunk_run_prediction(chunk, train_labels_df, predict_labels_df, config):
    """Make a prediction based on a model

    Given a directory of data, make a prediction using a model

    Args:
        chunk: A list of columns to process
        train_labels_df: The training labels to run predictions from
        predict_labels_df: The prediction labels to make predictions on
        config: The parsed ini file for this run

    Returns:
        Nothing. Writes prediction data to MongoDB collection.

    Raises:
        A list of `model.predict_proba` outputs
    """

    init_client(config)

    predict_proba = [
        run_prediction(col, train_labels_df, predict_labels_df, config) for col in chunk
    ]

    close_client()

    return chunk, predict_proba


def run_prediction(column, train_labels_df, predict_labels_df, config):
    """Make a prediction based on a model

    Given a directory of data, make a prediction using a model

    Args:
        column: The label to process
        train_labels_df: The training labels to run predictions from
        predict_labels_df: The prediction labels to make predictions on
        config: The parsed ini file for this run

    Returns:
        Nothing. Writes prediction data to MongoDB collection.

    Raises:
        Nothing
    """

    model, scaler = recall_model(column, config)
    X = build_X(predict_labels_df, train_labels_df[column], config)
    X = scaler.transform(X)

    return model.predict_proba(X)


def predict_algo(config):
    """Make a prediction based on a model

    Given a directory of data, make a prediction using a model

    Args:
        dir: A directory containing data
        config: The parsed ini file for this run

    Returns:
        Nothing. Writes prediction data to MongoDB collection.

    Raises:
        NotImplementedError: Not written yet.
    """
    # First, we need labels and files
    predict_labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['predict_file']}",
        index_col=0,
    )
    train_labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col=0,
    )

    if config["model_fit"]["species_list"] != "":
        train_labels_df = train_labels_df.loc[
            :, config["model_fit"]["species_list"].split(",")
        ]

    nprocs = return_cpu_count(config)
    executor = ProcessPoolExecutor(nprocs)

    # Run the statistics
    chunks = np.array_split(predict_labels_df.index, nprocs)
    fs = [
        executor.submit(chunk_run_stats, chunk, train_labels_df, config)
        for chunk in chunks
    ]
    # This shouldn't be necessary, but doesn't work otherwise...
    for future in as_completed(fs):
        _ = future.result()

    # Create a DF to store the results
    results_df = pd.DataFrame(index=predict_labels_df.index)

    # Run the predictions
    chunks = np.array_split(train_labels_df.columns, nprocs)
    fs = [
        executor.submit(
            chunk_run_prediction, chunk, train_labels_df, predict_labels_df, config
        )
        for chunk in chunks
    ]
    for future in as_completed(fs):
        indices, probas = future.result()
        for idx, proba in zip(indices, probas):
            results_df[idx] = [pred[1] for pred in proba]

    print(results_df.to_csv())
