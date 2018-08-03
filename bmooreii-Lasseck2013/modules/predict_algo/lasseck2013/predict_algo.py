import pandas as pd
import numpy as np
from modules.db_utils import cursor_item_to_data
from modules.db_utils import cursor_item_to_stats
from modules.db_utils import read_spectrogram
from modules.db_utils import recall_model
from modules.db_utils import return_cursor
from modules.db_utils import write_file_stats
from modules.spect_gen import spect_gen
from modules.view import extract_segments
from modules.utils import return_cpu_count
from modules.image_utils import apply_gaussian_filter
from scipy import stats
from cv2 import matchTemplate, minMaxLoc
from concurrent.futures import ProcessPoolExecutor
import progressbar
from itertools import repeat
from copy import copy
import sys
import json


def run_stats(predict_idx, train_labels_df, config):
    '''Wrapper to run parallel statistics generation

    Run within a parallel executor to generate file and file-file Statistics
    for a given label.

    Args:
        predict_idx: The label to make predictions on
        train_labels_df: Labels of the templates to slide over current prediction label
        config: The openbird ini configuration

    Returns:
        Nothing. Writes prediction data to MongoDB collection.
    '''

    # Expose the statistics functions necessary for making the prediction
    # -> Can't do this at top of file because it is dependent on the config
    # try:
    #     file_stats.__name__
    #     file_file_stats.__name__
    # except UnboundLocalError:
    sys.path.append(f"modules/model_fit_algo/{config['predict']['algo']}")
    from model_fit_algo import file_stats, file_file_stats

    df_predict, spec_predict, normal_predict, row_predict = file_stats(predict_idx, config)
    spec_predict = apply_gaussian_filter(spec_predict, config['model_fit']['gaussian_filter_sigma'])
    match_stats_dict = file_file_stats(df_predict, spec_predict, normal_predict, train_labels_df, config)
    write_file_stats(predict_idx, row_predict, match_stats_dict, config)


def build_X(predict_df, train_df, config):
    '''Build X from predict and train data

    Build X to fit a RandomForestClassifier

    Args:
        predict_df: labels dataframe for prediction
        train_df: labels dataframe for training
        config: The openbird ini configuration

    Returns:
        X: dataframe containing data to fit on

    Raises:
        Nothing.
    '''

    if config['model_fit']['template_pool']:
        pools_df = pd.read_csv(config['model_fit']['template_pool'], index_col=0)
        pools_df.templates = pools_df.templates.apply(lambda x: json.loads(x))
        get_file_file_stats_for = [x for x in pools_df.index]
    else:
        get_file_file_stats_for = [x for x in train_df.index if train_df[x] == 1]

    items = return_cursor(predict_df.index.values.tolist(), 'statistics', config)

    file_stats = [None] * predict_df.shape[0]
    file_file_stats = [None] * predict_df.shape[0]
    for item in items:
        mono_idx = predict_df.index.get_loc(item['label'])
        file_stats[mono_idx], file_file_stats[mono_idx] = cursor_item_to_stats(item)
        file_file_stats[mono_idx] = [file_file_stats[mono_idx][x] for x in get_file_file_stats_for]

    # Shape: (num_files, 80), 80 is number of file statistics
    # -> sometimes garbage data in file_stats
    file_stats = np.nan_to_num(np.array(file_stats))

    # Reshape file_file_stats into 3D numpy array
    # -> (num_files, num_templates, 3), 3 is the number of file-file statistics
    _tmp = [None] * predict_df.shape[0]
    for o_idx, _ in enumerate(file_file_stats):
        _tmp[o_idx] = np.vstack([file_file_stats[o_idx][x] for x in
            range(len(file_file_stats[o_idx]))])
    file_file_stats = np.array(_tmp)

    return pd.DataFrame(np.hstack(
        (file_stats, file_file_stats.reshape(file_file_stats.shape[0], -1))))


def run_prediction(column, train_labels_df, predict_labels_df, config):
    '''Make a prediction based on a model

    Given a directory of data, make a prediction using a model

    Args:
        dir: A directory containing data
        config: The parsed ini file for this run

    Returns:
        Nothing. Writes prediction data to MongoDB collection.

    Raises:
        NotImplementedError: Not written yet.
    '''

    model, scaler = recall_model(column, config)
    X = build_X(predict_labels_df, train_labels_df[column], config)
    X = scaler.transform(X)

    return (
        f"Class: {column}\n"
        "---\n"
        f"{model.predict_proba(X)}\n"
    )


def predict_algo(config):
    '''Make a prediction based on a model

    Given a directory of data, make a prediction using a model

    Args:
        dir: A directory containing data
        config: The parsed ini file for this run

    Returns:
        Nothing. Writes prediction data to MongoDB collection.

    Raises:
        NotImplementedError: Not written yet.
    '''
    # First, we need labels and files
    predict_labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['predict_file']}",
        index_col=0)
    train_labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col=0)

    # Get the number of processors
    nprocs = return_cpu_count(config)

    # Parallel process
    with progressbar.ProgressBar(max_value=predict_labels_df.shape[0]) as bar:
        with ProcessPoolExecutor(nprocs) as executor:
            for idx, ret in zip(np.arange(predict_labels_df.shape[0]),
                    executor.map(run_stats, predict_labels_df.index,
                        repeat(train_labels_df), repeat(config))):
                bar.update(idx)

    # Serial code for debugging
    # print("Running serial code...")
    # for idx, item in enumerate(predict_labels_df.index):
    #     run_stats(item, train_labels_df, config)

    print("For each class,")
    print("-> format `[no, yes]`, where `no` means probability it is not identified")
    print("-> and no is probability it is identified")
    with progressbar.ProgressBar(max_value=train_labels_df.shape[1]) as bar:
        with ProcessPoolExecutor(nprocs) as executor:
            for idx, ret in zip(np.arange(train_labels_df.shape[1]),
                executor.map(run_prediction, train_labels_df.columns, repeat(train_labels_df),
                    repeat(predict_labels_df), repeat(config))):
                print(ret)
                bar.update(idx)

    # Serial code for debugging
    # print("Running serial code...")
    # for bird in train_labels_df.columns:
    #     print(run_prediction(bird, train_labels_df, predict_labels_df, config))
