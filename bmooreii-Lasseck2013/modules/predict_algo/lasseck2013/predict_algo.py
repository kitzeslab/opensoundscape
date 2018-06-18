import pandas as pd
import numpy as np
from modules.db_utils import read_spectrogram, write_file_stats, return_spectrogram_cursor, cursor_item_to_data
from modules.spect_gen import spect_gen
from modules.view import extract_segments
from modules.utils import return_cpu_count
from modules.image_utils import gaussian_filter
from scipy import stats
from cv2 import matchTemplate, minMaxLoc
from concurrent.futures import ProcessPoolExecutor
import progressbar
from itertools import repeat
from copy import copy
import sys


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
    try:
        file_stats.__name__
        file_file_stats.__name__
    except UnboundLocalError:
        sys.path.append("modules/model_fit_algo/{}".format(config['predict_algo']))
        from model_fit_algo import file_stats, file_file_stats

    df_predict, spec_predict, normal_predict, row_predict = file_stats(predict_idx, config)
    spec_predict = gaussian_filter(spec_predict, config['gaussian_filter_sigma'])
    match_stats_dict = file_file_stats(df_predict, spec_predict, normal_predict, train_labels_df, config)
    write_file_stats(predict_idx, row_predict, match_stats_dict, config)


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
    predict_labels_df = pd.read_csv("{}/{}".format(config['data_dir'],
        config['predict_file']), index_col=0)
    train_labels_df = pd.read_csv("{}/{}".format(config['data_dir'],
        config['train_file']), index_col=0)

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
    # for idx, item in enumerate(predict_labels_df.index):
    #     run_stats(item, train_labels_df, config)
