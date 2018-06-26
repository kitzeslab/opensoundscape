import pandas as pd
import numpy as np
from modules.db_utils import read_spectrogram, write_file_stats, return_spectrogram_cursor, cursor_item_to_data
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


def file_stats(label, config):
    '''Generate the first order statistics

    Given a single label, generate the statistics for the corresponding file

    Args:
        label: A file label from the training set
        config: The parsed ini configuration

    Returns:
        The bounding box DF, spectrogram, and normalization factor for the input
        label

    Raises:
        Nothing.
    '''

    # Generate the df, spectrogram, and normalization factor
    # -> Read from MongoDB or preprocess
    if config['general'].getboolean('db_rw'):
        df, spec, normal = read_spectrogram(label, config)
    else:
        df, spec, normal = spect_gen(label, config)

    # Generate the Raw Spectrogram
    raw_spec = spec * normal

    # Raw Spectrogram Stats
    raw_spec_stats = stats.describe(raw_spec, axis=None)

    # Frequency Band Stats
    freq_bands = np.array_split(raw_spec, config['model_fit'].getint('num_frequency_bands'))
    freq_bands_stats = [None] * len(freq_bands)
    for idx, band in enumerate(freq_bands):
        freq_bands_stats[idx] = stats.describe(band, axis=None)

    # Segment Statistics
    df['width'] = df['x_max'] - df['x_min']
    df['height'] = df['y_max'] - df['y_min']
    df_stats = df[['width', 'height', 'y_min']].describe()

    # Generate the file_row
    # Raw Spectrogram Stats First
    row = np.array([raw_spec_stats.minmax[0], raw_spec_stats.minmax[1],
        raw_spec_stats.mean, raw_spec_stats.variance])

    # Followed by the band statistics
    row = np.append(row, [[s.minmax[0], s.minmax[1], s.mean, s.variance]
        for s in freq_bands_stats])

    # Finally the segment statistics
    # -> If the len(df_stats) == 2, it contains no segments append zeros
    if len(df_stats) == 2:
        row = np.append(row, np.zeros((3, 4)))
    else:
        row = np.append(row, (df_stats.loc['min'].values,
            df_stats.loc['max'].values,
            df_stats.loc['mean'].values,
            df_stats.loc['std'].values))

    # The row is now a complicated object, need to flatten it
    row = np.ravel(row)

    return df, spec, normal, row


def file_file_stats(df_one, spec_one, normal_one, labels_df, config):
    '''Generate the second order statistics

    Given a df, spec, and normal for label_one, generate the file-file statistics.

    Args:
        monotonic_idx_one: The monotonic index for df_one
        df_one: The bounding box dataframe for label_one
        spec_one: The spectrum for label_one
        normal_one: The normalization factor for label_one
        labels_df: All other labels
        config: The parsed ini configuration

    Returns:
        match_stats_dict: A dictionary which contains template matching statistics
         of all segments in labels_df slid over spec_one. Keys are the labels

    Raises:
        Nothing.
    '''

    # Get the MongoDB Cursor, indices is a Pandas Index object -> list
    if config['general'].getboolean('db_rw'):
        items = return_spectrogram_cursor(labels_df.index.values.tolist(), config)
    else:
        items = {'label': x for x in labels_df.index.values.tolist()}

    match_stats_dict = {}

    # Iterate through the cursor
    for item in items:
        # Need to get the index for match_stats
        idx_two = item['label']
        # monotonic_idx_two, = np.where(get_segments_from == idx_two)
        # monotonic_idx_two = monotonic_idx_two[0]

        if config['general'].getboolean('db_rw'):
            df_two, spec_two, normal_two = cursor_item_to_data(item, config)
        else:
            df_two, spec_two, normal_two = spect_gen(idx_two, config)

        spec_two = apply_gaussian_filter(spec_two, config['model_fit']['gaussian_filter_sigma'])

        # Extract segments
        df_two['segments'] = extract_segments(spec_two, df_two)

        # Generate the np.array to append
        match_stats_dict[idx_two] = np.zeros((df_two.shape[0], 3))

        # Slide segments over all other spectrograms
        frequency_buffer = config['model_fit'].getint('template_match_frequency_buffer')
        for idx, item in df_two.iterrows():
            # Determine minimum y target
            y_min_target = 0
            if item['y_min'] > frequency_buffer:
                y_min_target = item['y_min'] - frequency_buffer

            # Determine maximum y target
            y_max_target = spec_two.shape[0]
            if item['y_max'] < spec_two.shape[0] - frequency_buffer:
                y_max_target = item['y_max'] + frequency_buffer

            # If the template is small enough, do the following:
            # -> Match the template against the stripe of spec_one with the 5th
            # -> algorithm of matchTemplate, then grab the max correllation
            # -> max location x value, and max location y value
            if y_max_target - y_min_target <= spec_one.shape[0] and \
                item['x_max'] - item['x_min'] <= spec_one.shape[1]:
                output_stats = matchTemplate(
                    spec_one[y_min_target: y_max_target, :], item['segments'], 5)
                min_val, max_val, min_loc, max_loc = minMaxLoc(output_stats)
                match_stats_dict[idx_two][idx][0] = max_val
                match_stats_dict[idx_two][idx][1] = max_loc[0]
                match_stats_dict[idx_two][idx][2] = max_loc[1] + y_min_target
    return match_stats_dict


def run_stats(idx_one, labels_df, config):
    '''Wrapper for parallel stats execution

    Run within a parallel executor to generate file and file-file Statistics
    for a given label.

    Args:
        idx_one: The label for the file
        labels_df: Passed through to `file_file_stats` function
        config: The parsed ini file for this run

    Returns:
        Nothing, writes to MongoDB

    Raises:
        Nothing.
    '''
    monotonic_idx_one = labels_df.index.get_loc(idx_one)
    df_one, spec_one, normal_one, row_f = file_stats(idx_one, config)
    # Blur spec_one before feeding to file_file_stats
    spec_one = apply_gaussian_filter(spec_one, config['model_fit']['gaussian_filter_sigma'])
    match_stats = file_file_stats(df_one, spec_one, normal_one, labels_df, config)
    write_file_stats(idx_one, row_f, match_stats, config)


def model_fit_algo(config):
    '''Fit the lasseck2013 model

    We were directed here from model_fit to fit the lasseck2013 model.

    Args:
        config: The parsed ini file for this run

    Returns:
        Something or possibly writes to MongoDB

    Raises:
        Nothing.
    '''

    # First, we need labels and files
    labels_df = pd.read_csv("{}/{}".format(config['general']['data_dir'],
        config['general']['train_file']), index_col=0)

    # Get the processor counts
    nprocs = return_cpu_count(config)

    # For each file, we need to create a new DF with first and second order
    # statistics
    with progressbar.ProgressBar(max_value=labels_df.shape[0]) as bar:
        with ProcessPoolExecutor(nprocs) as executor:
            for idx, ret in zip(np.arange(labels_df.shape[0]),
                    executor.map(run_stats, labels_df.index,
                        repeat(labels_df), repeat(config))):
                bar.update(idx)

    # Serial code for debugging
    # print("Running serial code...")
    # for idx, item in enumerate(labels_df.index):
    #     run_stats(item, labels_df, config)

    # Now the file stats are available
    # -> Moving to Jupyter Notebook
