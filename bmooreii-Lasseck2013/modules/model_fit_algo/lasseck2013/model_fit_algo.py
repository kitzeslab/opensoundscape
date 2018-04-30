import pandas as pd
import numpy as np
from modules.db_utils import read_spectrogram, write_file_stats, return_spectrogram_cursor, cursor_item_to_data
from modules.spect_gen import spect_gen
from modules.view import extract_segments
from scipy import stats
from cv2 import matchTemplate, minMaxLoc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count, Pool
import progressbar
from itertools import repeat


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
    if config.getboolean('db_rw'):
        df, spec, normal = read_spectrogram(label, config)
    else:
        df, spec, normal = spect_gen(label, config)

    # Generate the Raw Spectrogram
    raw_spec = spec * normal

    # Raw Spectrogram Stats
    raw_spec_stats = stats.describe(raw_spec, axis=None)

    # Frequency Band Stats
    freq_bands = np.array_split(raw_spec, config.getint('num_frequency_bands'))
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
    row = np.append(row, df_stats.loc['min'].values)
    row = np.append(row, df_stats.loc['max'].values)
    row = np.append(row, df_stats.loc['mean'].values)
    row = np.append(row, df_stats.loc['std'].values)

    # The row is now a complicated object, need to flatten it
    row = np.ravel(row)

    return df, spec, normal, row


def file_file_stats(df_one, spec_one, normal_one, idx_one, labels_df, config):
    '''Generate the second order statistics

    Given a df, spec, and normal for label_one, generate the file-file statistics.

    Args:
        df_one: The bounding box dataframe for label_one
        spec_one: The spectrum for label_one
        normal_one: The normalization factor for label_one
        idx_one: The index of label_one which generated df_one, spec_one, normal_one
        labels_df: All other labels
        config: The parsed ini configuration

    Returns:
        match_stats_dict: A dictionary, where all keys are indices of df_two
            and values are the template match parameters (max correlation,
            max correlation x value, max correlation y value)

    Raises:
        Nothing.
    '''

    # Extract the segments for matching
    df_one['segments'] = extract_segments(spec_one, df_one)

    # Indices
    indices = labels_df.index[labels_df.index != idx_one]

    # Get the MongoDB Cursor, indices is a Pandas Index object -> list
    if config.getboolean('db_rw'):
        items = return_spectrogram_cursor(indices.values.tolist(), config)
    else:
        items = {'label': x for x in indices}

    # match_stats dimensions are 3 items by number of segments * number of
    # -> files excluding the current image
    match_stats = np.zeros([labels_df.shape[0], df_one.shape[0], 3])

    # Iterate through the cursor
    for item in items:
        idx_two = item['label']
        monotonic_idx_two = labels_df.index.get_loc(idx_two)

        if config.getboolean('db_rw'):
            df_two, spec_two, normal_two = cursor_item_to_data(item, config)
        else:
            df_two, spec_two, normal_two = spect_gen(idx_two, config)

        # Slide segments over all other spectrograms
        frequency_buffer = config.getint('template_match_frequency_buffer')
        for idx, item in df_one.iterrows():
            # Determine minimum y target
            y_min_target = 0
            if item['y_min'] > frequency_buffer:
                y_min_target = item['y_min'] - frequency_buffer

            # Determine maximum y target
            y_max_target = spec_one.shape[0]
            if item['y_max'] < spec_one.shape[0] - frequency_buffer:
                y_max_target = item['y_max'] + frequency_buffer

            # If the template is too large, output_stats should be all zeroes
            # Else:
            # -> Match the template against the stripe of spec_two with the 5th
            # -> algorithm of matchTemplate, then grab the max correllation
            # -> max location x value, and max location y value
            if y_max_target - y_min_target > spec_two.shape[0] or \
                item['x_max'] - item['x_min'] > spec_two.shape[1]:
                match_stats[monotonic_idx_two][idx] = [0, 0, 0]
            else:
                output_stats = matchTemplate(
                    spec_two[y_min_target: y_max_target, :], item['segments'], 5)
                min_val, max_val, min_loc, max_loc = minMaxLoc(output_stats)
                match_stats[monotonic_idx_two][idx][0] = max_val
                match_stats[monotonic_idx_two][idx][1] = max_loc[0]
                match_stats[monotonic_idx_two][idx][2] = max_loc[1] + y_min_target
    return match_stats


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
    birds_identified = labels_df.iloc[monotonic_idx_one][labels_df.iloc[monotonic_idx_one] == 1.0]
    if len(birds_identified) > 0:
        df_one, spec_one, normal_one, row_f = file_stats(idx_one, config)
        match_stats_dict = file_file_stats(df_one, spec_one, normal_one, idx_one, labels_df, config)
        write_file_stats(idx_one, row_f, match_stats_dict, config)


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
    labels_df = pd.read_csv("{}/{}".format(config['data_dir'],
        config['train_file']), index_col=0)

    # Get the processor counts
    if config['num_processors'] == '':
        nprocs = cpu_count()
    else:
        nprocs = config.getint('num_processors')

    # For each file, we need to create a new DF with first and second order
    # statistics
    with progressbar.ProgressBar(max_value=labels_df.shape[0]) as bar:
        with ProcessPoolExecutor(nprocs) as executor:
            for idx, ret in zip(np.arange(labels_df.shape[0]),
                    executor.map(run_stats, labels_df.index,
                        repeat(labels_df), repeat(config))):
                bar.update(idx)

    # Serial code for debugging
    # with progressbar.ProgressBar(max_value=labels_df.shape[0]) as bar:
    #     for idx, item in enumerate(labels_df.index):
    #         run_stats(item, labels_df, config)
    #         bar.update(idx)

    # Now the file stats are available
    # -> Moving to Jupyter Notebook
