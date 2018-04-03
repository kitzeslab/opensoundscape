import pandas as pd
import numpy as np
from modules.db_utils import read_spectrogram, write_file_stats
from modules.spect_gen import preprocess
from scipy import stats


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
        df, spec, normal = preprocess(label, config)

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
        Something or possibly writes to MongoDB

    Raises:
        Nothing.
    '''
    
    for idx_two, label_two in labels_df['Filename'].drop(idx_one).iteritems():
        # Generate the df, spectrogram, and normalization factor
        # -> Read from MongoDB or preprocess
        if config.getboolean('db_rw'):
            df_two, spec_two, normal_two = read_spectrogram(label_two, config)
        else:
            df_two, spec_two, normal_two = preprocess(label_two, config)

        # Slide segments over all other spectrograms
        # -> Moving to Jupyter Lab for now

    return None


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
        config['train_file']))

    # For each file, we need to create a new DF with first and second order
    # statistics
    for idx_one, label_one in labels_df['Filename'].iteritems():
        df_one, spec_one, normal_one, row_f = file_stats(label_one, config)
        row_ff = file_file_stats(df_one, spec_one, normal_one, idx_one,
                labels_df, config)
