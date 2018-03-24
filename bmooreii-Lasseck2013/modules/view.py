import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import pymongo
from modules.spect_gen import spect_gen
from modules.image_utils import apply_gaussian_filter
from modules.db_utils import read_spectrogram

def view(file, config):
    '''View a spectrogram
    
    This is a super function which provides all of the functionality to
    view a spectrogram, either from the database itself or recreated

    Args:
        file: A wav file for template matching
        config: The parsed ini file for this particular run

    Returns:
        Nothing. It writes to MongoDB collection defined in the ini file

    Raises:
        FileNotFoundError: If the wavfile doesn't exist, it can't be processed
    '''

    # Get the data, from the database or recreate
    if config.getboolean('db_readwrite'):
        df, spectrogram, normalization_factor = read_spectrogram(file, config)
    else:
        df, spectrogram, normalization_factor = preprocess_file(file, config)

    # Apply Gaussian Filter
    spectrogram = apply_gaussian_filter(spectrogram,
            config.getfloat('gaussian_filter_sigma'))

    # Generate segments from DF & Spectrogram
    segments = [None] * len(df.index)
    for idx in df.index:
        segments[idx] = spectrogram[df.loc[idx]['y_min']: df.loc[idx]['y_max'],
            df.loc[idx]['x_min']: df.loc[idx]['x_max']]

    # Generate Color Map
    cmap = plt.cm.get_cmap('jet', df.shape[0])
    rgb_vals = cmap(np.arange(df.shape[0]))[:, :-1]

    # Plot, flip the y-axis
    fig, ax = plt.subplots(1, figsize=(15, 5))
    ax.imshow(spectrogram, cmap=plt.get_cmap('gray_r'))
    ax.set_ylim(ax.get_ylim()[::-1])

    for idx, row in df.iterrows():
        rect = patches.Rectangle((row['x_min'], row['y_min']),
                (row['x_max'] - row['x_min'] + 1),
                (row['y_max'] - row['y_min'] + 1),
                linewidth=1, edgecolor=rgb_vals[idx], facecolor='none')
        ax.add_patch(rect)
        ax.text(row['x_min'], row['y_min'] - 2, idx, color=rgb_vals[idx],
                fontsize=12)
    plt.show()
